# -*- coding: utf-8 -*-
"""
Coding Memory Store - 双层物理存储

- SQLite (coding_memory.db)        — coding_memory_meta 表，每条 memory 一行
- VDB (独立 collection)              — 每个 search key 一行向量，由 _vec_collection 名隔离
                                       从 chat collection 派生独立 collection 名

每条 CodingMemory 在 VDB 里有 1 + N 行（task + N 个 search_keys）。
SQLite 里有 1 行（含 search_keys JSON 字段，便于 update 时知道有几行 keys）。

写入流程:
  1. SQLite upsert
  2. embed_batch(task + search_keys)
  3. VDB upsert_batch（每行一个 MemoryNode，custom 字段携带 coding_memory_id 等）

更新流程（覆盖式）:
  1. 拿旧 search_keys 长度 → 删旧 keys 的 VDB node_id
  2. SQLite update
  3. 重新走写入步骤 2-3

删除流程:
  1. 拿 search_keys 长度 → 删全部 VDB node_id
  2. SQLite delete

详见 docs/coding_memory_mvp_design.md §7。
"""

import asyncio
import concurrent.futures
import functools
import json
import logging
import os
import sqlite3
import threading
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..config import MemoryConfig
from ..core.embed_service import EmbedService
from ..data.vector_store import create_vector_store
from ..data.vector_store_base import VectorStoreBase
from ..models.memory import MemoryLayer, MemoryNode, MemoryStatus
from .types import BoundaryScope, CodingMemory

logger = logging.getLogger(__name__)


# ================================================================
# SQLite 线程池
# ================================================================

_CODING_SQLITE_POOL_SIZE = 16
_coding_sqlite_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_CODING_SQLITE_POOL_SIZE,
    thread_name_prefix="coding-sqlite",
)


def _run_in_pool(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    if args or kwargs:
        return loop.run_in_executor(
            _coding_sqlite_executor, functools.partial(func, *args, **kwargs)
        )
    return loop.run_in_executor(_coding_sqlite_executor, func)


# ================================================================
# DDL
# ================================================================

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS coding_memory_meta (
    memory_id        TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    agent_id         TEXT NOT NULL DEFAULT 'default_agent',
    task             TEXT NOT NULL,
    search_keys      TEXT NOT NULL DEFAULT '[]',
    solution         TEXT NOT NULL,
    boundary_envs    TEXT NOT NULL DEFAULT '',
    boundary_scope   TEXT NOT NULL,
    workspace_id     TEXT,
    branch           TEXT,
    session_id       TEXT,
    files            TEXT NOT NULL DEFAULT '[]',
    confidence       REAL NOT NULL DEFAULT 0.7,
    source           TEXT NOT NULL DEFAULT 'auto_extract',
    type             TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
"""

_CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_coding_user ON coding_memory_meta(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_coding_user_workspace ON coding_memory_meta(user_id, workspace_id);",
    "CREATE INDEX IF NOT EXISTS idx_coding_updated ON coding_memory_meta(updated_at DESC);",
]


# ================================================================
# Helpers
# ================================================================

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _vec_node_id(memory_id: str, key_idx: int) -> str:
    return f"{memory_id}:{key_idx}"


def _row_to_memory(row: sqlite3.Row) -> CodingMemory:
    """SQLite row → CodingMemory"""
    def _parse_dt(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            # 去掉末尾 Z
            return datetime.fromisoformat(s.rstrip("Z"))
        except Exception:
            return None

    return CodingMemory(
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        task=row["task"],
        search_keys=json.loads(row["search_keys"] or "[]"),
        solution=row["solution"],
        boundary_envs=row["boundary_envs"],
        boundary_scope=row["boundary_scope"],
        workspace_id=row["workspace_id"],
        branch=row["branch"],
        session_id=row["session_id"],
        files=json.loads(row["files"] or "[]"),
        confidence=row["confidence"],
        source=row["source"],
        type=row["type"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _derive_coding_collection_name(base: str) -> str:
    """从 chat collection 派生 coding_keys collection。"""
    base = (base or "agent_memories").strip()
    # 维度后缀由 VectorStoreBase 自动拼接，这里只在 base_name 上加 _coding_keys
    return f"{base}_coding_keys"


def _clone_config_for_coding(config: MemoryConfig) -> MemoryConfig:
    """深拷贝 config 并改 collection_name 为派生的 coding_keys 名。"""
    cfg = deepcopy(config)
    base = cfg.vector_store.collection_name or "agent_memories"
    cfg.vector_store.collection_name = _derive_coding_collection_name(base)
    return cfg


# ================================================================
# Store
# ================================================================

class CodingMemoryStore:
    """
    Coding Memory 双层存储。

    必须先 await initialize() 再用。close() 释放资源。
    """

    def __init__(
        self,
        config: MemoryConfig,
        embed_service: EmbedService,
        db_path: Optional[str] = None,
    ):
        self.config = config
        self.embed_service = embed_service

        # SQLite 路径
        if db_path is None:
            data_dir = os.path.dirname(config.history.db_path) if config.history.db_path else "./data"
            db_path = os.path.join(data_dir, "coding_memory.db")
        self._db_path = db_path

        # SQLite 连接（与现有 cache_sqlite 风格一致）
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

        # VDB 独立 collection
        self._coding_config = _clone_config_for_coding(config)
        self._vec: Optional[VectorStoreBase] = None

        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        # SQLite
        await _run_in_pool(self._init_sqlite_sync)
        # VDB
        self._vec = create_vector_store(self._coding_config)
        await self._vec.initialize()
        self._initialized = True
        logger.info(
            f"[coding-store] initialized: sqlite={self._db_path} "
            f"vdb={self._coding_config.vector_store.collection_name}"
        )

    def _init_sqlite_sync(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_CREATE_TABLE_SQL)
        for sql in _CREATE_INDEX_SQL:
            self._conn.execute(sql)

    async def close(self) -> None:
        if self._vec:
            try:
                await self._vec.close()
            except Exception:
                pass
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
        self._initialized = False

    # ------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------

    async def insert(self, memory: CodingMemory) -> None:
        """新增一条 coding memory（SQLite + VDB 全套）"""
        self._fill_timestamps(memory)
        await _run_in_pool(self._upsert_sql_sync, memory)
        await self._upsert_vec_keys(memory)
        logger.info(
            f"[coding-store] inserted memory_id={memory.memory_id} "
            f"task={memory.task!r} keys={1 + len(memory.search_keys)}"
        )

    async def update(self, memory: CodingMemory) -> None:
        """覆盖式更新一条 coding memory。"""
        # 拿旧 keys 长度，先删旧 VDB rows
        prev = await self.get_by_id(memory.memory_id, user_id=memory.user_id)
        prev_key_count = (1 + len(prev.search_keys)) if prev else 0

        memory.updated_at = datetime.utcnow()
        if memory.created_at is None and prev is not None:
            memory.created_at = prev.created_at
        await _run_in_pool(self._upsert_sql_sync, memory)
        # 删旧 VDB
        await self._delete_vec_keys_by_count(memory.memory_id, prev_key_count)
        # 写新 VDB
        await self._upsert_vec_keys(memory)
        logger.info(
            f"[coding-store] updated memory_id={memory.memory_id} "
            f"old_keys={prev_key_count} new_keys={1 + len(memory.search_keys)}"
        )

    async def delete(self, memory_id: str, user_id: str) -> bool:
        """删除一条 coding memory，返回是否真有删到。"""
        prev = await self.get_by_id(memory_id, user_id=user_id)
        if prev is None:
            logger.info(f"[coding-store] delete skipped (not found): {memory_id}")
            return False
        await self._delete_vec_keys_by_count(memory_id, 1 + len(prev.search_keys))
        await _run_in_pool(self._delete_sql_sync, memory_id, user_id)
        logger.info(f"[coding-store] deleted memory_id={memory_id}")
        return True

    # ------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------

    async def get_by_id(
        self, memory_id: str, user_id: Optional[str] = None
    ) -> Optional[CodingMemory]:
        return await _run_in_pool(self._get_by_id_sync, memory_id, user_id)

    async def get_many(
        self, memory_ids: List[str], user_id: Optional[str] = None
    ) -> List[CodingMemory]:
        if not memory_ids:
            return []
        return await _run_in_pool(self._get_many_sync, memory_ids, user_id)

    async def list_user_tasks(
        self, user_id: str, limit: int = 50
    ) -> List[str]:
        """返回该 user 已有的 task 列表（最近修改的优先）。供 extractor prompt 引用。"""
        return await _run_in_pool(self._list_user_tasks_sync, user_id, limit)

    async def list_by_workspace(
        self,
        user_id: str,
        workspace_id: str,
        branch: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List["CodingMemory"]:
        """按 workspace_id（+可选 branch）纯过滤列出该 user 的完整 coding memory。

        无语义排序，按 updated_at 倒序（最近修改优先）。branch 为空则只按
        workspace 过滤；非空则叠加 branch 精确过滤。
        """
        return await _run_in_pool(
            self._list_by_workspace_sync, user_id, workspace_id, branch, limit, offset
        )

    def _list_by_workspace_sync(
        self, user_id: str, workspace_id: str, branch: Optional[str],
        limit: int, offset: int,
    ) -> List["CodingMemory"]:
        with self._lock:
            assert self._conn is not None
            sql = ("SELECT * FROM coding_memory_meta "
                   "WHERE user_id = ? AND workspace_id = ?")
            params: List[Any] = [user_id, workspace_id]
            if branch:
                sql += " AND branch = ?"
                params.append(branch)
            sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            cur = self._conn.execute(sql, params)
            return [_row_to_memory(r) for r in cur.fetchall()]

    # ------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------

    async def search_by_query_embedding(
        self,
        query_embedding: List[float],
        user_id: str,
        *,
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        top: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        向量检索。返回 dedup 后按分数降序的 hit list：
            [
                {
                    "memory_id": "...",
                    "score": 0.83,
                    "matched_key_kind": "task" | "aspect",
                    "matched_key_text": "...",
                },
                ...
            ]
        boundary 过滤在客户端做（拉 top*N → 客户端 filter → dedup）。
        """
        if self._vec is None:
            return []

        raw = await self._vec.search(
            query_embedding=query_embedding,
            user_id=user_id,
            limit=top,
            score_threshold=0.0,
            only_latest=True,
        )

        best: Dict[str, Dict[str, Any]] = {}
        for hit in raw:
            node: MemoryNode = hit.get("node")
            score: float = hit.get("score", 0.0)
            if node is None:
                continue
            custom = node.custom or {}
            mid = custom.get("coding_memory_id")
            if not mid:
                continue
            scope = custom.get("coding_boundary_scope")
            ws = custom.get("coding_workspace_id")
            br = custom.get("coding_branch")
            # boundary filter（客户端实施）
            if not _passes_boundary(
                scope=scope,
                memory_workspace=ws,
                memory_branch=br,
                ctx_workspace=workspace_id,
                ctx_branch=branch,
            ):
                continue
            # dedup by memory_id, 保留最高分
            cur = best.get(mid)
            if cur is None or score > cur["score"]:
                best[mid] = {
                    "memory_id": mid,
                    "score": score,
                    "matched_key_kind": custom.get("coding_key_kind", "aspect"),
                    "matched_key_text": custom.get("coding_key_text") or node.content,
                }
        # 排序
        return sorted(best.values(), key=lambda h: -h["score"])

    # ------------------------------------------------------------
    # SQLite 同步实现
    # ------------------------------------------------------------

    def _fill_timestamps(self, memory: CodingMemory) -> None:
        now = datetime.utcnow()
        if memory.created_at is None:
            memory.created_at = now
        memory.updated_at = now

    def _upsert_sql_sync(self, memory: CodingMemory) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                """
                INSERT INTO coding_memory_meta (
                    memory_id, user_id, agent_id, task, search_keys, solution,
                    boundary_envs, boundary_scope, workspace_id, branch, session_id,
                    files, confidence, source, type, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    user_id        = excluded.user_id,
                    agent_id       = excluded.agent_id,
                    task           = excluded.task,
                    search_keys    = excluded.search_keys,
                    solution       = excluded.solution,
                    boundary_envs  = excluded.boundary_envs,
                    boundary_scope = excluded.boundary_scope,
                    workspace_id   = excluded.workspace_id,
                    branch         = excluded.branch,
                    session_id     = excluded.session_id,
                    files          = excluded.files,
                    confidence     = excluded.confidence,
                    source         = excluded.source,
                    type           = excluded.type,
                    updated_at     = excluded.updated_at
                """,
                (
                    memory.memory_id, memory.user_id, memory.agent_id,
                    memory.task, json.dumps(memory.search_keys, ensure_ascii=False),
                    memory.solution, memory.boundary_envs, memory.boundary_scope,
                    memory.workspace_id, memory.branch, memory.session_id,
                    json.dumps(memory.files, ensure_ascii=False),
                    memory.confidence, memory.source, memory.type,
                    (memory.created_at or datetime.utcnow()).isoformat(),
                    (memory.updated_at or datetime.utcnow()).isoformat(),
                ),
            )

    def _delete_sql_sync(self, memory_id: str, user_id: str) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "DELETE FROM coding_memory_meta WHERE memory_id = ? AND user_id = ?",
                (memory_id, user_id),
            )

    def _get_by_id_sync(
        self, memory_id: str, user_id: Optional[str]
    ) -> Optional[CodingMemory]:
        with self._lock:
            assert self._conn is not None
            if user_id:
                cur = self._conn.execute(
                    "SELECT * FROM coding_memory_meta WHERE memory_id = ? AND user_id = ?",
                    (memory_id, user_id),
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM coding_memory_meta WHERE memory_id = ?",
                    (memory_id,),
                )
            row = cur.fetchone()
            return _row_to_memory(row) if row else None

    def _get_many_sync(
        self, memory_ids: List[str], user_id: Optional[str]
    ) -> List[CodingMemory]:
        if not memory_ids:
            return []
        with self._lock:
            assert self._conn is not None
            placeholders = ",".join("?" * len(memory_ids))
            if user_id:
                sql = f"SELECT * FROM coding_memory_meta WHERE memory_id IN ({placeholders}) AND user_id = ?"
                params = list(memory_ids) + [user_id]
            else:
                sql = f"SELECT * FROM coding_memory_meta WHERE memory_id IN ({placeholders})"
                params = list(memory_ids)
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            # 保留输入顺序
            by_id = {r["memory_id"]: _row_to_memory(r) for r in rows}
            return [by_id[mid] for mid in memory_ids if mid in by_id]

    def _list_user_tasks_sync(self, user_id: str, limit: int) -> List[str]:
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "SELECT task FROM coding_memory_meta WHERE user_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [r["task"] for r in cur.fetchall()]

    # ------------------------------------------------------------
    # 列出该 user 已有 memory 的元数据（curator agent 用）
    # 注意：这是新增的 read-only 方法，不影响 legacy 路径。
    # ------------------------------------------------------------

    async def list_user_memories_metadata(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        返回该 user 已有 memory 的 metadata：memory_id / task / boundary_scope /
        confidence / workspace_id / branch（不含 solution / search_keys / files 详情）。
        给 CodingCurator agent 的初始 prompt 用，详情通过 read_existing_memory 取。

        最近修改的优先。
        """
        return await _run_in_pool(self._list_user_memories_metadata_sync, user_id, limit)

    def _list_user_memories_metadata_sync(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "SELECT memory_id, task, boundary_scope, confidence, workspace_id, branch "
                "FROM coding_memory_meta WHERE user_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [
                {
                    "memory_id": r["memory_id"],
                    "task": r["task"],
                    "boundary_scope": r["boundary_scope"],
                    "confidence": r["confidence"],
                    "workspace_id": r["workspace_id"],
                    "branch": r["branch"],
                }
                for r in cur.fetchall()
            ]

    # ------------------------------------------------------------
    # VDB 实现
    # ------------------------------------------------------------

    async def _upsert_vec_keys(self, memory: CodingMemory) -> None:
        """把 task + search_keys 写入 VDB（每行一个 MemoryNode）。"""
        if self._vec is None:
            return

        all_keys: List[Tuple[str, str]] = [("task", memory.task)]
        for k in memory.search_keys:
            if k:
                all_keys.append(("aspect", k))
        if not all_keys:
            return

        # batch embed
        texts = [t for _, t in all_keys]
        try:
            embeddings = await self.embed_service.embed_batch(texts)
        except Exception as e:
            logger.warning(f"[coding-store] embed_batch failed: {e}")
            return

        nodes: List[MemoryNode] = []
        for i, ((kind, text), emb) in enumerate(zip(all_keys, embeddings)):
            node = MemoryNode(
                node_id=_vec_node_id(memory.memory_id, i),
                user_id=memory.user_id,
                agent_id=memory.agent_id,
                session_id=memory.session_id or "",
                # layer 字段不参与 chat 路径过滤（独立 collection），用 L1_RAW 占位
                layer=MemoryLayer.L1_RAW,
                content=text,
                embedding=emb,
                status=MemoryStatus.ACTIVE,
                custom={
                    "coding_memory_id": memory.memory_id,
                    "coding_key_kind": kind,
                    "coding_key_text": text,
                    "coding_boundary_scope": memory.boundary_scope,
                    "coding_workspace_id": memory.workspace_id or "",
                    "coding_branch": memory.branch or "",
                    "coding_kind_marker": "coding_key",
                },
            )
            nodes.append(node)

        try:
            await self._vec.upsert_batch(nodes)
        except Exception as e:
            logger.warning(f"[coding-store] vdb upsert_batch failed: {e}")

    async def _delete_vec_keys_by_count(self, memory_id: str, count: int) -> None:
        """按 deterministic node_id 删除 0..count-1。"""
        if self._vec is None or count <= 0:
            return
        for i in range(count):
            try:
                await self._vec.delete(_vec_node_id(memory_id, i))
            except Exception as e:
                logger.debug(
                    f"[coding-store] delete vdb {_vec_node_id(memory_id, i)} ignored: {e}"
                )


# ================================================================
# Boundary filter（客户端实施）
# ================================================================

def _passes_boundary(
    scope: Optional[str],
    memory_workspace: Optional[str],
    memory_branch: Optional[str],
    ctx_workspace: Optional[str],
    ctx_branch: Optional[str],
) -> bool:
    """
    判断一条 coding memory 在当前查询上下文下是否可见。
    详见 docs §4.3 / §8.5。
    """
    s = (scope or "").strip().lower()
    if s == "user":
        return True
    if s == "global":
        return True
    if s == "project":
        return bool(ctx_workspace) and (memory_workspace or "") == ctx_workspace
    if s == "strict":
        return (
            bool(ctx_workspace)
            and (memory_workspace or "") == ctx_workspace
            and bool(ctx_branch)
            and (memory_branch or "") == ctx_branch
        )
    # 未知 scope 默认不命中（保守）
    return False
