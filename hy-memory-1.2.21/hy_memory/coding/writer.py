# -*- coding: utf-8 -*-
"""
CodingWriter - 写入端协调器

把 judge / preproc / extractor / reconciler / store 串成一条端到端流程：

    classify_messages_is_coding(messages, llm)
        ├── False → caller 应该把 messages strip_tool_messages 后走 chat 链
        └── True
             ├── truncate_messages(messages)  ← 截断长 tool_result
             ├── extract_files(messages)       ← 抽 file paths
             ├── extractor.extract(...)         ← LLM 产出 drafts
             └── reconciler.reconcile(drafts)   ← LLM 决策 + store 持久化

详见 docs/coding_memory_mvp_design.md §3 / §6.1。
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..agent.llm_provider import LLMProvider
from ..core.embed_service import EmbedService
from ..pipelines.base import ChatMessage
from .extractor import CodingMemoryExtractor
from .judge import classify_messages_is_coding
from .preproc import has_any_tool_message, truncate_messages
from .reconciler import CodingMemoryReconciler
from .store import CodingMemoryStore
from .types import ReconcileOp

logger = logging.getLogger(__name__)


class CodingWriter:
    """
    Coding 写入端协调器。

    生命周期：依赖外部传入的 store / extractor / reconciler；不负责 initialize/close。
    """

    def __init__(
        self,
        store: CodingMemoryStore,
        extractor: CodingMemoryExtractor,
        reconciler: CodingMemoryReconciler,
        llm_provider: LLMProvider,
        embed_service: EmbedService,
    ):
        self.store = store
        self.extractor = extractor
        self.reconciler = reconciler
        self.llm_provider = llm_provider
        self.embed_service = embed_service

    async def write(
        self,
        messages: List[ChatMessage],
        *,
        user_id: str,
        agent_id: str = "default_agent",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        session_id: Optional[str] = None,
        existing_tasks_limit: int = 30,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        coding 段写入端到端。返回响应 dict（兼容 client.add 返回结构）：
            {
                "success": True,
                "scene": "productivity",
                "request_id": "...",
                "elapsed_ms": ...,
                "ops": [
                    {"action": "ADD", "draft_idx": 0, "target_memory_id": "...", "reason": ...},
                    ...
                ],
                "memory_ids": ["..."],   # 所有 ADD/UPDATE 涉及的 memory_id
            }
        """
        request_id = request_id or str(uuid.uuid4())
        t0 = time.perf_counter()

        if not messages:
            return self._empty_response(request_id, t0)

        # 1. 截断长 tool_result（防御；judge 已经做过简化视图）
        truncated = truncate_messages(messages)

        # 2. 拉用户已有 task 列表给 extractor 参考
        try:
            existing_tasks = await self.store.list_user_tasks(
                user_id, limit=existing_tasks_limit
            )
        except Exception as e:
            logger.warning(f"[coding-write] list_user_tasks failed: {e}")
            existing_tasks = []

        # 3. 抽取 drafts
        drafts = await self.extractor.extract(
            truncated,
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            branch=branch,
            session_id=session_id,
            existing_tasks=existing_tasks,
        )

        if not drafts:
            logger.info(
                f"[coding-write] no drafts produced (value bar / boundary guard); "
                f"request_id={request_id}"
            )
            return {
                "success": True,
                "scene": "productivity",
                "request_id": request_id,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                "ops": [],
                "ops_summary": {"add": 0, "update": 0, "delete": 0, "skip": 0, "total": 0},
                "memory_ids": [],
            }

        # 4. Reconcile + 持久化
        ops: List[ReconcileOp] = await self.reconciler.reconcile(drafts)

        memory_ids = [op.target_memory_id for op in ops
                      if op.action in ("ADD", "UPDATE") and op.target_memory_id]

        add_cnt = sum(1 for op in ops if op.action == "ADD")
        update_cnt = sum(1 for op in ops if op.action == "UPDATE")
        delete_cnt = sum(1 for op in ops if op.action == "DELETE")
        skip_cnt = sum(1 for op in ops if op.action == "SKIP")
        ops_summary = {
            "add": add_cnt,
            "update": update_cnt,
            "delete": delete_cnt,
            "skip": skip_cnt,
            "total": add_cnt + update_cnt + delete_cnt + skip_cnt,
        }

        return {
            "success": True,
            "scene": "productivity",
            "request_id": request_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "ops": [op.to_dict() for op in ops],
            "ops_summary": ops_summary,
            "memory_ids": memory_ids,
        }

    @staticmethod
    def _empty_response(request_id: str, t0: float) -> Dict[str, Any]:
        return {
            "success": True,
            "scene": "productivity",
            "request_id": request_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "ops": [],
            "ops_summary": {"add": 0, "update": 0, "delete": 0, "skip": 0, "total": 0},
            "memory_ids": [],
        }


# ================================================================
# Top-level helper（client.async_add 用）
# ================================================================

async def is_coding_segment(
    messages: List[ChatMessage],
    llm_provider: LLMProvider,
) -> bool:
    """
    判定整段 messages 是否 coding 场景。无 tool 消息直接 False（O(1) 短路）。
    """
    if not has_any_tool_message(messages):
        return False
    return await classify_messages_is_coding(messages, llm_provider)
