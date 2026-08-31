"""
Agent Memory V2 - VectorStore (Qdrant)

基于 Qdrant 的向量存储层。使用同步 QdrantClient + asyncio.to_thread，
彻底避免 AsyncQdrantClient 的跨 event loop 问题。

职责:
- L1 Raw 对话 append-only 存储
- L2 Fact 语义检索
- Schema / Intention 向量匹配
- 时间增强向量检索

集合策略:
- 单集合设计: 所有层级的向量存入同一个 collection
- 通过 payload filter 区分 isolation_key / layer / status
- 简化运维，后续可按需拆分
"""

import asyncio
import concurrent.futures
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

from ..models.memory import (
    MemoryNode, MemoryLayer, MemoryStatus,
)
from ..config import MemoryConfig
from .vector_store_base import VectorStoreBase

logger = logging.getLogger(__name__)

# VDB 独立线程池（与 Chroma 共享同一策略）
from .vector_store_chroma import _vdb_executor


def _run_in_vdb_pool(func, *args, **kwargs):
    """在 VDB 独立线程池中执行同步函数"""
    import functools
    loop = asyncio.get_event_loop()
    if args or kwargs:
        return loop.run_in_executor(_vdb_executor, functools.partial(func, *args, **kwargs))
    return loop.run_in_executor(_vdb_executor, func)


class QdrantVectorStore(VectorStoreBase):
    """
    Qdrant 向量存储实现

    使用同步 QdrantClient + asyncio.to_thread 包装为异步，
    彻底绕开 httpx/httpcore 内部 asyncio.Event 跨 loop 绑定问题。
    """

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._client = None  # 同步 QdrantClient
        # qdrant 原生 BM25 走 sparse vector（fastembed Qdrant/bm25），返回的是
        # 真实 BM25 原始分（量级 0~20+），reader 端用 normalize_bm25 sigmoid 归一化
        # → 保持 _keyword_score_normalized=False（与经典 BM25 一致）。
        self._keyword_score_normalized = False
        # collection 是否含 sparse "bm25" 向量 + fastembed 可用 → 决定 keyword_search
        # 是否走真 BM25；旧的 dense-only collection 无 sparse 字段则降级。
        self._supports_fulltext = False
        self._sparse_name = "bm25"

    async def initialize(self) -> None:
        """初始化 Qdrant 客户端，确保集合存在"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                VectorParams, Distance,
                PayloadSchemaType, TextIndexParams, TokenizerType,
            )
            try:
                from qdrant_client.models import SparseVectorParams
            except Exception:
                SparseVectorParams = None
        except ImportError:
            raise ImportError(
                "qdrant-client is required for Qdrant backend. "
                "Install with: pip install hy-memory[qdrant]"
            )

        from ..pipelines._retrieval import bm25_fastembed
        # 新建 collection 是否带 sparse BM25：需 SparseVectorParams + fastembed 都可用
        enable_sparse = (SparseVectorParams is not None) and bm25_fastembed.is_available()

        vs_config = self.config.vector_store

        # 优先使用远程连接
        if vs_config.host:
            import os as _os
            # 扩大连接池以支持高并发（默认 keepalive=20 太小）
            _pool_size = int(_os.environ.get("QDRANT_POOL_SIZE", "256"))
            self._client = QdrantClient(
                host=vs_config.host,
                port=vs_config.port or 6333,
                api_key=vs_config.api_key or None,
                timeout=60,
                pool_size=_pool_size,
            )
            logger.info(f"[qdrant] pool_size={_pool_size}")
        else:
            # 兼容性处理: 清理高版本 qdrant-client 写入的 meta.json 中的未知字段
            self._patch_local_meta(vs_config.persist_directory)
            self._client = QdrantClient(
                path=vs_config.persist_directory,
            )

        # 确保集合存在（同步调用，在 to_thread 中执行）
        def _ensure_collection():
            collections = self._client.get_collections()
            collection_names = [c.name for c in collections.collections]

            created_with_sparse = False
            if self._collection_name not in collection_names:
                create_kwargs = dict(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=vs_config.embedding_dims,
                        distance=Distance.COSINE,
                    ),
                )
                if enable_sparse:
                    create_kwargs["sparse_vectors_config"] = {
                        self._sparse_name: SparseVectorParams()
                    }
                    created_with_sparse = True
                self._client.create_collection(**create_kwargs)
                # 创建 payload 索引以加速过滤
                for field_name, field_type in [
                    ("isolation_key", PayloadSchemaType.KEYWORD),
                    ("user_id", PayloadSchemaType.KEYWORD),
                    ("agent_id", PayloadSchemaType.KEYWORD),
                    ("session_id", PayloadSchemaType.KEYWORD),
                    ("layer", PayloadSchemaType.KEYWORD),
                    ("status", PayloadSchemaType.KEYWORD),
                    ("is_latest", PayloadSchemaType.BOOL),
                ]:
                    self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
                # Full-text index on search_text (content + tags) for keyword search
                # 注意：tokenizer 用 WHITESPACE 而不是 MULTILINGUAL —— 配合
                # _node_to_payload 中 lemmatize_for_bm25 (jieba) 预分词的 search_text，
                # 让查询和倒排索引两侧统一用 jieba 分词，避免 MULTILINGUAL/jieba 边界差异。
                try:
                    self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name="search_text",
                        field_schema=TextIndexParams(
                            type="text",
                            tokenizer=TokenizerType.WHITESPACE,
                            min_token_len=2,
                            max_token_len=20,
                        ),
                    )
                except Exception as e:
                    logger.warning(f"[qdrant] Failed to create search_text text index: {e}")
                logger.info(
                    f"Created Qdrant collection: {self._collection_name} "
                    f"(sparse_bm25={'on' if created_with_sparse else 'off'})"
                )
            else:
                # 对已有集合：尝试补建 search_text text index（幂等）
                self._ensure_text_index()

            # 探测该 collection 是否含 sparse "bm25" 向量（兼容旧 dense-only collection）
            has_sparse = created_with_sparse
            if not has_sparse:
                try:
                    info = self._client.get_collection(self._collection_name)
                    sp = getattr(getattr(info, "config", None), "params", None)
                    sparse_cfg = getattr(sp, "sparse_vectors", None) if sp else None
                    has_sparse = bool(sparse_cfg) and self._sparse_name in sparse_cfg
                except Exception as e:
                    logger.debug(f"[qdrant] sparse probe failed: {e}")
                    has_sparse = False
            return has_sparse

        has_sparse = await _run_in_vdb_pool(_ensure_collection)
        self._supports_fulltext = bool(has_sparse and bm25_fastembed.is_available())
        logger.info(
            f"[qdrant] fulltext(BM25 sparse)={'on' if self._supports_fulltext else 'off'}"
        )
        logger.debug(f"VectorStore initialized (Qdrant sync client), collection={self._collection_name}")

    # ================================================================
    # 写入
    # ================================================================

    @staticmethod
    def _dense_of(vec):
        """从 qdrant 返回的 vector 取 dense 部分。

        含 sparse 的 collection 里 point.vector 是命名多向量 dict
        （dense 在默认空键 ""，sparse 在 "bm25"）；dense-only collection 则直接
        是 List[float]。统一归一为 List[float]（取不到则 None）。
        """
        if isinstance(vec, dict):
            return vec.get("") or vec.get("dense")
        return vec

    def _build_vector_field(self, payload: Dict[str, Any], embedding: List[float]):
        """
        构造 PointStruct.vector 字段。

        - fulltext 关闭：返回纯 dense list（向后兼容旧 dense-only collection）
        - fulltext 开启：返回 {"": dense, "bm25": SparseVector(...)}，sparse 由
          fastembed BM25 对 search_text 编码而来；编码失败则回落纯 dense。

        跳过 L1_RAW：原始对话层不被任何召回路径消费（reader 召回 L0/L2/L3/L4，
        提取后又降 SHADOW，System2 也只取 L2/L4），给它编 sparse 纯浪费写入/存储。
        """
        # 防御：若上游误传了命名多向量 dict（如 clone 复制了读路径的 embedding），
        # 先抽出 dense，避免再包一层生成非法嵌套结构。
        embedding = self._dense_of(embedding)
        if not self._supports_fulltext:
            return embedding
        if payload.get("layer") == MemoryLayer.L1_RAW.value:
            return embedding
        from ..pipelines._retrieval import bm25_fastembed
        from qdrant_client.models import SparseVector

        text = payload.get("search_text") or payload.get("content") or ""
        sp = bm25_fastembed.encode_doc(text) if text else None
        if not sp:
            return embedding
        indices, values = sp
        return {"": embedding, self._sparse_name: SparseVector(indices=indices, values=values)}

    async def upsert(self, node: MemoryNode) -> str:
        """写入或更新一个 MemoryNode 的向量和 payload"""
        from qdrant_client.models import PointStruct

        if node.embedding is None:
            raise ValueError(f"Node {node.node_id} has no embedding, cannot upsert to vector store")

        payload = self._node_to_payload(node)
        point_id = self._node_id_to_point_id(node.node_id)
        vector_field = self._build_vector_field(payload, node.embedding)

        def _upsert():
            self._client.upsert(
                collection_name=self._collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector_field,
                        payload=payload,
                    )
                ],
            )

        await _run_in_vdb_pool(_upsert)
        logger.debug(
            f"[vector-store] upsert: id={node.node_id} "
            f"layer={node.layer.value if hasattr(node.layer, 'value') else node.layer} "
            f"isolation_key={payload.get('isolation_key', '')} "
            f"sparse={'yes' if isinstance(vector_field, dict) else 'no'} "
            f"content={node.content[:200]}"
        )
        return node.node_id

    async def upsert_batch(self, nodes: List[MemoryNode]) -> List[str]:
        """批量写入"""
        from qdrant_client.models import PointStruct

        points = []
        ids = []
        for node in nodes:
            if node.embedding is None:
                logger.warning(f"Skipping node {node.node_id}: no embedding")
                continue
            payload = self._node_to_payload(node)
            points.append(PointStruct(
                id=self._node_id_to_point_id(node.node_id),
                vector=self._build_vector_field(payload, node.embedding),
                payload=payload,
            ))
            ids.append(node.node_id)

        if points:
            def _upsert_batch():
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                )
            await _run_in_vdb_pool(_upsert_batch)
        return ids

    async def update_embedding(self, node_id: str, embedding: List[float]) -> bool:
        """仅更新向量（payload 不变）"""
        from qdrant_client.models import PointVectors

        point_id = self._node_id_to_point_id(node_id)
        try:
            def _update():
                self._client.update_vectors(
                    collection_name=self._collection_name,
                    points=[PointVectors(id=point_id, vector=embedding)],
                )
            await _run_in_vdb_pool(_update)
            return True
        except Exception as e:
            logger.warning(f"Failed to update embedding for {node_id}: {e}")
            return False

    async def update_payload(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """更新 payload 字段；若 updates 含 'embedding'，同时更新向量。"""
        if not updates:
            return True
        from qdrant_client.models import PointVectors
        point_id = self._node_id_to_point_id(node_id)
        updates = dict(updates)
        new_embedding = updates.pop("embedding", None)  # 取出向量单独处理
        try:
            def _update():
                if updates:
                    self._client.set_payload(
                        collection_name=self._collection_name,
                        payload=updates,
                        points=[point_id],
                    )
                if new_embedding is not None:
                    self._client.update_vectors(
                        collection_name=self._collection_name,
                        points=[PointVectors(id=point_id, vector=new_embedding)],
                    )
            await _run_in_vdb_pool(_update)
            logger.debug(
                f"[vector-store] update_payload: id={node_id} updates={list(updates.keys())} "
                f"embedding={'yes' if new_embedding is not None else 'no'}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to update payload for {node_id}: {e}")
            return False

    # ================================================================
    # 检索
    # ================================================================

    async def search(
        self,
        query_embedding: List[float],
        isolation_key: str = "",
        isolation_keys: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        layers: Optional[List[MemoryLayer]] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        status_filter: Optional[List[MemoryStatus]] = None,
        only_latest: bool = True,
        tags_match_any: Optional[List[str]] = None,
        created_after: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """语义检索。only_latest=True 时只搜索 is_latest=True 的节点（默认）。"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        must_conditions = []

        # 隔离过滤: 精确 keys > 单 key > user_ids/user_id + agent 级
        if isolation_keys:
            must_conditions.append(
                FieldCondition(key="isolation_key", match=MatchAny(any=isolation_keys))
            )
        elif isolation_key:
            must_conditions.append(
                FieldCondition(key="isolation_key", match=MatchValue(value=isolation_key))
            )
        else:
            effective_uids = user_ids if user_ids else ([user_id] if user_id else [])
            if effective_uids:
                if len(effective_uids) == 1:
                    must_conditions.append(
                        FieldCondition(key="user_id", match=MatchValue(value=effective_uids[0]))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key="user_id", match=MatchAny(any=effective_uids))
                    )
                if agent_ids:
                    must_conditions.append(
                        FieldCondition(key="agent_id", match=MatchAny(any=agent_ids))
                    )

        if layers:
            must_conditions.append(
                FieldCondition(key="layer", match=MatchAny(any=[l.value for l in layers]))
            )

        if status_filter:
            must_conditions.append(
                FieldCondition(key="status", match=MatchAny(any=[s.value for s in status_filter]))
            )
        else:
            # 默认只返回 ACTIVE 状态
            must_conditions.append(
                FieldCondition(key="status", match=MatchValue(value=MemoryStatus.ACTIVE.value))
            )

        # 只搜索 is_latest=True 的节点（演化链末端）
        if only_latest:
            from qdrant_client.models import FieldCondition as FC
            must_conditions.append(
                FC(key="is_latest", match=MatchValue(value=True))
            )

        # Tag 过滤：reader_hybrid_tag 路 B 使用。只保留 tags 字段与给定列表有交集的点。
        # Qdrant 对 list 类型字段的 MatchAny 等价于 "array has any of these values"。
        if tags_match_any:
            # 去重并过滤空串，避免 Qdrant 报错
            tag_list = sorted({t for t in tags_match_any if t})
            if tag_list:
                must_conditions.append(
                    FieldCondition(key="tags", match=MatchAny(any=tag_list))
                )

        # created_after: 只返回 gmt_created >= 指定时间的记忆
        if created_after:
            from qdrant_client.models import Range
            must_conditions.append(
                FieldCondition(key="gmt_created", range=Range(gte=created_after))
            )

        query_filter = Filter(must=must_conditions)

        def _search():
            # 兼容 qdrant-client >=1.17 (query_points) 和 <1.17 (search)
            if hasattr(self._client, 'query_points'):
                query_response = self._client.query_points(
                    collection_name=self._collection_name,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
                return query_response.points
            else:
                return self._client.search(
                    collection_name=self._collection_name,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )

        results = await _run_in_vdb_pool(_search)

        output = []
        for hit in results:
            node = self._payload_to_node(hit.payload)
            output.append({
                "node_id": node.node_id,
                "score": hit.score,
                "node": node,
            })

        logger.debug(
            f"[vector-store] search: isolation_key={isolation_key} "
            f"isolation_keys={isolation_keys} user_id={user_id} agent_ids={agent_ids} "
            f"limit={limit} found={len(output)}"
        )
        for i, item in enumerate(output):
            logger.debug(
                f"[vector-store] hit[{i}] score={item['score']:.4f} "
                f"id={item['node_id']} "
                f"layer={item['node'].layer.value if hasattr(item['node'].layer, 'value') else ''} "
                f"content={item['node'].content[:200]}"
            )

        return output

    async def get_by_id(self, node_id: str) -> Optional[MemoryNode]:
        """按 ID 获取"""
        point_id = self._node_id_to_point_id(node_id)
        try:
            def _retrieve():
                return self._client.retrieve(
                    collection_name=self._collection_name,
                    ids=[point_id],
                    with_payload=True,
                    with_vectors=True,
                )
            results = await _run_in_vdb_pool(_retrieve)
            if results:
                node = self._payload_to_node(results[0].payload)
                if results[0].vector:
                    node.embedding = self._dense_of(results[0].vector)
                return node
        except Exception as e:
            logger.warning(f"Failed to get vector point {node_id}: {e}")
        return None

    async def get_by_ids(self, node_ids: List[str]) -> List[MemoryNode]:
        """批量按 ID 获取（单次 retrieve 调用）"""
        if not node_ids:
            return []
        point_ids = [self._node_id_to_point_id(nid) for nid in node_ids]
        try:
            def _retrieve():
                return self._client.retrieve(
                    collection_name=self._collection_name,
                    ids=point_ids,
                    with_payload=True,
                    with_vectors=False,
                )
            results = await _run_in_vdb_pool(_retrieve)
            nodes = []
            for point in results:
                node = self._payload_to_node(point.payload)
                nodes.append(node)
            return nodes
        except Exception as e:
            logger.warning(f"Failed to batch get vector points {node_ids}: {e}")
            return []

    async def get_embeddings(self, node_ids: List[str]) -> Dict[str, List[float]]:
        """批量取向量（单次 retrieve，with_vectors=True），供去重本地算 cosine。"""
        if not node_ids:
            return {}
        point_ids = [self._node_id_to_point_id(nid) for nid in node_ids]
        try:
            def _retrieve():
                return self._client.retrieve(
                    collection_name=self._collection_name,
                    ids=point_ids,
                    with_payload=True,
                    with_vectors=True,
                )
            results = await _run_in_vdb_pool(_retrieve)
            out: Dict[str, List[float]] = {}
            for point in results:
                node = self._payload_to_node(point.payload)
                vec = self._dense_of(point.vector)
                if node.node_id and vec and not isinstance(vec, dict):
                    out[node.node_id] = list(vec)
            return out
        except Exception as e:
            logger.warning(f"Failed to batch get embeddings {node_ids}: {e}")
            return {}

    # ================================================================
    # 删除
    # ================================================================

    async def delete(self, node_id: str) -> bool:
        """删除单个向量点"""
        from qdrant_client.models import PointIdsList

        point_id = self._node_id_to_point_id(node_id)
        try:
            def _delete():
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=PointIdsList(points=[point_id]),
                )
            await _run_in_vdb_pool(_delete)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete vector point {node_id}: {e}")
            return False

    async def delete_by_isolation_key(self, isolation_key: str) -> int:
        """删除某隔离键下的所有向量"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        try:
            def _delete():
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(key="isolation_key", match=MatchValue(value=isolation_key))
                            ]
                        )
                    ),
                )
            await _run_in_vdb_pool(_delete)
            return -1  # Qdrant 不返回删除数量
        except Exception as e:
            logger.warning(f"Failed to delete vectors for {isolation_key}: {e}")
            return 0

    async def delete_by_metadata(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """按 metadata 字段组合删除向量"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        try:
            must_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]
            if agent_id is not None:
                must_conditions.append(
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
                )
            if session_id is not None:
                must_conditions.append(
                    FieldCondition(key="session_id", match=MatchValue(value=session_id))
                )

            def _delete():
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(must=must_conditions)
                    ),
                )
            await _run_in_vdb_pool(_delete)
            return -1  # Qdrant 不返回删除数量
        except Exception as e:
            logger.warning(f"Failed to delete vectors by metadata (user_id={user_id}): {e}")
            return 0

    # ================================================================
    # 枚举
    # ================================================================

    async def list_by_user(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        limit: int = 10000,
        status_filter: Optional[List] = None,
        layers: Optional[List] = None,
    ) -> List[MemoryNode]:
        """枚举某用户的记忆节点（含 embedding），使用 scroll API"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        if status_filter:
            status_values = [s.value if hasattr(s, 'value') else str(s) for s in status_filter]
            must_conditions.append(
                FieldCondition(key="status", match=MatchAny(any=status_values))
            )
        if layers:
            layer_values = [l.value if hasattr(l, 'value') else str(l) for l in layers]
            must_conditions.append(
                FieldCondition(key="layer", match=MatchAny(any=layer_values))
            )

        scroll_filter = Filter(must=must_conditions)

        def _scroll_all():
            nodes = []
            offset = None
            batch_size = min(limit, 100)

            while len(nodes) < limit:
                scroll_result = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=scroll_filter,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                # 兼容 qdrant-client >= 1.17 (返回 ScrollResult 对象)
                # 和 < 1.17 (返回 tuple(points, next_offset))
                if isinstance(scroll_result, tuple):
                    points, next_offset = scroll_result
                else:
                    points = scroll_result.points
                    next_offset = scroll_result.next_page_offset

                for point in points:
                    node = self._payload_to_node(point.payload)
                    if point.vector:
                        node.embedding = self._dense_of(point.vector)
                    nodes.append(node)

                if next_offset is None or not points:
                    break
                offset = next_offset

            return nodes[:limit]

        try:
            nodes = await _run_in_vdb_pool(_scroll_all)
        except Exception as e:
            logger.error(f"[vector-store] list_by_user scroll failed: {e}", exc_info=True)
            nodes = []

        logger.info(f"[vector-store] list_by_user: user_id={user_id} agent_id={agent_id} found={len(nodes)}")
        return nodes

    # ================================================================
    # 统计
    # ================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """获取集合统计"""
        try:
            def _stats():
                return self._client.get_collection(self._collection_name)
            info = await _run_in_vdb_pool(_stats)
            return {
                "collection": self._collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as e:
            return {"error": str(e)}

    async def count(self, isolation_key: str) -> int:
        """统计某隔离键下的向量数量"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        try:
            def _count():
                return self._client.count(
                    collection_name=self._collection_name,
                    count_filter=Filter(
                        must=[
                            FieldCondition(key="isolation_key", match=MatchValue(value=isolation_key))
                        ]
                    ),
                    exact=True,
                )
            result = await _run_in_vdb_pool(_count)
            return result.count
        except Exception as e:
            logger.warning(f"Failed to count vectors for {isolation_key}: {e}")
            return 0

    async def close(self) -> None:
        """关闭客户端"""
        if self._client:
            self._client.close()
            self._client = None

    # ================================================================
    # Keyword Search (Full-text)
    # ================================================================

    def _ensure_text_index(self) -> None:
        """
        对已有集合补建 search_text text index。

        现状期望：tokenizer = WHITESPACE（配合 _node_to_payload 的 jieba 预分词）。

        三种情况：
          1) 没有 index → 创建 WHITESPACE index
          2) 已有 WHITESPACE index → skip
          3) 已有别的 tokenizer (e.g. MULTILINGUAL 老索引) → 仅打 WARNING，不主动重建
             因为现存 payload 里 search_text 还是老格式（原始中文），新 tokenizer 重建
             会得到无用 token。需要先跑 scripts/migrate_search_text_to_jieba.py
             重新写入 payload，再 drop+recreate index。
        """
        try:
            from qdrant_client.models import TextIndexParams, TokenizerType

            # 先看一眼现状
            existing_tokenizer = None
            try:
                info = self._client.get_collection(self._collection_name)
                payload_schema = getattr(info, "payload_schema", None) or {}
                schema_entry = payload_schema.get("search_text") if isinstance(payload_schema, dict) else None
                if schema_entry is not None:
                    params = getattr(schema_entry, "params", None)
                    tok = getattr(params, "tokenizer", None) if params else None
                    if tok is not None:
                        # qdrant 返回的 tokenizer 可能是 enum 或 string
                        existing_tokenizer = getattr(tok, "value", None) or str(tok)
            except Exception as e:
                logger.debug(f"[qdrant] inspect search_text index failed (will try create anyway): {e}")

            target_tokenizer_value = getattr(TokenizerType.WHITESPACE, "value", "whitespace")

            if existing_tokenizer is not None:
                if str(existing_tokenizer).lower().endswith("whitespace") or str(existing_tokenizer).lower() == "whitespace":
                    return  # 已经是新格式，啥都不做
                logger.warning(
                    f"[qdrant] search_text text index uses old tokenizer={existing_tokenizer!r} "
                    f"(expected whitespace). Keyword search will be DEGRADED on this collection until "
                    f"you run scripts/migrate_search_text_to_jieba.py to re-upsert payloads + rebuild index. "
                    f"collection={self._collection_name}"
                )
                return

            # 没 index → 直接 create
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="search_text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WHITESPACE,
                    min_token_len=2,
                    max_token_len=20,
                ),
            )
            logger.info(f"[qdrant] Created search_text text index (whitespace) on existing collection")
        except Exception as e:
            # 兜底：未支持或并发竞争，静默
            logger.debug(f"[qdrant] _ensure_text_index swallowed: {e}")

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        user_id: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
        layers: Optional[List[MemoryLayer]] = None,
        status_filter: Optional[List[MemoryStatus]] = None,
        only_latest: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        关键词检索。

        - collection 含 sparse "bm25" 向量且 fastembed 可用（_supports_fulltext）：
          走 query_points(using="bm25") 真 BM25 sparse 检索，返回真实 BM25 分
          （量级 0~20+，reader 端用 normalize_bm25 sigmoid 归一）。
        - 否则（旧 dense-only collection）：降级返回 []，hybrid 退化为纯向量。
          注：不再用旧的 scroll text-match + binary 1.0 假 BM25。
        """
        from qdrant_client.models import (
            Filter, FieldCondition, MatchValue, MatchAny,
        )

        if not query or not query.strip():
            return []

        if not self._supports_fulltext:
            if not getattr(self, "_warned_no_fulltext", False):
                self._warned_no_fulltext = True
                logger.warning(
                    f"[qdrant] keyword_search degraded to empty: collection "
                    f"{self._collection_name} has no sparse 'bm25' vector (or fastembed "
                    f"missing). Recreate collection (auto-adds sparse) + re-upsert to "
                    f"enable real BM25. pip install fastembed"
                )
            return []

        from ..pipelines._retrieval import bm25_fastembed
        from qdrant_client.models import SparseVector

        q_sp = bm25_fastembed.encode_query(query.strip())
        if not q_sp:
            return []
        q_indices, q_values = q_sp

        must_conditions = []
        if user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )
        if agent_ids:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchAny(any=agent_ids))
            )
        if layers:
            must_conditions.append(
                FieldCondition(key="layer", match=MatchAny(any=[l.value for l in layers]))
            )
        if status_filter:
            must_conditions.append(
                FieldCondition(key="status", match=MatchAny(any=[s.value for s in status_filter]))
            )
        else:
            must_conditions.append(
                FieldCondition(key="status", match=MatchValue(value=MemoryStatus.ACTIVE.value))
            )
        if only_latest:
            must_conditions.append(
                FieldCondition(key="is_latest", match=MatchValue(value=True))
            )
        query_filter = Filter(must=must_conditions) if must_conditions else None

        def _bm25_search():
            resp = self._client.query_points(
                collection_name=self._collection_name,
                query=SparseVector(indices=q_indices, values=q_values),
                using=self._sparse_name,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return resp.points

        try:
            points = await _run_in_vdb_pool(_bm25_search)
        except Exception as e:
            logger.warning(f"[qdrant] keyword_search(bm25 sparse) failed: {e}")
            return []

        output = []
        for point in points:
            node = self._payload_to_node(point.payload)
            output.append({
                "node_id": node.node_id,
                "score": float(getattr(point, "score", 0.0) or 0.0),  # 真实 BM25 分
                "node": node,
            })

        logger.debug(
            f"[vector-store] keyword_search(bm25): query='{query[:50]}' "
            f"user_id={user_id} top_k={top_k} found={len(output)}"
        )
        return output

    # ================================================================
    # Qdrant 特有方法
    # ================================================================

    @staticmethod
    def _patch_local_meta(persist_directory: str) -> None:
        """
        兼容性补丁: 清理本地 Qdrant meta.json 中当前版本不支持的字段。

        高版本 qdrant-client (>=1.13) 会在 meta.json 的 collection config 中写入
        strict_mode_config、metadata 等字段，低版本 (<1.13) 的 pydantic model
        配置了 extra='forbid'，加载时会报 ValidationError。
        """
        import json as _json
        meta_path = Path(persist_directory) / "meta.json"
        if not meta_path.exists():
            return

        try:
            from qdrant_client.http.models import CreateCollection
            known_fields = set(CreateCollection.model_fields.keys())
        except Exception:
            return

        try:
            raw = meta_path.read_text(encoding="utf-8")
            meta = _json.loads(raw)
        except Exception:
            return

        changed = False
        for coll_name, coll_cfg in meta.get("collections", {}).items():
            if not isinstance(coll_cfg, dict):
                continue
            extra_keys = set(coll_cfg.keys()) - known_fields
            for k in extra_keys:
                del coll_cfg[k]
                changed = True

        if changed:
            meta_path.write_text(_json.dumps(meta), encoding="utf-8")
            logger.info(f"Patched Qdrant meta.json: removed unsupported fields")

    # ================================================================
    # Tag Index（per-user tag embedding，供 reader_hybrid_tag 使用）
    # ================================================================

    _supports_tag_index = True

    def _tag_index_collection(self) -> str:
        """
        tag_index 独立 collection 名称。与主 memories collection 同库不同表。

        命名规则对齐 `_retrieval.config.tag_index_collection_name`。
        """
        # 延迟 import 避免循环依赖
        from ..pipelines._retrieval.config import tag_index_collection_name
        return tag_index_collection_name(self._collection_name)

    async def _ensure_tag_index_collection(self) -> None:
        """
        确保 tag_index collection 存在。空 collection 惰性创建，避免启动时就占资源。

        幂等：多次调用安全。
        """
        from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

        coll = self._tag_index_collection()

        def _ensure():
            collections = self._client.get_collections()
            names = [c.name for c in collections.collections]
            if coll in names:
                return
            self._client.create_collection(
                collection_name=coll,
                vectors_config=VectorParams(
                    size=self.config.vector_store.embedding_dims,
                    distance=Distance.COSINE,
                ),
            )
            for field_name, field_type in [
                ("user_id", PayloadSchemaType.KEYWORD),
                ("tag", PayloadSchemaType.KEYWORD),
            ]:
                try:
                    self._client.create_payload_index(
                        collection_name=coll,
                        field_name=field_name,
                        field_schema=field_type,
                    )
                except Exception:
                    pass  # 已存在等情况忽略
            logger.info(f"Created Qdrant tag_index collection: {coll}")

        await _run_in_vdb_pool(_ensure)

    async def upsert_tag_embedding(
        self, user_id: str, tag: str, embedding: List[float]
    ) -> None:
        from qdrant_client.models import PointStruct

        await self._ensure_tag_index_collection()
        coll = self._tag_index_collection()
        point_id = self._tag_point_id(user_id, tag)

        def _upsert():
            self._client.upsert(
                collection_name=coll,
                points=[PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"user_id": user_id, "tag": tag},
                )],
            )

        await _run_in_vdb_pool(_upsert)
        logger.debug(f"[tag-index] upsert: user={user_id} tag={tag}")

    async def has_tag_embedding(self, user_id: str, tag: str) -> bool:
        coll = self._tag_index_collection()
        point_id = self._tag_point_id(user_id, tag)

        def _retrieve():
            try:
                points = self._client.retrieve(
                    collection_name=coll,
                    ids=[point_id],
                    with_payload=False,
                    with_vectors=False,
                )
                return bool(points)
            except Exception:
                # Collection 不存在也算 False
                return False

        return await _run_in_vdb_pool(_retrieve)

    async def delete_tag_embedding(self, user_id: str, tag: str) -> None:
        from qdrant_client.models import PointIdsList

        coll = self._tag_index_collection()
        point_id = self._tag_point_id(user_id, tag)

        def _delete():
            try:
                self._client.delete(
                    collection_name=coll,
                    points_selector=PointIdsList(points=[point_id]),
                )
            except Exception as e:
                logger.debug(f"[tag-index] delete no-op ({user_id}, {tag}): {e}")

        await _run_in_vdb_pool(_delete)

    async def search_tag_embeddings(
        self,
        user_id: str,
        query_embedding: List[float],
        topk: int = 5,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        在 per-user tag_index 中做向量检索。

        语义：
          - **topk 为主**：先按 cosine 取最近的 topk 个 tag
          - **min_score 为地板线**：从 topk 里软过滤完全不沾边的
          - tag embedding 本身信噪弱（单词/短语 embedding），若 min_score 用 Qdrant
            服务端 score_threshold 做硬过滤，会把"仅次于阈值"的相近 tag 也屏蔽掉，
            导致 user 没有高度相关 tag 时整体空召回。这里改为应用层软过滤。
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        coll = self._tag_index_collection()
        qf = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

        def _search():
            try:
                # 不传 score_threshold：topk 在前、min_score 在后（软过滤）
                if hasattr(self._client, "query_points"):
                    resp = self._client.query_points(
                        collection_name=coll,
                        query=query_embedding,
                        query_filter=qf,
                        limit=topk,
                        with_payload=True,
                        with_vectors=False,
                    )
                    return resp.points
                return self._client.search(
                    collection_name=coll,
                    query_vector=query_embedding,
                    query_filter=qf,
                    limit=topk,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                logger.debug(f"[tag-index] search failed (user={user_id}): {e}")
                return []

        hits = await _run_in_vdb_pool(_search)
        # 应用层软过滤：topk 已取回，仅剔除完全不沾边（< min_score）的噪声 tag
        result: List[Dict[str, Any]] = []
        for h in hits or []:
            score = float(getattr(h, "score", 0.0))
            if score < min_score:
                continue
            payload = getattr(h, "payload", {}) or {}
            tag = payload.get("tag")
            if tag:
                result.append({"tag": tag, "score": score})
        return result

    async def count_memories_with_tag(
        self, user_id: str, tag: str, isolation_key: str = ""
    ) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        must = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            # Qdrant 对 list 类型字段的 MatchValue 等价于 "array contains value"
            FieldCondition(key="tags", match=MatchValue(value=tag)),
            FieldCondition(key="status", match=MatchAny(any=[MemoryStatus.ACTIVE.value])),
        ]
        if isolation_key:
            must.append(FieldCondition(key="isolation_key", match=MatchValue(value=isolation_key)))
        qf = Filter(must=must)

        def _count():
            try:
                result = self._client.count(
                    collection_name=self._collection_name,
                    count_filter=qf,
                    exact=True,
                )
                return int(getattr(result, "count", 0))
            except Exception as e:
                logger.debug(f"[tag-index] count_memories_with_tag failed ({user_id}, {tag}): {e}")
                return -1

        c = await _run_in_vdb_pool(_count)
        # 若失败返回 -1（调用方可据此跳过清理）
        return c

    # ================================================================
    # Entity Store（独立 {collection}_entities collection，对齐 mem0）
    #   - mirror tag_index 的 aux-collection 模式
    #   - Qdrant payload 原生支持 list，linked_memory_ids 直接存 list（无需 JSON）
    # ================================================================

    def _entity_collection(self) -> str:
        """entity store 独立 collection 名称（对齐 _retrieval.config.entity_collection_name）。"""
        from ..pipelines._retrieval.config import entity_collection_name
        return entity_collection_name(self._collection_name)

    async def _ensure_entity_collection(self) -> None:
        """惰性创建 entity store collection。幂等。"""
        from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

        coll = self._entity_collection()

        def _ensure():
            collections = self._client.get_collections()
            names = [c.name for c in collections.collections]
            if coll in names:
                return
            self._client.create_collection(
                collection_name=coll,
                vectors_config=VectorParams(
                    size=self.config.vector_store.embedding_dims,
                    distance=Distance.COSINE,
                ),
            )
            for field_name in ("user_id", "agent_id"):
                try:
                    self._client.create_payload_index(
                        collection_name=coll,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
            logger.info(f"Created Qdrant entity collection: {coll}")

        await _run_in_vdb_pool(_ensure)

    @staticmethod
    def _entity_filter(user_id: str, agent_ids: Optional[List[str]]):
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if agent_ids:
            if len(agent_ids) == 1:
                must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_ids[0])))
            else:
                must.append(FieldCondition(key="agent_id", match=MatchAny(any=list(agent_ids))))
        return Filter(must=must)

    async def upsert_entity(
        self,
        *,
        entity_text: str,
        entity_type: str,
        embedding: List[float],
        memory_id: str,
        user_id: str,
        agent_id: str = "",
        merge_threshold: float = 0.95,
    ) -> str:
        import uuid as _uuid
        from qdrant_client.models import PointStruct

        await self._ensure_entity_collection()
        coll = self._entity_collection()
        qf = self._entity_filter(user_id, [agent_id] if agent_id else None)

        def _do():
            # 1) 查最相似的同 user entity
            try:
                if hasattr(self._client, "query_points"):
                    resp = self._client.query_points(
                        collection_name=coll, query=embedding, query_filter=qf,
                        limit=1, with_payload=True, with_vectors=False,
                    )
                    hits = resp.points
                else:
                    hits = self._client.search(
                        collection_name=coll, query_vector=embedding, query_filter=qf,
                        limit=1, with_payload=True, with_vectors=False,
                    )
            except Exception:
                hits = []

            if hits:
                top = hits[0]
                score = float(getattr(top, "score", 0.0))
                if score >= merge_threshold:
                    payload = dict(getattr(top, "payload", {}) or {})
                    linked = payload.get("linked_memory_ids") or []
                    if not isinstance(linked, list):
                        linked = []
                    if memory_id not in linked:
                        linked.append(memory_id)
                        # set_payload 只更新 payload，不动向量
                        self._client.set_payload(
                            collection_name=coll,
                            payload={"linked_memory_ids": linked},
                            points=[top.id],
                        )
                    return str(top.id)

            # 2) 新建
            eid = str(_uuid.uuid4())
            self._client.upsert(
                collection_name=coll,
                points=[PointStruct(
                    id=eid,
                    vector=embedding,
                    payload={
                        "data": entity_text,
                        "entity_type": entity_type or "",
                        "linked_memory_ids": [memory_id],
                        "user_id": user_id,
                        "agent_id": agent_id or "",
                    },
                )],
            )
            return eid

        return await _run_in_vdb_pool(_do)

    async def search_entities(
        self,
        *,
        query_embedding: List[float],
        user_id: str,
        agent_ids: Optional[List[str]] = None,
        top_k: int = 500,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        coll = self._entity_collection()
        qf = self._entity_filter(user_id, agent_ids)

        def _search():
            try:
                if hasattr(self._client, "query_points"):
                    resp = self._client.query_points(
                        collection_name=coll, query=query_embedding, query_filter=qf,
                        limit=top_k, with_payload=True, with_vectors=False,
                    )
                    return resp.points
                return self._client.search(
                    collection_name=coll, query_vector=query_embedding, query_filter=qf,
                    limit=top_k, with_payload=True, with_vectors=False,
                )
            except Exception as e:
                logger.debug(f"[entity] search failed (user={user_id}): {e}")
                return []

        hits = await _run_in_vdb_pool(_search)
        out: List[Dict[str, Any]] = []
        for h in hits or []:
            score = float(getattr(h, "score", 0.0))
            if score < min_score:
                continue
            payload = getattr(h, "payload", {}) or {}
            linked = payload.get("linked_memory_ids") or []
            if not isinstance(linked, list):
                linked = []
            out.append({
                "entity_id": str(h.id),
                "data": payload.get("data", ""),
                "entity_type": payload.get("entity_type", ""),
                "linked_memory_ids": linked,
                "score": score,
            })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    async def list_entities(
        self, *, user_id: str, agent_ids: Optional[List[str]] = None, top_k: int = 10000
    ) -> List[Dict[str, Any]]:
        coll = self._entity_collection()
        qf = self._entity_filter(user_id, agent_ids)

        def _scroll():
            try:
                points, _ = self._client.scroll(
                    collection_name=coll, scroll_filter=qf,
                    limit=top_k, with_payload=True, with_vectors=False,
                )
                return points
            except Exception as e:
                logger.debug(f"[entity] list failed (user={user_id}): {e}")
                return []

        points = await _run_in_vdb_pool(_scroll)
        out: List[Dict[str, Any]] = []
        for p in points or []:
            payload = getattr(p, "payload", {}) or {}
            linked = payload.get("linked_memory_ids") or []
            if not isinstance(linked, list):
                linked = []
            out.append({
                "entity_id": str(p.id),
                "data": payload.get("data", ""),
                "entity_type": payload.get("entity_type", ""),
                "linked_memory_ids": linked,
            })
        return out

    async def delete_entities_for_memory(
        self, *, memory_id: str, user_id: str, agent_ids: Optional[List[str]] = None
    ) -> int:
        from qdrant_client.models import PointIdsList

        coll = self._entity_collection()
        qf = self._entity_filter(user_id, agent_ids)

        def _do():
            try:
                points, _ = self._client.scroll(
                    collection_name=coll, scroll_filter=qf,
                    limit=100000, with_payload=True, with_vectors=False,
                )
            except Exception as e:
                logger.debug(f"[entity] delete scroll failed (user={user_id}): {e}")
                return 0
            affected = 0
            for p in points or []:
                payload = dict(getattr(p, "payload", {}) or {})
                linked = payload.get("linked_memory_ids") or []
                if not isinstance(linked, list) or memory_id not in linked:
                    continue
                affected += 1
                remaining = [m for m in linked if m != memory_id]
                if not remaining:
                    try:
                        self._client.delete(
                            collection_name=coll,
                            points_selector=PointIdsList(points=[p.id]),
                        )
                    except Exception as e:
                        logger.debug(f"[entity] delete point {p.id} failed: {e}")
                else:
                    try:
                        self._client.set_payload(
                            collection_name=coll,
                            payload={"linked_memory_ids": remaining},
                            points=[p.id],
                        )
                    except Exception as e:
                        logger.debug(f"[entity] update point {p.id} failed: {e}")
            return affected

        return await _run_in_vdb_pool(_do)
