"""
Agent Memory V2 - VectorStore (Tencent Cloud VectorDB)

基于腾讯云向量数据库的向量存储层。

特点:
- 云端托管，免运维
- HTTP API，通过 tcvectordb SDK 调用
- 同步 SDK → _run_in_vdb_pool() 包装为异步
"""

from typing import Optional, List, Dict, Any
import asyncio
import concurrent.futures
import logging

from ..config import MemoryConfig
from ..models.memory import MemoryNode, MemoryLayer, MemoryStatus
from .vector_store_base import VectorStoreBase

logger = logging.getLogger(__name__)

# VDB 独立线程池（与 Chroma/Qdrant 共享）
from .vector_store_chroma import _vdb_executor


def _run_in_vdb_pool(func, *args, **kwargs):
    """在 VDB 独立线程池中执行同步函数"""
    import functools
    loop = asyncio.get_event_loop()
    if args or kwargs:
        return loop.run_in_executor(_vdb_executor, functools.partial(func, *args, **kwargs))
    return loop.run_in_executor(_vdb_executor, func)


class TencentVectorStore(VectorStoreBase):
    """
    腾讯云向量数据库后端

    使用 tcvectordb SDK (HTTP 模式) 连接腾讯云 VectorDB 实例。

    配置:
        vector_store.provider = "tencent"
        vector_store.url = MEMORY_TENCENT_VDB_URL
        vector_store.username = MEMORY_TENCENT_VDB_USERNAME (默认 root)
        vector_store.api_key = MEMORY_VECTOR_API_KEY
        vector_store.database_name = MEMORY_TENCENT_VDB_DATABASE (默认 hy_memory)
        vector_store.collection_name = MEMORY_COLLECTION_NAME (默认 agent_memories)
        vector_store.embedding_dims = MEMORY_EMBEDDING_DIMS
    """

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._client = None
        self._db_name = ""
        self._collection_name = ""
        self._dims = 1536
        # 全文/关键词检索（sparse vector + BM25）能力开关。
        # initialize() 时按 collection 是否有 sparse_vector 索引 + BM25Encoder
        # 是否可用来探测；不支持时 keyword_search/hybrid 自动降级为纯向量。
        self._supports_fulltext = False
        self._sparse_field = "sparse_vector"
        # tencent sparse fulltext_search 返回的分是 IP 内积（~[0,1]），已是
        # 归一化相关性分；reader 直接用，不再过 normalize_bm25 sigmoid。
        self._keyword_score_normalized = True

    async def initialize(self) -> None:
        """初始化: 创建客户端、确保 database 和 collection 存在"""
        try:
            import tcvectordb
            from tcvectordb.model.enum import FieldType, IndexType, MetricType
            from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams
            try:
                from tcvectordb.model.index import SparseIndex
            except Exception:
                SparseIndex = None
        except ImportError:
            raise ImportError(
                "tcvectordb is required for Tencent Cloud VectorDB. "
                "Install with: pip install tcvectordb"
            )

        from ..pipelines._retrieval import bm25_sparse
        # BM25 编码器是否可用（tcvdb_text 已安装）；不可用则不开 sparse
        sparse_encoder_ok = bm25_sparse.is_available()
        sparse_index_ok = SparseIndex is not None

        vs_config = self.config.vector_store
        url = vs_config.url or ""
        key = vs_config.api_key or ""
        username = vs_config.username or "root"
        self._db_name = vs_config.database_name or "hy_memory"
        base_collection = vs_config.collection_name or "agent_memories"
        self._dims = vs_config.embedding_dims or 1536
        # 自动拼接维度后缀，避免不同 embedding 模型写入同一 collection 导致维度冲突
        self._collection_name = f"{base_collection}_{self._dims}"

        if not url:
            raise ValueError(
                "MEMORY_TENCENT_VDB_URL is required. "
                "Set it in .env or pass via from_config({'vector_store': {'url': '...'}})"
            )

        # 创建 HTTP 客户端
        # 是否在新建 collection 时启用 sparse 全文索引：需 SparseIndex 类 + BM25 编码器都可用
        enable_sparse_on_create = sparse_index_ok and sparse_encoder_ok

        def _init():
            client = tcvectordb.VectorDBClient(
                url=url,
                key=key,
                username=username,
                timeout=30,
            )

            # 确保 database 存在
            client.create_database_if_not_exists(self._db_name)

            created_with_sparse = False
            # 确保 collection 存在
            if not client.exists_collection(self._db_name, self._collection_name):
                index = Index(
                    FilterIndex('id', FieldType.String, IndexType.PRIMARY_KEY),
                    VectorIndex(
                        'vector', self._dims, IndexType.HNSW,
                        MetricType.COSINE, HNSWParams(m=16, efconstruction=200),
                    ),
                    FilterIndex('isolation_key', FieldType.String, IndexType.FILTER),
                    FilterIndex('user_id', FieldType.String, IndexType.FILTER),
                    FilterIndex('agent_id', FieldType.String, IndexType.FILTER),
                    FilterIndex('session_id', FieldType.String, IndexType.FILTER),
                    FilterIndex('layer', FieldType.String, IndexType.FILTER),
                    FilterIndex('status', FieldType.String, IndexType.FILTER),
                    FilterIndex('is_latest', FieldType.Uint64, IndexType.FILTER),
                    FilterIndex('gmt_created', FieldType.Uint64, IndexType.FILTER),
                )
                # 新建 collection 时加入 sparse 全文索引（BM25 / keyword / hybrid）
                if enable_sparse_on_create:
                    try:
                        index.add(SparseIndex(
                            self._sparse_field, FieldType.SparseVector,
                            IndexType.SPARSE_INVERTED, MetricType.IP,
                        ))
                        created_with_sparse = True
                    except Exception as se:
                        logger.warning(f"[vector-store] add SparseIndex failed, dense-only: {se}")
                client.create_collection(
                    database_name=self._db_name,
                    collection_name=self._collection_name,
                    shard=1,
                    replicas=1,
                    description="hy_memory vector store",
                    index=index,
                )
                logger.info(
                    f"Created Tencent VDB collection: {self._db_name}/{self._collection_name} "
                    f"(sparse_fulltext={'on' if created_with_sparse else 'off'})"
                )

            # 探测该 collection 是否含 sparse_vector 索引（兼容已存在的老 collection）
            has_sparse = created_with_sparse
            if not has_sparse:
                try:
                    coll = client.describe_collection(self._db_name, self._collection_name)
                    indexes = getattr(coll, "indexes", None) or {}
                    names = set()
                    if isinstance(indexes, dict):
                        names = set(indexes.keys())
                    else:
                        for idx in indexes:
                            n = getattr(idx, "field_name", None) or getattr(idx, "name", None)
                            if n:
                                names.add(n)
                    has_sparse = self._sparse_field in names
                except Exception as de:
                    logger.debug(f"[vector-store] describe_collection probe failed: {de}")
                    has_sparse = False

            return client, has_sparse

        self._client, has_sparse = await _run_in_vdb_pool(_init)
        # 最终 fulltext 能力 = collection 有 sparse 索引 且 BM25 编码器可用
        self._supports_fulltext = bool(has_sparse and sparse_encoder_ok)
        if has_sparse and not sparse_encoder_ok:
            logger.warning(
                "[vector-store] collection has sparse index but BM25Encoder "
                "(tcvdb_text) unavailable → fulltext disabled; pip install tcvdb_text"
            )
        logger.info(
            f"VectorStore initialized (Tencent VDB), "
            f"db={self._db_name} collection={self._collection_name} "
            f"fulltext={'on' if self._supports_fulltext else 'off'}"
        )

    async def close(self) -> None:
        if self._client:
            try:
                await _run_in_vdb_pool(self._client.close)
            except Exception:
                pass
            self._client = None

    # ================================================================
    # 写入
    # ================================================================

    def _maybe_sparse(self, payload: Dict[str, Any]) -> Optional[list]:
        """按 search_text 生成 sparse_vector（fulltext 开启时）；否则 None。

        跳过 L1_RAW：原始对话层不被任何召回路径消费（reader 召回 L0/L2/L3/L4，
        提取后又降 SHADOW，System2 也只取 L2/L4），给它编 sparse 纯浪费写入/存储。
        """
        if not self._supports_fulltext:
            return None
        if payload.get("layer") == MemoryLayer.L1_RAW.value:
            return None
        from ..pipelines._retrieval import bm25_sparse
        text = payload.get("search_text") or payload.get("content") or ""
        return bm25_sparse.encode_doc(text)

    async def upsert(self, node: MemoryNode) -> str:
        if node.embedding is None:
            raise ValueError(f"Node {node.node_id} has no embedding")

        payload = self._node_to_payload(node)
        doc = {
            "id": node.node_id,
            "vector": node.embedding,
            **{k: self._to_vdb_value(v, k) for k, v in payload.items() if v is not None},
        }
        sparse = self._maybe_sparse(payload)
        if sparse:
            doc[self._sparse_field] = sparse

        def _upsert():
            self._client.upsert(
                database_name=self._db_name,
                collection_name=self._collection_name,
                documents=[doc],
                build_index=True,
            )

        await _run_in_vdb_pool(_upsert)
        logger.debug(
            f"[vector-store] upsert: id={node.node_id} "
            f"layer={node.layer.value if hasattr(node.layer, 'value') else node.layer} "
            f"sparse_terms={len(sparse) if sparse else 0} "
            f"content={node.content}"
        )
        return node.node_id

    async def upsert_batch(self, nodes: List[MemoryNode]) -> List[str]:
        valid_nodes = [n for n in nodes if n.embedding is not None]
        if not valid_nodes:
            return []

        docs = []
        for node in valid_nodes:
            payload = self._node_to_payload(node)
            doc = {
                "id": node.node_id,
                "vector": node.embedding,
                **{k: v for k, v in payload.items() if v is not None},
            }
            sparse = self._maybe_sparse(payload)
            if sparse:
                doc[self._sparse_field] = sparse
            docs.append(doc)

        def _upsert_batch():
            # 腾讯 VDB 单次 upsert 限制，分批
            batch_size = 100
            for i in range(0, len(docs), batch_size):
                self._client.upsert(
                    database_name=self._db_name,
                    collection_name=self._collection_name,
                    documents=docs[i:i + batch_size],
                    build_index=True,
                )

        await _run_in_vdb_pool(_upsert_batch)
        return [n.node_id for n in valid_nodes]

    async def update_embedding(self, node_id: str, embedding: List[float]) -> bool:
        try:
            def _update():
                self._client.update(
                    database_name=self._db_name,
                    collection_name=self._collection_name,
                    data={
                        "id": node_id,
                        "vector": embedding,
                    },
                )
            await _run_in_vdb_pool(_update)
            return True
        except Exception as e:
            logger.warning(f"Failed to update embedding for {node_id}: {e}")
            return False

    async def update_payload(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """更新 payload 字段；若 updates 含 'embedding'，同时更新向量（写入 vector 字段）。"""
        if not updates:
            return True
        try:
            updates = dict(updates)
            new_embedding = updates.pop("embedding", None)  # 取出向量单独处理
            # 转换值为 VDB 兼容格式
            data = {}
            for k, v in updates.items():
                data[k] = self._to_vdb_value(v, k)
            if new_embedding is not None:
                data["vector"] = new_embedding  # tencent update 支持直接更新向量

            def _update():
                self._client.update(
                    database_name=self._db_name,
                    collection_name=self._collection_name,
                    data=data,
                    document_ids=[node_id],
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

    # payload 中需要返回的字段
    _OUTPUT_FIELDS = [
        "id", "node_id", "isolation_key", "user_id", "agent_id", "session_id",
        "owner",
        "layer", "content", "status",
        "confidence", "source_type",
        "emotional_valence", "emotional_arousal",
        "specificity_score", "rarity_score", "longtail_flag",
        "meta_tags", "source_session_id", "source_turn_index", "evidence_chain",
        "gmt_created", "gmt_modified", "memory_at", "observed_at",
        "temporal_relation", "event_time_text", "event_start", "event_end",
        "normalization_confidence",
        "valid_from", "valid_until", "temporal_anchor",
        "access_count", "last_accessed_at",
        "supersedes", "superseded_by", "is_latest",
        "speculate", "tags", "source_raw_memory_id",
        "custom",
    ]

    def _build_filter_expr(
        self,
        *,
        isolation_key: str = "",
        isolation_keys: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        layers: Optional[List] = None,
        status_filter: Optional[List] = None,
        only_latest: bool = True,
        created_after: Optional[float] = None,
    ) -> Optional[str]:
        """构建 tcvectordb filter 表达式（search / keyword_search / hybrid 共用）。"""
        parts: List[str] = []

        if isolation_keys:
            keys_str = ",".join(f'"{k}"' for k in isolation_keys)
            parts.append(f"isolation_key in ({keys_str})")
        elif isolation_key:
            parts.append(f'isolation_key = "{isolation_key}"')
        else:
            effective_uids = user_ids if user_ids else ([user_id] if user_id else [])
            if effective_uids:
                if len(effective_uids) == 1:
                    parts.append(f'user_id = "{effective_uids[0]}"')
                else:
                    uids_str = ",".join(f'"{u}"' for u in effective_uids)
                    parts.append(f"user_id in ({uids_str})")
                if agent_ids:
                    if len(agent_ids) == 1:
                        parts.append(f'agent_id = "{agent_ids[0]}"')
                    else:
                        aids_str = ",".join(f'"{a}"' for a in agent_ids)
                        parts.append(f"agent_id in ({aids_str})")

        if layers:
            layer_values = [l.value if hasattr(l, 'value') else str(l) for l in layers]
            if len(layer_values) == 1:
                parts.append(f'layer = "{layer_values[0]}"')
            else:
                lvs = ",".join(f'"{v}"' for v in layer_values)
                parts.append(f"layer in ({lvs})")

        if status_filter:
            status_values = [s.value if hasattr(s, 'value') else str(s) for s in status_filter]
            if len(status_values) == 1:
                parts.append(f'status = "{status_values[0]}"')
            else:
                svs = ",".join(f'"{v}"' for v in status_values)
                parts.append(f"status in ({svs})")
        else:
            parts.append(f'status = "{MemoryStatus.ACTIVE.value}"')

        # 演化链末端过滤：only_latest=True 时只搜索 is_latest=1 的节点
        # 注意：_to_vdb_value 把 bool 转成 int（0/1），所以 filter 用 1 而非 true
        if only_latest:
            parts.append("is_latest = 1")

        # 时间过滤：gmt_created >= int(created_after)
        if created_after is not None:
            parts.append(f"gmt_created >= {int(created_after)}")

        return " and ".join(parts) if parts else None

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
        """语义检索"""
        from tcvectordb.model.document import Filter

        filter_expr = self._build_filter_expr(
            isolation_key=isolation_key,
            isolation_keys=isolation_keys,
            user_id=user_id,
            user_ids=user_ids,
            agent_ids=agent_ids,
            layers=layers,
            status_filter=status_filter,
            only_latest=only_latest,
            created_after=created_after,
        )

        def _search():
            kwargs = {
                "database_name": self._db_name,
                "collection_name": self._collection_name,
                "vectors": [query_embedding],
                "limit": limit,
                "output_fields": self._OUTPUT_FIELDS,
                "retrieve_vector": False,
            }
            if filter_expr:
                kwargs["filter"] = Filter(filter_expr)
            return self._client.search(**kwargs)

        results = await _run_in_vdb_pool(_search)

        output = []
        # results 是 List[List[Dict]]，第一层对应每个查询向量
        hits = results[0] if results else []
        for hit in hits:
            score = hit.get("score", 0.0)
            if (score_threshold or 0) > 0 and score < score_threshold:
                continue

            node = self._payload_to_node(hit)
            output.append({
                "node_id": hit.get("id", ""),
                "score": score,
                "node": node,
            })

        logger.debug(
            f"[vector-store] search: user_ids={user_ids or user_id} "
            f"agent_ids={agent_ids} limit={limit} found={len(output)}"
        )
        return output

    # ================================================================
    # Keyword / Full-text Search (sparse vector + BM25)
    # ================================================================

    @property
    def supports_fulltext(self) -> bool:
        """该 collection 是否支持 sparse 全文检索（含 BM25 编码器可用）。"""
        return self._supports_fulltext

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        user_id: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
        layers: Optional[List[MemoryLayer]] = None,
        status_filter: Optional[List[MemoryStatus]] = None,
        only_latest: bool = True,
        isolation_key: str = "",
        isolation_keys: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        基于 sparse vector (BM25) 的全文关键词检索。

        collection 无 sparse 索引或 BM25 编码器不可用时降级返回 []（一次性告警）。
        返回与 search() 同构：[{"node_id", "score"(BM25), "node"}, ...]

        session 隔离：传 isolation_key/isolation_keys（user::agent::session 复合键）时，
        过滤精确到 session；否则退化到 user_id + agent_ids 过滤。
        """
        if not self._supports_fulltext:
            if not getattr(self, "_warned_no_fulltext", False):
                self._warned_no_fulltext = True
                logger.warning(
                    "[vector-store] keyword_search degraded to empty: collection "
                    f"{self._collection_name} has no sparse index (or tcvdb_text "
                    "missing). Recreate collection with SparseIndex + backfill "
                    "sparse_vector to enable BM25/hybrid."
                )
            return []

        from ..pipelines._retrieval import bm25_sparse
        from tcvectordb.model.document import Filter

        q_sparse = bm25_sparse.encode_query(query)
        if not q_sparse:
            return []

        filter_expr = self._build_filter_expr(
            isolation_key=isolation_key,
            isolation_keys=isolation_keys,
            user_id=user_id,
            agent_ids=agent_ids,
            layers=layers,
            status_filter=status_filter,
            only_latest=only_latest,
        )

        def _fts():
            kwargs = {
                "database_name": self._db_name,
                "collection_name": self._collection_name,
                "data": q_sparse,
                "field_name": self._sparse_field,
                "output_fields": self._OUTPUT_FIELDS,
                "retrieve_vector": False,
                "limit": top_k,
            }
            if filter_expr:
                kwargs["filter"] = Filter(filter_expr)
            return self._client.fulltext_search(**kwargs)

        try:
            results = await _run_in_vdb_pool(_fts)
        except Exception as e:
            logger.warning(f"[vector-store] keyword_search(fulltext) failed: {e}")
            return []

        # fulltext_search 返回 List[Dict]（单路），不是 search 的 List[List[Dict]]
        hits = results[0] if (results and isinstance(results[0], list)) else results
        output = []
        for hit in hits or []:
            output.append({
                "node_id": hit.get("id", ""),
                "score": hit.get("score", 0.0),
                "node": self._payload_to_node(hit),
            })
        logger.debug(
            f"[vector-store] keyword_search: user_id={user_id} limit={top_k} "
            f"found={len(output)}"
        )
        return output

    async def hybrid_search_native(
        self,
        query_embedding: List[float],
        query_text: str,
        *,
        user_id: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        isolation_key: str = "",
        isolation_keys: Optional[List[str]] = None,
        layers: Optional[List[MemoryLayer]] = None,
        status_filter: Optional[List[MemoryStatus]] = None,
        only_latest: bool = True,
        created_after: Optional[float] = None,
        limit: int = 10,
        ann_limit: int = 0,
        kw_limit: int = 0,
        w_dense: float = 0.6,
        w_sparse: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """
        腾讯云 native hybrid_search：ANN(dense) + KeywordSearch(sparse BM25) +
        WeightedRerank，融合在 DB 侧完成。

        返回每条带：score(rerank后最终分) + _dense(ann分) + _sparse(bm25分)，
        供 reader 在 pipeline log 中分别展示各通道得分。

        不支持 fulltext 时返回 None（调用方自行回落到纯向量 search）。
        """
        if not self._supports_fulltext:
            return None

        from ..pipelines._retrieval import bm25_sparse
        from tcvectordb.model.document import Filter, AnnSearch, KeywordSearch, WeightedRerank

        q_sparse = bm25_sparse.encode_query(query_text) if query_text else None
        if not q_sparse:
            return None

        ann_limit = ann_limit or max(limit * 3, 30)
        kw_limit = kw_limit or max(limit * 3, 30)
        filter_expr = self._build_filter_expr(
            isolation_key=isolation_key,
            isolation_keys=isolation_keys,
            user_id=user_id,
            user_ids=user_ids,
            agent_ids=agent_ids,
            layers=layers,
            status_filter=status_filter,
            only_latest=only_latest,
            created_after=created_after,
        )

        def _hybrid():
            flt = Filter(filter_expr) if filter_expr else None
            return self._client.hybrid_search(
                database_name=self._db_name,
                collection_name=self._collection_name,
                ann=[AnnSearch(
                    field_name="vector",
                    data=query_embedding,
                    limit=ann_limit,
                )],
                match=[KeywordSearch(
                    field_name=self._sparse_field,
                    data=q_sparse,
                    limit=kw_limit,
                )],
                rerank=WeightedRerank(
                    field_list=["vector", self._sparse_field],
                    weight=[w_dense, w_sparse],
                ),
                filter=flt,
                retrieve_vector=False,
                output_fields=self._OUTPUT_FIELDS,
                limit=limit,
            )

        try:
            results = await _run_in_vdb_pool(_hybrid)
        except Exception as e:
            logger.warning(f"[vector-store] hybrid_search_native failed: {e}")
            return None

        hits = results[0] if (results and isinstance(results[0], list)) else results
        output = []
        for hit in hits or []:
            output.append({
                "node_id": hit.get("id", ""),
                "score": hit.get("score", 0.0),
                # 腾讯云 hybrid 返回融合分；如 SDK 暴露分路分数则带出，否则置 None
                "_dense": hit.get("ann_score", hit.get("vector_score")),
                "_sparse": hit.get("match_score", hit.get("sparse_score")),
                "node": self._payload_to_node(hit),
            })
        logger.debug(
            f"[vector-store] hybrid_search_native: user_id={user_id} "
            f"limit={limit} found={len(output)}"
        )
        return output


    async def get_by_id(self, node_id: str) -> Optional[MemoryNode]:
        try:
            def _query():
                return self._client.query(
                    database_name=self._db_name,
                    collection_name=self._collection_name,
                    document_ids=[node_id],
                    output_fields=self._OUTPUT_FIELDS,
                    retrieve_vector=False,
                )

            docs = await _run_in_vdb_pool(_query)
            if not docs:
                return None
            return self._payload_to_node(docs[0])
        except Exception as e:
            logger.warning(f"Failed to get vector point {node_id}: {e}")
            return None

    async def get_embeddings(self, node_ids: List[str]) -> Dict[str, List[float]]:
        """批量取向量（retrieve_vector=True），供去重本地算 cosine。

        tencent SDK doc 的向量在 'vector' 字段；from_dict 读的是 'embedding'，
        故这里直接从 doc 读 'vector'，不经 _payload_to_node。
        """
        if not node_ids:
            return {}
        try:
            def _query():
                return self._client.query(
                    database_name=self._db_name,
                    collection_name=self._collection_name,
                    document_ids=list(node_ids),
                    output_fields=["node_id"],
                    retrieve_vector=True,
                )

            docs = await _run_in_vdb_pool(_query)
            out: Dict[str, List[float]] = {}
            for d in (docs or []):
                # SDK doc 可能是 dict 或带属性的对象
                nid = d.get("node_id") if isinstance(d, dict) else getattr(d, "node_id", None)
                if not nid:
                    nid = d.get("id") if isinstance(d, dict) else getattr(d, "id", None)
                vec = d.get("vector") if isinstance(d, dict) else getattr(d, "vector", None)
                if nid and vec:
                    out[str(nid)] = list(vec)
            return out
        except Exception as e:
            logger.warning(f"Failed to batch get embeddings {node_ids}: {e}")
            return {}

    # ================================================================
    # 删除
    # ================================================================

    async def delete(self, node_id: str) -> bool:
        try:
            def _delete():
                self._client.delete(
                    database_name=self._db_name,
                    collection_name=self._collection_name,
                    document_ids=[node_id],
                )
            await _run_in_vdb_pool(_delete)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete vector point {node_id}: {e}")
            return False

    async def delete_by_isolation_key(self, isolation_key: str) -> int:
        try:
            from tcvectordb.model.document import Filter

            def _delete_batch():
                return self._client.delete(
                    database_name=self._db_name,
                    collection_name=self._collection_name,
                    filter=Filter(f'isolation_key = "{isolation_key}"'),
                    limit=16384,
                )

            total = 0
            while True:
                result = await _run_in_vdb_pool(_delete_batch)
                affected = result.get("affectedCount", 0) if isinstance(result, dict) else 0
                total += affected
                if affected < 16384:
                    break  # 删完了
            return total if total > 0 else -1
        except Exception as e:
            logger.warning(f"Failed to delete vectors for {isolation_key}: {e}")
            return -1

    async def delete_by_metadata(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        try:
            from tcvectordb.model.document import Filter

            parts = [f'user_id = "{user_id}"']
            if agent_id:
                parts.append(f'agent_id = "{agent_id}"')
            if session_id:
                parts.append(f'session_id = "{session_id}"')
            filter_expr = " and ".join(parts)

            def _delete_batch():
                return self._client.delete(
                    database_name=self._db_name,
                    collection_name=self._collection_name,
                    filter=Filter(filter_expr),
                    limit=16384,
                )

            total = 0
            while True:
                result = await _run_in_vdb_pool(_delete_batch)
                affected = result.get("affectedCount", 0) if isinstance(result, dict) else 0
                total += affected
                if affected < 16384:
                    break
            return total if total > 0 else -1
        except Exception as e:
            logger.warning(f"Failed to delete vectors by metadata (user_id={user_id}): {e}")
            return -1

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
        """枚举某用户的记忆节点（含 embedding）"""
        try:
            from tcvectordb.model.enum import ReadConsistency

            parts = [f'user_id = "{user_id}"']
            if agent_id:
                parts.append(f'agent_id = "{agent_id}"')
            if layers:
                layer_values = [l.value if hasattr(l, 'value') else str(l) for l in layers]
                if len(layer_values) == 1:
                    parts.append(f'layer = "{layer_values[0]}"')
                else:
                    lvs = ", ".join(f'"{v}"' for v in layer_values)
                    parts.append(f'layer in ({lvs})')
            if status_filter:
                status_values = [s.value if hasattr(s, 'value') else str(s) for s in status_filter]
                if len(status_values) == 1:
                    parts.append(f'status = "{status_values[0]}"')
                else:
                    in_clause = ", ".join(f'"{v}"' for v in status_values)
                    parts.append(f'status in ({in_clause})')
            filter_expr = " and ".join(parts)

            from tcvectordb.model.document import Filter

            nodes = []
            offset = 0
            batch_size = min(limit, 100)

            def _query(off):
                return self._client.query(
                    database_name=self._db_name,
                    collection_name=self._collection_name,
                    filter=Filter(filter_expr),
                    output_fields=self._OUTPUT_FIELDS,
                    retrieve_vector=True,
                    limit=batch_size,
                    offset=off,
                )

            while len(nodes) < limit:
                docs = await _run_in_vdb_pool(_query, offset)
                if not docs:
                    break
                for doc in docs:
                    node = self._payload_to_node(doc)
                    vec = getattr(doc, "vector", None)
                    if vec:
                        node.embedding = vec
                    nodes.append(node)
                if len(docs) < batch_size:
                    break
                offset += batch_size

            logger.info(f"[vector-store] list_by_user: user_id={user_id} agent_id={agent_id} found={len(nodes)}")
            return nodes[:limit]
        except Exception as e:
            logger.error(f"list_by_user failed: {e}", exc_info=True)
            return []

    # ================================================================
    # 统计
    # ================================================================

    async def get_stats(self) -> Dict[str, Any]:
        try:
            def _stats():
                coll = self._client.describe_collection(
                    self._db_name, self._collection_name
                )
                return {
                    "collection": self._collection_name,
                    "database": self._db_name,
                    "document_count": getattr(coll, 'document_count', -1),
                }
            return await _run_in_vdb_pool(_stats)
        except Exception as e:
            return {"error": str(e)}

    async def count(self, isolation_key: str) -> int:
        try:
            from tcvectordb.model.document import Filter

            def _count():
                return self._client.count(
                    database_name=self._db_name,
                    collection_name=self._collection_name,
                    filter=Filter(f'isolation_key = "{isolation_key}"'),
                )
            return await _run_in_vdb_pool(_count)
        except Exception as e:
            logger.warning(f"Failed to count vectors for {isolation_key}: {e}")
            return 0

    # 需要转 int 的时间戳字段（VDB 索引为 uint64，不接受 float）
    _TIMESTAMP_FIELDS = {
        "gmt_created", "gmt_modified", "memory_at", "observed_at",
        "event_start", "event_end", "valid_from", "valid_until",
        "last_accessed_at",
    }

    @staticmethod
    def _to_vdb_value(v: Any, field_name: str = "") -> Any:
        """腾讯云 VDB 类型适配：bool → int, float timestamp → int"""
        if isinstance(v, bool):
            return 1 if v else 0
        # 时间戳字段：float → int（云端索引为 uint64）
        if field_name in TencentVectorStore._TIMESTAMP_FIELDS and isinstance(v, float):
            return int(v)
        return v
