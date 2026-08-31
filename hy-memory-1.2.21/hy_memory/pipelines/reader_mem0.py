"""
Mem0-style Read Pipeline (reader_mem0).

忠实复刻 mem0 OSS v3 hybrid search 的打分逻辑（见
`mem0/SEARCH_PIPELINE.md` 与 `mem0/utils/scoring.py`），用于与本项目其他
reader 做对比实验。

与 mem0 的对应关系：
  - Semantic over-fetch：`internal_limit = max(top_k * 4, 60)` 主召回池
  - Keyword(BM25)：`vector_store.keyword_search()` + sigmoid 归一化
    （midpoint/steepness 随 query 词数自适应，见 lemmatize.get_bm25_params）
  - **候选池 = semantic only**：BM25 只为已进入语义池的 memory 加分；
    纯关键词命中、语义未召回的不会出现在结果里（mem0 设计约束 6.1）
  - score_and_rank：`combined = (semantic + bm25 + entity) / max_possible`，
    **max_possible 是 per-query 全局分母**（不是每条 mem 各自的权重）：
      仅 semantic=1.0 / +BM25=2.0 / +entity=1.5 / 三者=2.5。
    是否启用某路加权，由「整个召回池里该路是否有命中」统一决定——一个 query 里
    两条很接近的 mem，不会因为一条命中 BM25、另一条没命中就走不同分母。
  - **threshold 只 gate semantic**：融合前 `semantic < threshold` 直接丢弃，
    BM25 / entity 救不回（mem0 设计约束 6.2）

entity boost（mem0 §3.6）：
  - `_compute_entity_boosts()` 接 entity store（独立 {collection}_entities collection）：
    从 query 抽 entity → embed → `vector_store.search_entities` → 按 mem0 公式
    `boost = similarity × 0.5 × 1/(1+0.001×(num_linked-1)²)`（similarity ≥ 0.5，
    同一 memory 取 max）。命中即 has_entity=True，分母升到 1.5 / 2.5。
  - **对齐 mem0：entity boost 总是自动尝试，不受写入开关控制。** 是否生效完全由
    数据/能力自然决定，下列任一不满足即返回 {}（has_entity=False，分母不变）：
    后端实现 entity store（chroma/qdrant）+ spaCy 可用 + query 抽到 entity +
    entity store 里有该 user 的相关 entity。
  - 写入侧 entity 怎么进库：add 时开 `MEMORY_ENTITY_STORE_ENABLED`（写入开关），
    或对旧数据调 `client.build_entity_store(user_id)` 迁移。该 env **只**控制写入，
    不影响 reader 行为。

与 mem0 的差异（本项目不具备的能力）：
  - **无 graph / evolution / strength**：纯 VDB 语义 + BM25 (+ 未来 entity) 融合

分层处理：
  - mem0 没有 layer 概念（整个 collection 一个扁平池）。本 reader 在等价的「抽取
    记忆」层（L0/L2/L3/L4）上跑同一个扁平池，不做 profile / intention 旁路——
    与 mem0 严格对齐：只传 query 即采用 mem0 默认 top_k=20 / threshold=0.1。
  - 不受 client.search 的 limit / min_score 影响（那是为本项目其他 reader 调的）。
    需要为对比实验微调时改 env `HY_MEMORY_MEM0_TOP_K` / `HY_MEMORY_MEM0_THRESHOLD`。
"""

import os
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from .base import (
    ReadPipeline, ReadRequest, ReadResponse, PipelineContext,
    memory_temporal_fields, memory_provenance_fields,
)
from ..config import MemoryConfig
from ..core.embed_service import EmbedService
from ..models.memory import MemoryNode, MemoryLayer
from ..data.vector_store import create_vector_store
from ..data.vector_store_base import VectorStoreBase
from ..data.graph_store_base import GraphStoreBase
from ..utils.tracer import PipelineTracer, create_tracer
from ._retrieval.lemmatize import lemmatize_for_bm25, get_bm25_params
from ._retrieval.scoring import normalize_bm25
from ._retrieval.trace import ReadTraceLogger

logger = logging.getLogger(__name__)

# mem0 OSS search() 默认参数（mem0/memory/main.py::search）。
# reader_mem0 严格对齐：只传 query 时即采用这些默认值，不受 client.search 的
# limit/min_score（那是为本项目其他 reader 调的）影响。
# 需要为对比实验微调时改这两个 env，相当于改 mem0 的 search 默认值。
MEM0_DEFAULT_TOP_K = 20       # mem0 top_k 默认 20
MEM0_DEFAULT_THRESHOLD = 0.1  # mem0 threshold 默认 0.1

# mem0 没有 layer 概念：整个 collection 一个扁平池。本项目里存放「抽取出的记忆」
# 的等价层 = L0/L2/L3/L4（basic_info / fact / summary / identity）。
# 不含 L1_RAW（原始对话，append-only，mem0 不存）、L5_KNOWLEDGE，
# 以及 L6_SCHEMA / L7_INTENTION（本项目 graph/intention 概念，mem0 没有）。
_MEM0_POOL_LAYERS = [
    MemoryLayer.L0_BASIC_INFO,
    MemoryLayer.L2_FACT,
    MemoryLayer.L3_SUMMARY,
    MemoryLayer.L4_IDENTITY,
]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


ENTITY_BOOST_WEIGHT = 0.5  # mem0/utils/scoring.py::ENTITY_BOOST_WEIGHT


def score_and_rank_mem0(
    semantic_results: List[Dict[str, Any]],
    bm25_scores: Dict[str, float],
    entity_boosts: Dict[str, float],
    threshold: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    忠实复刻 mem0 `score_and_rank()`（含 entity boost）。

    - 候选池仅为 semantic_results；
    - threshold 在融合**前** gate semantic_score（BM25 / entity 都救不回）；
    - **max_possible 是「整个 query」级别的全局分母，不是每条 mem 各自的**：
        仅 semantic               → 1.0
        semantic + BM25           → 2.0   （bm25_scores 非空即触发，对池内全部 mem 生效）
        semantic + entity         → 1.5   （entity_boosts 非空）
        semantic + BM25 + entity  → 2.5
      所以一个 query 里两条很接近的 mem，不会因为一条命中 BM25、另一条没命中就走
      不同分母——是否启用某路加权由「整个池子是否有该路命中」统一决定。
    - combined = min((semantic + bm25 + entity) / max_possible, 1.0)；
    - 按 combined 降序，取 top_k。

    semantic_results 每条须含 {"node_id", "node", "score"}。
    bm25_scores / entity_boosts: {node_id(str): score}，未命中视为 0。
    返回每条含 {"node_id", "node", "score"(=combined), "_semantic", "_bm25", "_entity"}。
    """
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)

    # 全局分母（per-query，不是 per-memory）
    max_possible = 1.0
    if has_bm25:
        max_possible += 1.0
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT

    scored: List[Dict[str, Any]] = []
    for r in semantic_results:
        nid = r.get("node_id")
        if nid is None:
            continue
        semantic_score = r.get("score") or 0.0
        # threshold 只 gate semantic，BM25 / entity 救不回
        if semantic_score < threshold:
            continue
        nid_str = str(nid)
        bm25_score = bm25_scores.get(nid_str, 0.0)
        entity_boost = entity_boosts.get(nid_str, 0.0)
        raw_combined = semantic_score + bm25_score + entity_boost
        combined = min(raw_combined / max_possible, 1.0)
        scored.append({
            "node_id": nid,
            "node": r.get("node"),
            "score": combined,
            "_semantic": semantic_score,
            "_bm25": bm25_score,
            "_entity": entity_boost,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


class Mem0ReadPipeline(ReadPipeline):
    """
    Mem0 风格 reader：semantic over-fetch + BM25 sigmoid + entity boost 融合，
    候选池仅 semantic，max_possible 为 per-query 全局分母（1.0/1.5/2.0/2.5）。

    严格对齐 mem0：单一扁平池，无 graph / evolution / strength / profile 旁路。
    entity boost 通过 `_compute_entity_boosts()` 计算；当前未接 entity store 时
    返回 {}（即 has_entity=False，分母不含 0.5）。等 memory 刷上 entity 后，
    只要实现 entity store 检索即自动启用 1.5 / 2.5 分母。
    """

    VERSION = "mem0"

    def __init__(
        self,
        config: MemoryConfig,
        embed_service: Optional[EmbedService] = None,
        vector_store: Optional[VectorStoreBase] = None,
        graph_store: Optional[GraphStoreBase] = None,
        cache: Any = None,
    ):
        self.config = config
        self._embed_service = embed_service
        self._external_vector_store = vector_store
        self._graph_store = graph_store  # 接受但不使用（mem0 无 graph）
        self._cache = cache
        self._vector_store: Optional[VectorStoreBase] = None
        self._vector_store_initialized = False
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self._embed_service is None:
            self._embed_service = EmbedService(self.config)
        self._vector_store = self._external_vector_store or create_vector_store(self.config)
        if self._external_vector_store and getattr(self._external_vector_store, "_client", None):
            self._vector_store_initialized = True
        self._initialized = True
        logger.debug("Mem0ReadPipeline initialized")

    @property
    def embed_service(self) -> EmbedService:
        if self._embed_service is None:
            self._embed_service = EmbedService(self.config)
        return self._embed_service

    async def _get_vector_store(self) -> VectorStoreBase:
        if self._vector_store is None:
            self._vector_store = self._external_vector_store or create_vector_store(self.config)
        if not self._vector_store_initialized:
            await self._vector_store.initialize()
            self._vector_store_initialized = True
        return self._vector_store

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def read(
        self,
        request: ReadRequest,
        ctx: Optional[PipelineContext] = None,
        tracer: Optional[PipelineTracer] = None,
    ) -> ReadResponse:
        start_time = datetime.now()
        response = ReadResponse()

        if tracer is None:
            tracer = create_tracer(
                operation="read",
                pipeline_version="mem0",
                uid=request.user_id,
                agent_id=request.agent_id,
                content_preview=request.query,
            )

        trace_log = ReadTraceLogger(
            cache=self._cache,
            request_id=request.request_id,
            user_id=request.user_id or (request.user_ids[0] if request.user_ids else ""),
            agent_id=request.agent_id or "default_agent",
            reader_version=self.VERSION,
            session_id=request.session_id or (request.session_ids[0] if len(request.session_ids) == 1 else ""),
        )

        try:
            if not request.query:
                response.error_code = 400
                response.error_message = "query is required"
                return response

            # mem0 默认值（不读 request.limit / request.min_score——那是本项目其他
            # reader 的调参）。需对比微调时改 env，相当于改 mem0 search 的默认值。
            top_k = _int_env("HY_MEMORY_MEM0_TOP_K", MEM0_DEFAULT_TOP_K)
            threshold = _float_env("HY_MEMORY_MEM0_THRESHOLD", MEM0_DEFAULT_THRESHOLD)
            # mem0: internal_limit = max(top_k * 4, 60)
            internal_limit = max(top_k * 4, 60)

            await trace_log.log_request(
                query=request.query,
                limit=top_k,
                layers=list(request.layers) if request.layers else None,
                min_score=threshold,
                user_ids=list(request.user_ids or []),
                agent_ids=list(request.agent_ids or []),
                session_ids=list(request.session_ids or []),
            )

            # =========================================================
            # Stage 1: Embed + Lemmatize query (mem0 §3.2)
            # =========================================================
            _t_embed = datetime.now()
            query_embedding = request.query_embedding
            cache_hit = False
            if not query_embedding:
                query_embedding = await self.embed_service.embed(request.query)
            else:
                cache_hit = True
            query_lemmatized = lemmatize_for_bm25(request.query)
            _embed_ms = (datetime.now() - _t_embed).total_seconds() * 1000
            await trace_log.log_embed_query(
                query=request.query,
                dims=len(query_embedding),
                cache_hit=cache_hit,
                elapsed_ms=_embed_ms,
            )

            # =========================================================
            # Stage 2: Isolation params
            # =========================================================
            isolation_params = self._build_isolation_params(request)
            if isolation_params.get("error_msg"):
                response.error_code = 400
                response.error_message = isolation_params["error_msg"]
                return response

            search_kwargs = {
                "isolation_key": isolation_params.get("isolation_key", ""),
                "isolation_keys": isolation_params.get("isolation_keys"),
                "user_ids": isolation_params.get("user_ids"),
                "agent_ids": isolation_params.get("agent_ids"),
            }
            kw_user_id = request.user_id or (request.user_ids[0] if request.user_ids else "")
            kw_agent_ids = request.agent_ids if request.agent_ids else (
                [request.agent_id] if request.agent_id else None
            )

            # =========================================================
            # Stage 3: Parallel recall — semantic(over-fetch) + keyword(BM25)
            #          (mem0 §3.3 / §3.4) — 单一扁平池，无 profile/intention 旁路
            # =========================================================
            import asyncio
            _t_recall = datetime.now()
            vector_store = await self._get_vector_store()

            recall_tasks = [
                # semantic 主召回（over-fetch 池）
                vector_store.search(
                    query_embedding=query_embedding,
                    layers=_MEM0_POOL_LAYERS,
                    limit=internal_limit,
                    **search_kwargs,
                ),
                # keyword（BM25）召回
                vector_store.keyword_search(
                    query=query_lemmatized,
                    top_k=internal_limit,
                    user_id=kw_user_id,
                    agent_ids=kw_agent_ids,
                    layers=_MEM0_POOL_LAYERS,
                ),
            ]
            results = await asyncio.gather(*recall_tasks, return_exceptions=True)
            semantic_hits = results[0] if not isinstance(results[0], Exception) else []
            keyword_hits = results[1] if not isinstance(results[1], Exception) else []
            if isinstance(results[0], Exception):
                logger.warning(f"[mem0] semantic search failed: {results[0]}")
            if isinstance(results[1], Exception):
                logger.warning(f"[mem0] keyword search failed: {results[1]}")

            _recall_ms = (datetime.now() - _t_recall).total_seconds() * 1000
            await trace_log.log_recall_vec(
                pool_size=internal_limit,
                hits=semantic_hits,
                elapsed_ms=_recall_ms,
            )

            # =========================================================
            # Stage 4: BM25 normalize (mem0 §3.5)
            # =========================================================
            kw_normalized = getattr(vector_store, "keyword_score_normalized", False)
            midpoint, steepness = get_bm25_params(request.query, query_lemmatized)
            bm25_scores: Dict[str, float] = {}
            for r in keyword_hits:
                nid = r["node_id"]
                raw = r["score"]
                if kw_normalized:
                    bm25_scores[nid] = max(0.0, min(float(raw), 1.0))
                else:
                    bm25_scores[nid] = normalize_bm25(raw, midpoint, steepness)

            await trace_log.log_bm25(
                pool_size=len(semantic_hits),
                query_terms=query_lemmatized.split() if query_lemmatized else [],
                hits=keyword_hits[:20],
                raw_hits=keyword_hits,
                bm25_scores=bm25_scores,
                normalize_method="passthrough" if kw_normalized else "sigmoid",
                sigmoid_midpoint=None if kw_normalized else midpoint,
                sigmoid_steepness=None if kw_normalized else steepness,
                has_bm25=bool(bm25_scores),
                elapsed_ms=0,
            )

            # =========================================================
            # Stage 4.5: Entity boost (mem0 §3.6) — 未接 entity store 时返回 {}
            # =========================================================
            _t_entity = datetime.now()
            entity_texts: List[str] = []
            try:
                from ._retrieval.entities import extract_entities
                entity_texts = [
                    t for (_etype, t) in extract_entities(request.query)[:8]
                    if t and t.strip()
                ]
            except Exception:
                entity_texts = []
            entity_boosts = await self._compute_entity_boosts(
                request, query_embedding,
                user_ids=isolation_params.get("user_ids"),
                agent_ids=isolation_params.get("agent_ids"),
                entity_texts=entity_texts,
            )
            await trace_log.log_entity(
                entity_texts=entity_texts,
                boosts=entity_boosts,
                elapsed_ms=(datetime.now() - _t_entity).total_seconds() * 1000,
            )

            # =========================================================
            # Stage 5: score_and_rank (mem0 §3.7) — candidate pool = semantic only
            # =========================================================
            _t_fuse = datetime.now()
            scored = score_and_rank_mem0(
                semantic_hits, bm25_scores, entity_boosts,
                threshold=threshold, top_k=top_k,
            )
            # 重算 per-query 全局分母用于埋点（与 score_and_rank_mem0 内部一致）
            _has_bm25 = bool(bm25_scores)
            _has_entity = bool(entity_boosts)
            _max_possible = 1.0 + (1.0 if _has_bm25 else 0.0) + (
                ENTITY_BOOST_WEIGHT if _has_entity else 0.0
            )
            await trace_log.log_fuse(
                has_bm25=_has_bm25,
                has_entity=_has_entity,
                max_possible=_max_possible,
                candidate_pool=len(semantic_hits),
                threshold=threshold,
                scored=scored,
                elapsed_ms=(datetime.now() - _t_fuse).total_seconds() * 1000,
            )

            # =========================================================
            # Stage 6: format → response.memories（单一扁平结果，无旁路）
            # =========================================================
            top_scores = []
            for item in scored:
                node = item.get("node")
                score = item.get("score", 0.0)
                node_id = item.get("node_id", "")
                content = node.content if node else item.get("content", "")
                tags = list(node.tags) if (node and getattr(node, "tags", None)) else []
                mem_entry = {
                    "memory_id": node_id,
                    "content": content,
                    "layer": (node.layer.value if (node and node.layer) else item.get("layer", "")),
                    "score": score,
                    "agent_id": (getattr(node, "agent_id", "") if node else "") or "",
                    "access_count": getattr(node, "access_count", 0) if node else 0,
                    "owner": getattr(node, "owner", None) if node else None,
                    "speculate": getattr(node, "speculate", None) if node else None,
                    "tags": tags,
                    "memory_at": int(node.memory_at.timestamp()) if (node and node.memory_at) else None,
                    "gmt_created": int(node.gmt_created.timestamp()) if (node and node.gmt_created) else None,
                    **memory_temporal_fields(node),
                    **memory_provenance_fields(node),
                }
                response.memories.append(mem_entry)
                top_scores.append(round(score, 4))

            response.total_found = len(scored)
            response.success = True

            logger.info(
                f"[mem0-read] semantic_pool={len(semantic_hits)} keyword={len(keyword_hits)} "
                f"bm25_active={bool(bm25_scores)} top_k={top_k} threshold={threshold} "
                f"returned={len(scored)} for query='{request.query}'"
            )

            await trace_log.log_summary(
                query=request.query,
                intent=None,
                confidence=None,
                is_low_confidence=None,
                channels={"normal": len(scored)},
                total_found=response.total_found,
                elapsed_ms=(datetime.now() - start_time).total_seconds() * 1000,
                returned_memories=response.memories,
            )

        except Exception as e:
            logger.error(f"Mem0ReadPipeline.read failed: {e}", exc_info=True)
            response.error_code = 500
            response.error_message = str(e)
            tracer.set_error(str(e))

        response.elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        tracer.set_output({
            "success": response.success,
            "total_found": response.total_found,
            "pipeline_ms": response.elapsed_ms,
        })
        return response

    # ------------------------------------------------------------------
    # Entity boost (mem0 §3.6)
    # ------------------------------------------------------------------

    async def _compute_entity_boosts(
        self,
        request: ReadRequest,
        query_embedding: List[float],
        *,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        entity_texts: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        计算 per-memory 的 entity boost（复刻 mem0 `_compute_entity_boosts`）。

        流程：
          1. 从 query 抽取 entity（spaCy），去重最多 8 个；
             （调用方可传 `entity_texts` 复用已抽取结果，避免重复 spaCy parse）
          2. 每个 entity embed 后在 entity store 检索（top_k=500）；
          3. similarity >= 0.5 的 match，按其 linked_memory_ids 传播 boost：
               boost = similarity × 0.5 × 1/(1 + 0.001 × (num_linked-1)²)
             同一 memory 被多个 entity 命中取 max。

        **对齐 mem0：entity boost 是 search 的标准一路，总是自动尝试，不受写入侧
        开关（MEMORY_ENTITY_STORE_ENABLED）控制。** 是否真正生效完全由数据/能力
        自然决定，下列任一条件不满足即返回 {}（has_entity=False，分母不变）：
          - vector_store 后端未实现 entity store（非 chroma/qdrant）→ NotImplementedError；
          - spaCy 不可用或 query 抽不到 entity；
          - entity store 里没有该 user 的相关 entity。
        """
        vector_store = await self._get_vector_store()

        if entity_texts is None:
            # 后端不支持 entity store 时直接降级（search_entities 抛 NotImplementedError）
            try:
                from ._retrieval.entities import extract_entities
            except Exception:
                logger.debug("[mem0] entities module unavailable; entity boost skipped")
                return {}
            try:
                extracted = extract_entities(request.query)[:8]
            except Exception as e:
                logger.debug(f"[mem0] entity extraction failed: {e}")
                return {}
            entity_texts = [t for (_etype, t) in extracted if t and t.strip()]

        if not entity_texts:
            return {}

        kw_user_id = (
            (user_ids[0] if user_ids else None)
            or request.user_id
            or (request.user_ids[0] if request.user_ids else "")
        )
        if not kw_user_id:
            return {}

        boosts: Dict[str, float] = {}
        try:
            embeddings = await self.embed_service.embed_batch(entity_texts)
            for emb in embeddings:
                matches = await vector_store.search_entities(
                    query_embedding=emb,
                    user_id=kw_user_id,
                    agent_ids=agent_ids,
                    top_k=500,
                    min_score=0.5,
                )
                for m in matches:
                    similarity = float(m.get("score", 0.0))
                    if similarity < 0.5:
                        continue
                    linked = m.get("linked_memory_ids") or []
                    if not isinstance(linked, list):
                        continue
                    num_linked = max(len(linked), 1)
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight
                    for mid in linked:
                        if mid:
                            k = str(mid)
                            boosts[k] = max(boosts.get(k, 0.0), boost)
        except NotImplementedError:
            logger.debug("[mem0] vector store has no entity store; entity boost skipped")
            return {}
        except Exception as e:
            logger.warning(f"[mem0] entity boost computation failed: {e}")
            return {}

        return boosts

    # ------------------------------------------------------------------
    # Isolation helper (mirrors hybrid_v2)
    # ------------------------------------------------------------------

    def _build_isolation_params(self, request: ReadRequest) -> Dict[str, Any]:
        user_ids = request.user_ids if request.user_ids else ([request.user_id] if request.user_id else [])
        agent_ids = request.agent_ids
        session_ids = request.session_ids

        if len(agent_ids) > 1 and len(session_ids) > 1:
            return {
                "error_msg": (
                    "Cannot specify multiple agent_ids and multiple session_ids simultaneously. "
                    "Use single agent_id with multiple session_ids, or multiple agent_ids without session_ids."
                )
            }

        isolation_key = ""
        isolation_keys = None
        search_user_ids = None
        search_agent_ids = None

        if not agent_ids and not session_ids:
            if request.agent_id:
                keys = [
                    MemoryNode.build_isolation_key(u, request.agent_id or "default")
                    for u in user_ids
                ] if user_ids else None
                if keys and len(keys) == 1:
                    isolation_key = keys[0]
                else:
                    isolation_keys = keys
            else:
                search_user_ids = user_ids if user_ids else None
        elif agent_ids and not session_ids:
            search_user_ids = user_ids if user_ids else None
            search_agent_ids = agent_ids
        else:
            effective_agent_ids = agent_ids if agent_ids else [request.agent_id or "default"]
            isolation_keys = [
                MemoryNode.build_isolation_key(u, a, s)
                for u in (user_ids or ["default"])
                for a in effective_agent_ids
                for s in session_ids
            ]

        return {
            "isolation_key": isolation_key,
            "isolation_keys": isolation_keys,
            "user_ids": search_user_ids,
            "agent_ids": search_agent_ids,
        }

    async def close(self) -> None:
        logger.debug("Mem0ReadPipeline closed")
