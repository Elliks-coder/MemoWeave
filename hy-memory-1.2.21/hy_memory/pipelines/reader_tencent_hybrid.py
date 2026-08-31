# -*- coding: utf-8 -*-
"""
HY Memory - Tencent Native Hybrid Reader

腾讯云 VectorDB 专属 hybrid reader：把 dense(ANN) + sparse(BM25) 召回与
WeightedRerank 融合**下沉到 DB 侧**（client.hybrid_search），而不是在
Python 里 pool+BM25+融合（reader_hybrid_v2 的做法）。

为什么单独做：
  - 性能：召回与融合在 DB 内完成，少一轮数据搬运。
  - 准确：用腾讯云原生 BM25 sparse 检索，而非进程内近似。

可观测性（满足"BM25 得分要在 pipeline log 可见"）：
  - 主召回走 hybrid_search，最终分落 READ_FUSED；
  - 额外单独跑一次 keyword_search（纯 BM25），把每条的 BM25 原始分落
    READ_BM25，确保 inspector 能单独看到关键词通道得分。

启用条件（reader.py dispatcher 判定）：
  - 后端是 TencentVectorStore 且 supports_fulltext=True；
  - 否则 dispatcher 不会选它（回落 hybrid_v2 / legacy）。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import (
    ReadPipeline, ReadRequest, ReadResponse, PipelineContext,
    memory_temporal_fields, memory_provenance_fields,
)
from ..config import MemoryConfig
from ..core.embed_service import EmbedService
from ..data.vector_store_base import VectorStoreBase
from ..data.vector_store import create_vector_store
from ..models.memory import MemoryNode, MemoryLayer, MemoryStatus
from ..utils.tracer import PipelineTracer, create_tracer
from ._retrieval.trace import ReadTraceLogger
from ._retrieval.evolution import expand_evolution_chains
from ._retrieval.intention import recall_intentions
from ._retrieval.strength import apply_strength_to_normal
from ._retrieval import config as _retrieval_config

logger = logging.getLogger(__name__)

_PROFILE_LAYERS = [MemoryLayer.L0_BASIC_INFO, MemoryLayer.L6_SCHEMA]

# 主召回（dense + sparse BM25）覆盖层：fact / summary / identity。
# identity（L4）按 VDB 普通记忆处理，参与 hybrid 融合排序，最终归 normal 通道。
# 与 _PROFILE_LAYERS 互斥（L0/L6 只走 profile 路）：杜绝同节点被两路双重召回、
# 再在 merge 时 profile 高分被 hybrid 融合低分顶掉。
# 不含 L1_RAW（原始对话不召回）、L5_KNOWLEDGE、L7。
_VDB_RECALL_LAYERS = [
    MemoryLayer.L2_FACT,
    MemoryLayer.L3_SUMMARY,
    MemoryLayer.L4_IDENTITY,
]


def _float_env(name: str, default: float) -> float:
    import os
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class TencentHybridReadPipeline(ReadPipeline):
    """腾讯云 native hybrid（ANN + BM25 sparse + WeightedRerank）reader。"""

    VERSION = "tencent_hybrid"

    def __init__(
        self,
        config: MemoryConfig,
        embed_service: Optional[EmbedService] = None,
        vector_store: Optional[VectorStoreBase] = None,
        graph_store: Any = None,
        cache: Any = None,
    ):
        self.config = config
        self._embed_service = embed_service
        self._external_vector_store = vector_store
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
        logger.debug("TencentHybridReadPipeline initialized")

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

    def _build_isolation_params(self, request: ReadRequest) -> Dict[str, Any]:
        """把 user/agent/session 归一成召回用的隔离参数（与 hybrid_v2 同款）。

        session_ids 非空时拼成 isolation_keys（user::agent::session 复合键），
        使 session 过滤精确生效；否则退化到 user_ids + agent_ids。
        """
        user_ids = request.user_ids if request.user_ids else (
            [request.user_id] if request.user_id else []
        )
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
                pipeline_version=self.VERSION,
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

        W_DENSE = _float_env("MEMORY_TENCENT_HYBRID_W_DENSE", 0.6)
        W_SPARSE = _float_env("MEMORY_TENCENT_HYBRID_W_SPARSE", 0.4)
        MIN_SCORE = request.min_score if request.min_score is not None else 0.0

        hybrid_hits: List[Dict[str, Any]] = []
        bm25_hits: List[Dict[str, Any]] = []
        profile_hits: List[Dict[str, Any]] = []

        try:
            await trace_log.log_request(
                query=request.query,
                limit=request.limit,
                layers=list(request.layers) if request.layers else None,
                min_score=MIN_SCORE,
                profile_min_score=request.profile_min_score,
                profile_limit=request.profile_limit,
                user_ids=list(request.user_ids or []),
                agent_ids=list(request.agent_ids or []),
                session_ids=list(request.session_ids or []),
            )

            # Stage 1: embed query
            # 失败降级：Venus embedding 服务偶发卡死时，不抛 500，而是标记 degraded_bm25_only
            # 后续跳过依赖向量的召回通道，只跑 BM25 关键词召回，返回部分结果。
            # 可通过 MEMORY_EMBED_FALLBACK_BM25=false 关闭降级（回到硬失败模式，便于排障）。
            #
            # 硬上限（deadline）：无论 embed_service 内部重试如何计算，整个 embed 调用
            # 最多允许 MEMORY_EMBED_HARD_DEADLINE 秒（默认 10s）。超时立即取消，走 BM25。
            # 这层兜底防的是：慢响应累计、连接层 hang 死、asyncio.CancelledError 穿透等
            # per-attempt timeout 无法覆盖的场景。
            import os as _os
            _fallback_enabled = _os.getenv("MEMORY_EMBED_FALLBACK_BM25", "true").lower() != "false"
            _embed_hard_deadline = float(_os.getenv("MEMORY_EMBED_HARD_DEADLINE", "10"))
            _t = datetime.now()
            query_embedding = request.query_embedding
            cache_hit = bool(query_embedding)
            embed_meta: Dict[str, Any] = {}
            degraded_bm25_only = False
            if not query_embedding:
                try:
                    query_embedding, embed_meta = await asyncio.wait_for(
                        self.embed_service.embed_with_meta(request.query),
                        timeout=_embed_hard_deadline,
                    )
                except (Exception, asyncio.TimeoutError) as _embed_exc:
                    if not _fallback_enabled:
                        raise
                    degraded_bm25_only = True
                    query_embedding = []
                    _is_deadline = isinstance(_embed_exc, asyncio.TimeoutError)
                    embed_meta = {
                        "error": type(_embed_exc).__name__,
                        "error_msg": (str(_embed_exc) or "embed hard deadline exceeded")[:300],
                        "degraded": "bm25_only",
                        "hard_deadline_hit": _is_deadline,
                        "hard_deadline_s": _embed_hard_deadline,
                    }
                    logger.warning(
                        f"[read/tencent_hybrid] embed failed, degrade to BM25-only "
                        f"(deadline_hit={_is_deadline}, budget={_embed_hard_deadline}s): "
                        f"{type(_embed_exc).__name__}: {str(_embed_exc)[:200]}"
                    )
            _embed_ms = (datetime.now() - _t).total_seconds() * 1000
            await trace_log.log_embed_query(
                query=request.query, dims=len(query_embedding),
                cache_hit=cache_hit, elapsed_ms=_embed_ms,
                provider_meta=embed_meta,
            )

            vector_store = await self._get_vector_store()

            # 仅 tencent + fulltext 才走 native hybrid；否则回落纯向量 search
            use_native = (
                hasattr(vector_store, "hybrid_search_native")
                and getattr(vector_store, "supports_fulltext", False)
            )

            user_ids = request.user_ids if request.user_ids else (
                [request.user_id] if request.user_id else []
            )
            agent_ids = request.agent_ids if request.agent_ids else (
                [request.agent_id] if request.agent_id else None
            )
            layers = (
                [MemoryLayer.from_string(l) for l in request.layers]
                if request.layers else _VDB_RECALL_LAYERS
            )
            kw_user_id = request.user_id or (user_ids[0] if user_ids else "")
            final_limit = request.limit

            # session 隔离：把 session_ids 拼成 isolation_keys 复合键（与 hybrid_v2 一致），
            # 传给所有召回通道，使 session 过滤精确生效。缺 session 时退化为 user/agent 过滤。
            iso = self._build_isolation_params(request)
            if iso.get("error_msg"):
                response.error_code = 400
                response.error_message = iso["error_msg"]
                return response
            # 复合键存在时优先用它（含 session）；否则回落到 user_ids/agent_ids
            iso_kwargs = {
                "isolation_key": iso.get("isolation_key", ""),
                "isolation_keys": iso.get("isolation_keys"),
            }
            _has_iso = bool(iso.get("isolation_key") or iso.get("isolation_keys"))
            recall_user_ids = None if _has_iso else (iso.get("user_ids") or user_ids or None)
            recall_agent_ids = None if _has_iso else iso.get("agent_ids")

            # Stage 2: 主召回 —— native hybrid（dense + sparse + rerank）
            # 降级模式下 embedding 为空，跳过 dense 召回，只依赖 Stage 3 的 BM25 通道。
            _t = datetime.now()
            if degraded_bm25_only:
                hybrid_hits = []
            elif use_native:
                hybrid_hits = await vector_store.hybrid_search_native(
                    query_embedding=query_embedding,
                    query_text=request.query,
                    user_ids=recall_user_ids,
                    agent_ids=recall_agent_ids,
                    layers=layers,
                    status_filter=[MemoryStatus.ACTIVE],
                    only_latest=True,
                    limit=max(final_limit * 2, 20),
                    w_dense=W_DENSE,
                    w_sparse=W_SPARSE,
                    **iso_kwargs,
                ) or []
            if not degraded_bm25_only and (not use_native or not hybrid_hits):
                # 回落：纯向量 search
                hybrid_hits = await vector_store.search(
                    query_embedding=query_embedding,
                    user_ids=recall_user_ids,
                    agent_ids=recall_agent_ids,
                    layers=layers,
                    limit=max(final_limit * 2, 20),
                    status_filter=[MemoryStatus.ACTIVE],
                    only_latest=True,
                    **iso_kwargs,
                )
            _recall_ms = (datetime.now() - _t).total_seconds() * 1000
            await trace_log.log_recall_vec(
                pool_size=len(hybrid_hits), hits=hybrid_hits, elapsed_ms=_recall_ms,
            )

            # Stage 3: 单独 BM25 关键词召回（仅为可观测：把 BM25 原始分落 READ_BM25）
            _t = datetime.now()
            if use_native:
                try:
                    bm25_hits = await vector_store.keyword_search(
                        query=request.query,
                        top_k=max(final_limit * 2, 20),
                        user_id=(None if _has_iso else kw_user_id),
                        agent_ids=recall_agent_ids,
                        layers=layers,
                        status_filter=[MemoryStatus.ACTIVE],
                        only_latest=True,
                        **iso_kwargs,
                    )
                except Exception as ke:
                    logger.debug(f"[tencent_hybrid] keyword_search(observe) failed: {ke}")
            _bm25_ms = (datetime.now() - _t).total_seconds() * 1000
            await trace_log.log_bm25(
                pool_size=len(bm25_hits),
                query_terms=[request.query],
                hits=bm25_hits,
                elapsed_ms=_bm25_ms,
            )
            # BM25 分映射（node_id → bm25 score），用于在最终结果上附 _bm25
            bm25_score_by_id = {h.get("node_id", ""): h.get("score", 0.0) for h in bm25_hits}

            # Stage 4: Profile 独立召回
            # 依赖 embedding；降级模式下跳过（BM25 无法覆盖 L0/L6 profile 语义召回）。
            _t = datetime.now()
            if not degraded_bm25_only:
                try:
                    profile_hits = await vector_store.search(
                        query_embedding=query_embedding,
                        user_ids=recall_user_ids,
                        agent_ids=recall_agent_ids,
                        layers=_PROFILE_LAYERS,
                        limit=request.profile_limit,
                        score_threshold=request.profile_min_score,
                        status_filter=[MemoryStatus.ACTIVE],
                        only_latest=True,
                        **iso_kwargs,
                    )
                except Exception as pe:
                    logger.debug(f"[tencent_hybrid] profile recall failed: {pe}")
            _profile_ms = (datetime.now() - _t).total_seconds() * 1000
            await trace_log.log_recall_profile(
                profile_min_score=request.profile_min_score,
                profile_limit=request.profile_limit,
                hits=profile_hits,
                elapsed_ms=_profile_ms,
            )

            # Stage 5: 合并（profile 优先占位）+ min_score 过滤。
            # profile 路（L0/L6）与 hybrid 路（L2/L3/L4）现已互斥，正常不会有同节点
            # 双路命中；下面的 score 复用分支作为兜底保留（万一配置改动导致重叠时，
            # 复用 hybrid 加权分而非 profile 纯向量分）。
            hybrid_score_by_id = {
                r.get("node_id", ""): r.get("score", 0.0) for r in hybrid_hits
            }
            merged: List[Dict[str, Any]] = []
            seen = set()
            for r in profile_hits:
                nid = r.get("node_id", "")
                if nid and nid not in seen:
                    seen.add(nid)
                    # score 复用 hybrid 加权分（若 hybrid 也召回了该节点）
                    if nid in hybrid_score_by_id:
                        r = {**r, "score": hybrid_score_by_id[nid]}
                    merged.append(r)
            for r in hybrid_hits:
                nid = r.get("node_id", "")
                if not nid or nid in seen:
                    continue
                if r.get("score", 0.0) < MIN_SCORE:
                    continue
                seen.add(nid)
                merged.append(r)

            # Memory Strength（默认关闭）：把 idle 衰减 × 频次乘进 normal 通道分数
            # （profile 层 —— L0/L6 —— 不参与），随后重排截断。
            if getattr(self.config.recall, "strength_enabled", False):
                _tau = getattr(self.config.recall, "strength_tau", 180.0)
                _profile_vals = {l.value for l in _PROFILE_LAYERS}
                apply_strength_to_normal(merged, profile_layers=_profile_vals, score_key="score", tau=_tau)

            merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            merged = merged[:final_limit]

            await trace_log.log_merge_profile(
                fused_size=len(hybrid_hits),
                profile_size=len(profile_hits),
                merged_size=len(merged),
                final_limit=final_limit,
            )

            # Stage 6: 演化链回溯（profile L0 链等），同时屏蔽 raw（evolution.py 内已过滤）
            _t = datetime.now()
            expandable = [m for m in merged if m.get("node") is not None]
            expanded = await expand_evolution_chains(vector_store, expandable) if expandable else merged
            _evo_ms = (datetime.now() - _t).total_seconds() * 1000
            await trace_log.log_evolution(
                input_size=len(expandable),
                evolved_count=sum(1 for r in expanded if r.get("is_evolved")),
                elapsed_ms=_evo_ms,
            )
            final_results = expanded[:final_limit]

            # Stage 6b: Proactive 路 —— 召回未过期 intention（L7），过期惰性转 L2_FACT。
            # 仅当 intention_limit > 0 时启用；不占 final_limit 配额，追加在末尾。
            # 依赖 embedding；降级模式下跳过。
            intention_hits: List[Dict[str, Any]] = []
            if request.intention_limit > 0 and not degraded_bm25_only:
                intention_hits = await recall_intentions(
                    vector_store,
                    query_embedding,
                    user_ids=user_ids or None,
                    agent_ids=agent_ids,
                    limit=request.intention_limit,
                    score_threshold=request.intention_min_score,
                    now=request.as_of,
                )

            # Stage 7: 组装响应
            for item in final_results + intention_hits:
                node: Optional[MemoryNode] = item.get("node")
                node_id = item.get("node_id", "")
                if node is None:
                    continue
                mem_entry = {
                    "memory_id": node_id,
                    "content": node.content,
                    "layer": item.get("layer") or node.layer.value,
                    "score": float(item.get("score", 0.0)),
                    "agent_id": getattr(node, "agent_id", "") or "",
                    "access_count": getattr(node, "access_count", 0),
                    "owner": getattr(node, "owner", None),
                    "_bm25": float(bm25_score_by_id.get(node_id, 0.0)),
                    "speculate": getattr(node, "speculate", None),
                    "source_raw_memory_id": getattr(node, "source_raw_memory_id", None),
                    "tags": list(node.tags) if getattr(node, "tags", None) else [],
                    "memory_at": int(node.memory_at.timestamp()) if node.memory_at else None,
                    "gmt_created": int(node.gmt_created.timestamp()) if node.gmt_created else None,
                    "source": item.get("source", "tencent_hybrid"),
                    **memory_temporal_fields(node),
                    **memory_provenance_fields(node),
                }
                if item.get("is_evolved"):
                    mem_entry["evolution_chain"] = item.get("evolution_chain", [])
                response.memories.append(mem_entry)

            response.total_found = len(response.memories)
            response.extra["reader"] = self.VERSION
            response.extra["native_hybrid"] = bool(use_native)
            response.extra["channels"] = {
                "hybrid": len(hybrid_hits),
                "bm25": len(bm25_hits),
                "profile": len(profile_hits),
                "intention": len(intention_hits),
            }
            response.extra["config"] = {"w_dense": W_DENSE, "w_sparse": W_SPARSE, "min_score": MIN_SCORE}
            if degraded_bm25_only:
                # 透明降级：success 仍为 true，通过 extra 暴露给需要感知的调用方。
                response.extra["degraded"] = "embed_failed"
                response.extra["degraded_mode"] = "bm25_only"
                response.extra["partial_result"] = True
            response.success = True

            tracer.set_output({
                "success": True,
                "total_found": response.total_found,
                "native_hybrid": bool(use_native),
                "channels": response.extra["channels"],
            })
            logger.info(
                f"[read/tencent_hybrid] native={use_native} hybrid={len(hybrid_hits)} "
                f"bm25={len(bm25_hits)} profile={len(profile_hits)} "
                f"returned={len(response.memories)} "
                f"degraded={'bm25_only' if degraded_bm25_only else 'no'} "
                f"query='{request.query[:80]}'"
            )

        except Exception as e:
            logger.error(f"TencentHybridReadPipeline.read failed: {e}", exc_info=True)
            response.error_code = 500
            response.error_message = str(e)
            tracer.set_error(str(e))

        response.elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        await trace_log.log_summary(
            query=request.query,
            intent=None,
            confidence=None,
            is_low_confidence=None,
            channels={
                "hybrid": len(hybrid_hits),
                "bm25": len(bm25_hits),
                "profile": len(profile_hits),
            },
            total_found=response.total_found,
            elapsed_ms=response.elapsed_ms,
            returned_memories=response.memories,
        )
        return response

    async def close(self) -> None:
        if self._vector_store and not self._external_vector_store:
            await self._vector_store.close()
