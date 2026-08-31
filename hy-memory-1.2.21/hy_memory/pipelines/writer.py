"""
HY Memory - Write (System 1)

写入流程:
  1. 参数校验
  2. 向量化 (EmbedService)
  3. Qdrant 持久化 (VectorStore) - 原始内容始终存为 L1_RAW
  4. [可选] MemAgent 智能处理 (提取实体、生成摘要、冲突检测)
     - Agent 提取的高层信息存为 L2_FACT/L3_SUMMARY（不再区分 L4_IDENTITY，统一 L2_FACT）
     - intentions（前瞻意图）存为 L7_INTENTION（带 valid_until，过期由 reader 惰性转 L2）
     - 提取成功后，L1_RAW 降级为 SHADOW 状态（不被召回）
  5. [可选] 合并检测 (Merger)

mode 行为差异:
  - lite:  只存 L1_RAW，不调 LLM，最简 embed 入库
  - pro:   L1_RAW + MemAgent 提取高层信息 → reconcile → L2/L4/L3
  - ultra: 同 pro，但 System2Writer 会在之后异步执行 System 2 认知加工
"""

import json
import math
import os
import re
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
import logging

from .base import WritePipeline, WriteRequest, WriteResponse, PipelineContext
from ..config import MemoryConfig
from ..core.scorer import MemoryScorer as Scorer
from ..core.merger import Merger
from ..core.embed_service import EmbedService
from ..agent.mem_agent import MemAgent, ProcessMode
from ..agent.reconciler import MemoryReconciler
from ..agent.tools.basic_profile import upsert_basic_profile
from ..models.memory import MemoryNode, MemoryLayer, MemoryStatus, SourceType
from ..data.vector_store import create_vector_store
from ..data.vector_store_base import VectorStoreBase
from ..utils.tracer import PipelineTracer, create_tracer
from ..utils.log_setup import get_request_id
from ..utils.pipeline_observability import is_pipeline_trace_enabled
from ._retrieval import tag_index as _tag_index_helper

logger = logging.getLogger(__name__)

_RECONCILE_ENABLED = os.getenv("RECONCILE_ENABLED", "true").lower() == "true"


def _norm_owner(value: Any) -> Optional[str]:
    """归一化 extractor/reconcile 给出的 owner：仅接受 'user' / 'agent'，否则 None。"""
    if not value:
        return None
    v = str(value).strip().lower()
    if v in ("user", "agent"):
        return v
    # 兼容 mem0 风格的 'assistant' → 'agent'
    if v == "assistant":
        return "agent"
    return None


_TEMPORAL_META_KEYS = (
    "observed_at",
    "temporal_relation",
    "event_time_text",
    "event_start",
    "event_end",
    "normalization_confidence",
    "valid_from",
    "valid_until",
)


def _parse_temporal_datetime(
    value: Any,
    *,
    end_of_day: bool = False,
    default_timezone: Any = None,
) -> Optional[datetime]:
    """Parse extractor/reconciler ISO values using the conversation timezone.

    LLMs commonly return date-only values such as ``2023-10-20``.  Leaving
    those datetimes naive makes ``timestamp()`` depend on the host timezone,
    so the same memory can become the previous UTC date on another machine.
    When the write request has a timezone-aware temporal anchor, inherit that
    timezone for all otherwise-naive temporal fields.
    """
    if value is None or str(value).strip().lower() in ("", "null", "none"):
        return None
    if isinstance(value, datetime):
        parsed = value
        date_only = False
    else:
        raw = str(value).strip()
        date_only = len(raw) == 10
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    if end_of_day and date_only:
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=0)
    if parsed.tzinfo is None and default_timezone is not None:
        parsed = parsed.replace(tzinfo=default_timezone)
    return parsed


def _temporal_kwargs(
    meta: Optional[Dict[str, Any]],
    *,
    observed_fallback: Optional[datetime],
    default_relation: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert temporal metadata into MemoryNode constructor/update values."""
    meta = meta or {}
    relation = meta.get("temporal_relation") or default_relation
    relation = str(relation).strip().lower() if relation else None
    if relation not in {None, "past", "present", "future", "recurring", "atemporal", "unknown"}:
        relation = "unknown"
    raw_conf = meta.get("normalization_confidence")
    confidence = None
    if raw_conf is not None and str(raw_conf).strip().lower() not in ("", "null", "none"):
        try:
            confidence = max(0.0, min(1.0, float(raw_conf)))
        except (TypeError, ValueError):
            confidence = None
    anchor_timezone = (
        observed_fallback.tzinfo
        if isinstance(observed_fallback, datetime) and observed_fallback.tzinfo is not None
        else None
    )
    observed_at = _parse_temporal_datetime(
        meta.get("observed_at"), default_timezone=anchor_timezone,
    ) or observed_fallback
    if isinstance(observed_at, datetime) and observed_at.tzinfo is not None:
        anchor_timezone = observed_at.tzinfo

    values = {
        "observed_at": observed_at,
        "temporal_relation": relation,
        "event_time_text": (str(meta.get("event_time_text")).strip() if meta.get("event_time_text") else None),
        "event_start": _parse_temporal_datetime(
            meta.get("event_start"), default_timezone=anchor_timezone,
        ),
        "event_end": _parse_temporal_datetime(
            meta.get("event_end"), end_of_day=True,
            default_timezone=anchor_timezone,
        ),
        "normalization_confidence": confidence,
        "valid_from": _parse_temporal_datetime(
            meta.get("valid_from"), default_timezone=anchor_timezone,
        ),
        "valid_until": _parse_temporal_datetime(
            meta.get("valid_until"), end_of_day=True,
            default_timezone=anchor_timezone,
        ),
    }
    if (
        values["valid_from"] is None
        and relation in {"past", "present", "future", "recurring"}
    ):
        values["valid_from"] = values["event_start"] or values["observed_at"]
    return values


def _copy_temporal_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    """Copy only the public L2/L7 temporal schema from an extracted item."""
    return {key: item.get(key) for key in _TEMPORAL_META_KEYS}


def _op_temporal_meta(op: Any) -> Dict[str, Any]:
    return {key: getattr(op, key, None) for key in _TEMPORAL_META_KEYS}


class MemoryWriter(WritePipeline):
    """
    核心写入器 (System 1)

    写入流程: 原始内容存 L1_RAW + 单路 Qdrant 存储 + 可选 MemAgent LLM 提取高层信息
    lite 模式不调 LLM，pro/ultra 模式调 MemAgent。
    """

    VERSION = "writer"

    # Calibrated against multilingual positive/negative citation pairs.  Keep
    # a margin above weak topical association: direct paraphrases in the probe
    # set scored 0.71-0.97 while an unrelated real-turn citation scored 0.40.
    _PROVENANCE_MIN_SCORE = 0.45

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        denominator = math.sqrt(
            sum(value * value for value in left)
            * sum(value * value for value in right)
        )
        if denominator <= 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / denominator

    def __init__(
        self,
        config: MemoryConfig,
        embed_service: Optional[EmbedService] = None,
        vector_store: Optional[VectorStoreBase] = None,
        graph_store: Optional[Any] = None,
        cache=None,
    ):
        self.config = config
        self._embed_service = embed_service
        self._external_vector_store = vector_store
        self._graph_store = graph_store
        self._cache = cache

        self._merger: Optional[Merger] = None
        self._mem_agent: Optional[MemAgent] = None
        self._reconciler: Optional[MemoryReconciler] = None
        self._vector_store: Optional[VectorStoreBase] = None
        self._vector_store_initialized = False

        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self._embed_service is None:
            self._embed_service = EmbedService(self.config)
        self._merger = Merger()
        self._mem_agent = MemAgent(self.config)
        self._reconciler = MemoryReconciler(self.config)
        self._vector_store = self._external_vector_store or create_vector_store(self.config)
        if self._external_vector_store and getattr(self._external_vector_store, '_client', None):
            self._vector_store_initialized = True
        self._initialized = True
        logger.debug("MemoryWriter initialized")

    @property
    def embed_service(self) -> EmbedService:
        if self._embed_service is None:
            self._embed_service = EmbedService(self.config)
        return self._embed_service

    @property
    def merger(self) -> Merger:
        if self._merger is None:
            self._merger = Merger()
        return self._merger

    @property
    def mem_agent(self) -> MemAgent:
        if self._mem_agent is None:
            self._mem_agent = MemAgent(self.config)
        return self._mem_agent

    @property
    def reconciler(self) -> MemoryReconciler:
        if self._reconciler is None:
            self._reconciler = MemoryReconciler(self.config)
        return self._reconciler

    async def _get_vector_store(self) -> VectorStoreBase:
        if self._vector_store is None:
            self._vector_store = self._external_vector_store or create_vector_store(self.config)
        if not self._vector_store_initialized:
            await self._vector_store.initialize()
            self._vector_store_initialized = True
        return self._vector_store

    # ================================================================
    # 私有辅助方法
    # ================================================================

    @staticmethod
    def _build_custom(op, request_id: str) -> Dict[str, Any]:
        """构造 VDB payload 中的 custom 字段。始终包含 request_id 以便追溯。"""
        custom: Dict[str, Any] = {}
        if request_id:
            custom["request_id"] = request_id
        if op.supersede_reason:
            custom["supersede_reason"] = op.supersede_reason
        return custom

    @staticmethod
    def _request_turn_records(request: WriteRequest) -> List[Dict[str, Any]]:
        """Return extractor input turns with stable, deterministic provenance IDs."""
        session_id = request.session_id or "default_session"
        start_index = request.turn_index if request.turn_index is not None else 0
        if request.messages:
            records: List[Dict[str, Any]] = []
            for offset, message in enumerate(request.messages):
                turn_index = start_index + offset
                turn_id = (message.turn_id or "").strip() or (
                    f"{session_id}:turn:{turn_index}"
                )
                records.append({
                    "role": message.role,
                    "content": message.content,
                    "turn_id": turn_id,
                    "turn_index": turn_index,
                })
            return records
        if request.content:
            turn_id = f"{session_id}:turn:{start_index}"
            return [{
                "role": request.role or "user",
                "content": request.content,
                "turn_id": turn_id,
                "turn_index": start_index,
            }]
        return []

    @classmethod
    def _sanitize_extracted_provenance(
        cls,
        extracted_info: Dict[str, Any],
        request: WriteRequest,
    ) -> None:
        """Keep only turn IDs actually present in this write request.

        The extractor is asked to cite exact IDs, but an LLM-produced identifier is
        never trusted directly.  A single-turn input can be filled deterministically;
        a multi-turn item with no citation remains uncited rather than claiming every
        turn as evidence and inflating retrieval recall.
        """
        if not isinstance(extracted_info, dict):
            return
        records = cls._request_turn_records(request)
        id_to_index = {
            str(record["turn_id"]): int(record["turn_index"])
            for record in records
        }
        index_to_id = {
            int(record["turn_index"]): str(record["turn_id"])
            for record in records
        }
        for section in ("memory", "facts", "identity", "intentions"):
            for item in extracted_info.get(section) or []:
                if not isinstance(item, dict):
                    continue
                raw_ids = item.get("source_turn_ids") or []
                if isinstance(raw_ids, (str, int)):
                    raw_ids = [raw_ids]
                valid_ids: List[str] = []
                for raw_id in raw_ids if isinstance(raw_ids, list) else []:
                    candidate = str(raw_id)
                    if candidate in id_to_index and candidate not in valid_ids:
                        valid_ids.append(candidate)
                if not valid_ids:
                    raw_indices = item.get("source_turn_indices") or []
                    if isinstance(raw_indices, (str, int)):
                        raw_indices = [raw_indices]
                    for raw_index in raw_indices if isinstance(raw_indices, list) else []:
                        try:
                            candidate_id = index_to_id[int(raw_index)]
                        except (KeyError, TypeError, ValueError):
                            continue
                        if candidate_id not in valid_ids:
                            valid_ids.append(candidate_id)
                if not valid_ids and len(records) == 1:
                    valid_ids = [str(records[0]["turn_id"])]
                item["source_turn_ids"] = valid_ids
                item["source_turn_indices"] = [id_to_index[value] for value in valid_ids]

    async def _validate_extracted_provenance(
        self,
        extracted_info: Dict[str, Any],
        request: WriteRequest,
    ) -> Dict[str, Any]:
        """Remove citations whose local source context is semantically unrelated.

        ID whitelisting prevents invented turn IDs, but it cannot prevent an LLM
        from attaching a real ID from an unrelated topic.  Validate every cited
        turn against the extracted statement using the configured multilingual
        embedder.  A one-turn context window on each side preserves short answers
        and temporal scope inherited from an adjacent question.

        Validation is deliberately fail-open on embedder errors: write
        availability must not depend on this quality guard, while successful
        validation is conservative and may leave an item uncited rather than
        claiming unsupported evidence.
        """
        summary: Dict[str, Any] = {
            "enabled": False,
            "checked": 0,
            "accepted": 0,
            "rejected": 0,
            "min_score": None,
        }
        if not isinstance(extracted_info, dict):
            return summary
        if os.getenv("MEMORY_PROVENANCE_VALIDATION_ENABLED", "true").lower() not in {
            "true", "1", "yes", "on",
        }:
            return summary

        try:
            min_score = float(os.getenv(
                "MEMORY_PROVENANCE_MIN_SCORE", str(self._PROVENANCE_MIN_SCORE),
            ))
        except ValueError:
            min_score = self._PROVENANCE_MIN_SCORE
        min_score = max(0.0, min(1.0, min_score))
        summary.update({"enabled": True, "min_score": min_score})

        records = self._request_turn_records(request)
        if not records:
            return summary
        id_to_record = {str(record["turn_id"]): record for record in records}
        id_to_context: Dict[str, str] = {}
        for index, record in enumerate(records):
            start = max(0, index - 1)
            end = min(len(records), index + 2)
            id_to_context[str(record["turn_id"])] = "\n".join(
                str(records[position].get("content") or "").strip()
                for position in range(start, end)
                if str(records[position].get("content") or "").strip()
            )

        items: List[Dict[str, Any]] = []
        texts: List[str] = []
        for section in ("memory", "facts", "identity", "intentions"):
            for item in extracted_info.get(section) or []:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content") or "").strip()
                cited_ids = [
                    str(turn_id) for turn_id in (item.get("source_turn_ids") or [])
                    if str(turn_id) in id_to_record
                ]
                if not content or not cited_ids:
                    continue
                items.append(item)
                texts.append(content)
                texts.extend(id_to_context[turn_id] for turn_id in cited_ids)

        if not items:
            return summary

        try:
            vectors = await self.embed_service.embed_batch(texts)
        except Exception as error:
            summary.update({"enabled": False, "error": str(error)})
            logger.warning(
                f"[provenance] semantic validation skipped after embed failure: {error}"
            )
            return summary
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            summary.update({"enabled": False, "error": "incomplete embedding batch"})
            logger.warning("[provenance] semantic validation skipped: incomplete embedding batch")
            return summary

        cursor = 0
        rejected_detail: List[Dict[str, Any]] = []
        for item in items:
            content_vector = vectors[cursor]
            cursor += 1
            original_ids = list(item.get("source_turn_ids") or [])
            accepted_ids: List[str] = []
            accepted_indices: List[int] = []
            for turn_id in original_ids:
                context_vector = vectors[cursor]
                cursor += 1
                score = self._cosine_similarity(content_vector, context_vector)
                summary["checked"] += 1
                if score >= min_score:
                    accepted_ids.append(str(turn_id))
                    accepted_indices.append(int(id_to_record[str(turn_id)]["turn_index"]))
                    summary["accepted"] += 1
                else:
                    summary["rejected"] += 1
                    rejected_detail.append({
                        "turn_id": str(turn_id),
                        "score": round(score, 4),
                        "memory": str(item.get("content") or "")[:120],
                    })
            item["source_turn_ids"] = accepted_ids
            item["source_turn_indices"] = accepted_indices

        if rejected_detail:
            logger.warning(
                "[provenance] rejected semantically unsupported citations: %s",
                json.dumps(rejected_detail, ensure_ascii=False),
            )
        return summary

    @classmethod
    def _sanitize_basic_info(
        cls,
        extracted_info: Dict[str, Any],
        request: WriteRequest,
    ) -> Dict[str, Any]:
        """Conservatively require an explicit current-residence statement.

        The default ``location`` profile field means current primary residence.
        Origin, hometown and travel statements remain useful L2 facts but must
        not overwrite L0.  Other fields stay governed by the extractor schema.
        """
        if not isinstance(extracted_info, dict):
            return {}
        raw = extracted_info.get("basic_info")
        if not isinstance(raw, dict):
            return {}
        sanitized = dict(raw)
        location = str(sanitized.get("location") or "").strip()
        if not location:
            return sanitized

        user_texts = [
            str(record.get("content") or "")
            for record in cls._request_turn_records(request)
            if str(record.get("role") or "").lower() == "user"
        ]
        value = re.escape(location)
        current_patterns = (
            rf"\b(?:currently\s+|now\s+)?(?:live|living|reside|residing|based|settled)\s+(?:in|at)\s+{value}\b",
            rf"\b(?:my\s+)?(?:current\s+)?(?:home|residence|address)\s+(?:is|is\s+in|in)\s+{value}\b",
            rf"\b(?:moved|relocated)\s+to\s+{value}\b",
            rf"(?:现居|现住|住在|居住在|定居于?|搬到|搬去了?)\s*{value}",
        )
        explicitly_current = any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for text in user_texts
            for pattern in current_patterns
        )
        if not explicitly_current:
            sanitized.pop("location", None)
            logger.warning(
                "[basic-profile] dropped location without explicit current-residence support: %r",
                location,
            )
        return sanitized

    @staticmethod
    def _op_provenance(
        op: Any,
        new_memories_meta: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[int]]:
        """Resolve reconciler memory-list indices to original conversation turns."""
        turn_ids: List[str] = []
        turn_indices: List[int] = []
        for raw_index in getattr(op, "source_indices", None) or []:
            try:
                meta = new_memories_meta[int(raw_index)]
            except (IndexError, TypeError, ValueError):
                continue
            for turn_id in meta.get("source_turn_ids") or []:
                value = str(turn_id)
                if value and value not in turn_ids:
                    turn_ids.append(value)
            for turn_index in meta.get("source_turn_indices") or []:
                try:
                    value = int(turn_index)
                except (TypeError, ValueError):
                    continue
                if value not in turn_indices:
                    turn_indices.append(value)
        return turn_ids, turn_indices

    @staticmethod
    def _merge_evidence(*chains: Any) -> List[str]:
        merged: List[str] = []
        for chain in chains:
            for value in chain or []:
                value = str(value)
                if value and value not in merged:
                    merged.append(value)
        return merged

    async def _maybe_index_entities(
        self, vector_store, node, request,
    ) -> None:
        """若 entity_store 开关开启且为 L2_FACT，落库后刷 entity store（best-effort）。

        覆盖 reconcile（ADD/UPDATE/SUPERSEDE）与首写（_direct_store）两条路。
        """
        try:
            if not getattr(self.config.recall, "entity_store_enabled", False):
                return
            if node is None or node.layer != MemoryLayer.L2_FACT:
                return
            from ._retrieval.entity_store import index_memory_entities
            await index_memory_entities(
                vector_store=vector_store,
                embed_service=self.embed_service,
                memory_id=node.node_id,
                content=node.content or "",
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
            )
        except Exception as e:
            logger.debug(f"[entity-index] skipped (non-fatal): {e}")

    @staticmethod
    def _collect_new_memories(
        extracted_info: Dict[str, Any],
    ) -> Tuple[List[str], List[Dict]]:
        """
        从 extract 结果中收集新 memory 文本列表和完整 meta 列表。

        注意：basic_info 字段由 extractor 在 JSON 输出中返回，writer 单独处理
        （走 upsert_basic_profile() 落 L0_BASIC_INFO 演化链），这里不再收集。

        Returns:
            (new_memory_texts, new_memories_meta)
            每条 meta: {"content", "layer", "tags"}
        """
        new_memory_texts: List[str] = []
        new_memories_meta: List[Dict] = []

        # 1) memory → 每条独立 memory（统一 L2_FACT）
        #    新版 extractor 输出 `memory`；兼容旧版 `facts` 字段。
        for item in (extracted_info.get("memory") or extracted_info.get("facts") or []):
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            if not content:
                continue
            new_memory_texts.append(content)
            new_memories_meta.append({
                "content": content,
                "layer": "L2_FACT",
                "tags": item.get("tags") or [],
                "owner": _norm_owner(item.get("owner")),
                "relations": list(item.get("relations") or []),
                "source_turn_ids": list(item.get("source_turn_ids") or []),
                "source_turn_indices": list(item.get("source_turn_indices") or []),
                **_copy_temporal_meta(item),
            })

        # 2) 向后兼容：旧 extractor 输出的 identity 也并入 L2_FACT
        #    （新版 extractor 不再产出 identity；L4_IDENTITY 不再写入，仅读历史数据）
        for item in (extracted_info.get("identity") or []):
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            if not content:
                continue
            new_memory_texts.append(content)
            new_memories_meta.append({
                "content": content,
                "layer": "L2_FACT",
                "tags": item.get("tags") or [],
                "owner": _norm_owner(item.get("owner")),
                "relations": list(item.get("relations") or []),
                "source_turn_ids": list(item.get("source_turn_ids") or []),
                "source_turn_indices": list(item.get("source_turn_indices") or []),
                **_copy_temporal_meta(item),
            })

        # 3) 兼容旧版 extract 输出（profile + facts）— 不处理 basic_info
        if not new_memory_texts:
            profile = extracted_info.get("profile", {})
            if profile and isinstance(profile, dict):
                profile_items = []
                for k, v in profile.items():
                    if k == "preferences":
                        continue
                    if v and str(v).lower() not in ("null", "none", ""):
                        profile_items.append(f"{k}: {v}")
                prefs = profile.get("preferences", [])
                if prefs and isinstance(prefs, list):
                    pref_strs = [str(p) for p in prefs if p]
                    if pref_strs:
                        profile_items.append(f"preferences: {', '.join(pref_strs)}")
                if profile_items:
                    t = "; ".join(profile_items)
                    new_memory_texts.append(t)
                    new_memories_meta.append({"content": t, "layer": "L2_FACT", "tags": []})
            for f in (extracted_info.get("facts") or []):
                if isinstance(f, dict) and f.get("content"):
                    new_memory_texts.append(f["content"])
                    new_memories_meta.append({"content": f["content"], "layer": "L2_FACT", "tags": f.get("tags") or []})

        return new_memory_texts, new_memories_meta

    async def _dedup_extracted(
        self,
        new_memory_texts: List[str],
        new_memories_meta: List[Dict],
        request: "WriteRequest",
        req_id: str,
    ) -> Tuple[List[str], List[Dict]]:
        """对 extractor 新抽取的多条 memory 互相去重（入库前）。

        这些条目还没落库、无 node_id、无演化链 → 全按非链处理，额外 embed 一次，
        delete_from_store=False（只丢弃重复项，不删库），并记 DEDUP log。
        保留：用列表下标做确定性 gmt（越靠前越优先保留，等价 extractor 输出顺序）。
        """
        from ..pipelines._retrieval.dedup import DedupItem, execute_dedup

        embeds = await self.embed_service.embed_batch(list(new_memory_texts))
        items: List[DedupItem] = []
        for i, (text, emb) in enumerate(zip(new_memory_texts, embeds)):
            if not emb:
                continue
            items.append(DedupItem(
                node_id=str(i),               # 用下标作临时 id
                embedding=emb,
                content=text,
                is_latest=True,
                is_chain_head=False,
                gmt_created=float(i),         # 越靠前 gmt 越小 → 优先保留
                chain_node_ids=[str(i)],
            ))
        if len(items) < 2:
            return new_memory_texts, new_memories_meta

        plan = await execute_dedup(
            items,
            vector_store=None,                # 还没入库
            cache=self._cache,
            trigger="extractor",
            request_id=req_id,
            user_id=request.user_id,
            agent_id=request.agent_id or "default_agent",
            delete_from_store=False,
        )
        drop_idx = {int(d) for d in plan.get("delete_ids", [])}
        if not drop_idx:
            return new_memory_texts, new_memories_meta

        kept_texts, kept_meta = [], []
        for i, (t, m) in enumerate(zip(new_memory_texts, new_memories_meta)):
            if i in drop_idx:
                continue
            kept_texts.append(t)
            kept_meta.append(m)
        logger.info(
            f"[write] extractor dedup dropped {len(drop_idx)} of {len(new_memory_texts)} extracted"
        )
        return kept_texts, kept_meta

    @staticmethod
    def _collect_intentions(extracted_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从 extract 结果中收集 intentions（前瞻意图，存 L7_INTENTION）。

        每条 intention: {"content", "tags", "valid_until"}
        valid_until 是 extractor 输出的 ISO 日期串（或 null）；这里解析为
        datetime（当天 23:59:59，宽松到日末），解析失败/缺失 → None。
        """
        out: List[Dict[str, Any]] = []
        for item in (extracted_info.get("intentions") or []):
            if not isinstance(item, dict):
                continue
            content = (item.get("content") or "").strip()
            if not content:
                continue
            out.append({
                "content": content,
                "tags": item.get("tags") or [],
                "owner": _norm_owner(item.get("owner")),
                "source_turn_ids": list(item.get("source_turn_ids") or []),
                "source_turn_indices": list(item.get("source_turn_indices") or []),
                **_copy_temporal_meta(item),
            })
        return out

    async def _store_intentions(
        self,
        intentions: List[Dict[str, Any]],
        request: WriteRequest,
        vector_store: VectorStoreBase,
        req_id: str,
    ) -> List[str]:
        """
        把 intentions 直接 upsert 为 L7_INTENTION 节点（不走 reconcile）。

        意图是 point-in-time 信号，不与已有 fact 合并；过期后由 reader 惰性
        转成 L2_FACT。返回新建节点 id 列表。
        """
        if not intentions:
            return []

        stored: List[str] = []
        # 批量 embed
        contents = [it["content"] for it in intentions]
        embeddings: List[Optional[List[float]]] = [None] * len(contents)
        try:
            batch = await self.embed_service.embed_batch(contents)
            for i, emb in enumerate(batch):
                embeddings[i] = emb
        except Exception as e:
            logger.warning(f"[intention] batch embed failed, falling back to sequential: {e}")

        for i, it in enumerate(intentions):
            emb = embeddings[i]
            if emb is None:
                try:
                    emb = await self.embed_service.embed_queued(it["content"])
                except Exception as e:
                    logger.error(f"[intention] embed failed, skip: {e}")
                    continue
            node = MemoryNode(
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "default_session",
                layer=MemoryLayer.L7_INTENTION,
                content=it["content"],
                owner=it.get("owner") or "user",
                supersedes=None,
                is_latest=True,
                source_type=SourceType.INFERRED,
                status=MemoryStatus.ACTIVE,
                embedding=emb,
                memory_at=request.memory_at,
                tags=list(it.get("tags") or []),
                source_session_id=request.session_id or "default_session",
                source_turn_index=(it.get("source_turn_indices") or [None])[0],
                evidence_chain=list(it.get("source_turn_ids") or []),
                custom={"request_id": req_id} if req_id else {},
                **_temporal_kwargs(
                    it,
                    observed_fallback=request.memory_at,
                    default_relation="future",
                ),
            )
            try:
                nid = await vector_store.upsert(node)
                stored.append(nid)
                logger.debug(
                    f"[intention] L7 upsert: {it['content'][:80]} → node_id={nid} "
                    f"valid_until={it.get('valid_until')}"
                )
            except Exception as e:
                logger.error(f"[intention] upsert failed: {e}")
        return stored

    async def _reconcile_and_store(
        self,
        new_memory_texts: List[str],
        new_memories_meta: List[Dict],
        request: WriteRequest,
        vector_store: VectorStoreBase,
        req_id: str,
    ) -> Tuple[List[str], Optional[str], Dict[str, int], Dict[str, int]]:
        """
        对新 memories 做 reconcile，执行 ADD（含 EVOLVE），写 DIGEST_SUMMARY log。

        Returns:
            (stored_ids, error_message, ops_counts, recon_tokens)
            error_message=None 表示成功；ops_counts 含 add/supersede/update/total；
            recon_tokens 含 prompt/completion/total（reconcile 链路 LLM 消耗）
        """
        stored_ids: List[str] = []
        current_time = request.memory_at.isoformat(timespec="seconds") if request.memory_at else ""

        recon_result = await self.reconciler.reconcile(
            new_memories=new_memory_texts,
            user_id=request.user_id,
            agent_id=request.agent_id or "default_agent",
            vector_store=vector_store,
            embed_service=self.embed_service,
            layers=[MemoryLayer.L2_FACT, MemoryLayer.L4_IDENTITY],
            cache=self._cache,
            request_id=req_id,
            current_time=current_time,
            new_memories_with_meta=new_memories_meta,
            session_id=request.session_id or "",
        )

        if not recon_result.success:
            return [], recon_result.error, {"add": 0, "supersede": 0, "update": 0, "total": 0}, {"prompt": 0, "completion": 0, "total": 0}

        # 分类统计
        _op_counts = {}
        for op in recon_result.ops:
            _op_counts[op.op] = _op_counts.get(op.op, 0) + 1
        _op_counts_str = " ".join(f"{k}={v}" for k, v in sorted(_op_counts.items()))

        logger.info(
            f"TRACE_PERF [{req_id}] S1_RECONCILE_DONE "
            f"ops={len(recon_result.ops)} candidates={len(new_memory_texts)} | {_op_counts_str}"
        )
        # 逐条 op 详情由 reconciler 的 "[reconciler] ops detail" 单条 JSON list 输出，
        # 这里不再逐条打印（避免重复刷屏）。

        # ── 批量 embed：收集所有需要 embed 的 content，一次性 batch 调用 ──
        contents_to_embed: List[str] = []
        content_indices: List[int] = []  # 对应 ops 索引
        for i, op in enumerate(recon_result.ops):
            if op.op in ("SUPERSEDE", "UPDATE"):
                if op.op == "UPDATE" and not op.content:
                    continue  # shadow-only, no embed needed
                if not op.memory_id:
                    continue
                contents_to_embed.append(op.content or "")
                content_indices.append(i)
            elif op.op == "ADD":
                contents_to_embed.append(op.content or "")
                content_indices.append(i)

        # 一次 batch embed（不逐个串行）
        embeddings_map: Dict[int, List[float]] = {}
        if contents_to_embed:
            try:
                batch_embeddings = await self.embed_service.embed_batch(contents_to_embed)
                for idx, emb in zip(content_indices, batch_embeddings):
                    embeddings_map[idx] = emb
            except Exception as e:
                logger.warning(f"[reconciler] batch embed failed, falling back to sequential: {e}")
                # fallback: 逐个 embed
                for idx, content in zip(content_indices, contents_to_embed):
                    try:
                        embeddings_map[idx] = await self.embed_service.embed_queued(content)
                    except Exception as e2:
                        logger.error(f"[reconciler] embed failed for op {idx}: {e2}")

        # 统计。apply_errors 用于把局部写入失败传播给上层，
        # 避免 DIGEST_SUMMARY 在部分 op 未落库时仍显示全部成功。
        add_cnt = 0           # ADD（全新）
        supersede_cnt = 0     # SUPERSEDE（矛盾演化）
        update_cnt = 0        # UPDATE（合并精炼）
        apply_errors: List[str] = []
        graph_counts = {"published": 0, "rejected": 0, "failures": 0}

        async def _publish_graph(
            fact_node: MemoryNode,
            op: Any,
            supersedes_fact_ids: Optional[List[str]] = None,
        ) -> None:
            """把本 op 对齐到的显式候选关系发布到可信 L2 图。"""
            graph_config = getattr(self.config, "graph_store", None)
            if not getattr(graph_config, "graphrag_enabled", False):
                return
            from ._retrieval.graphrag import (
                collect_relations_for_op,
                publish_l2_fact_relations,
            )

            relations = collect_relations_for_op(
                op,
                new_memories_meta,
                min_confidence=float(
                    getattr(graph_config, "graphrag_min_confidence", 0.8)
                ),
            )
            if not relations and not supersedes_fact_ids:
                return
            try:
                summary = await publish_l2_fact_relations(
                    graph_store=self._graph_store,
                    fact_node=fact_node,
                    relations=relations,
                    supersedes_fact_ids=supersedes_fact_ids or [],
                )
            except Exception as graph_error:
                graph_counts["failures"] += 1
                logger.warning(
                    "[graphrag-write] unexpected publish error for fact=%s: %s",
                    fact_node.node_id,
                    graph_error,
                    exc_info=True,
                )
                return
            graph_counts["published"] += int(summary.get("published") or 0)
            graph_counts["rejected"] += int(summary.get("rejected") or 0)
            if not summary.get("success"):
                graph_counts["failures"] += 1
                logger.warning(
                    "[graphrag-write] publish failed for fact=%s: %s",
                    fact_node.node_id,
                    summary.get("errors") or summary.get("error") or "unknown",
                )

        for op_idx, op in enumerate(recon_result.ops):
            op_turn_ids, op_turn_indices = self._op_provenance(op, new_memories_meta)
            # ------------------------------------------------
            # SUPERSEDE / UPDATE：标记旧节点 → 创建新节点
            # SUPERSEDE: 旧节点 status=SUPERSEDED（进演化链，仍可召回+展开），
            #            新节点 supersedes=[old_id]
            # UPDATE: 原地合并兼容信息，保留 node_id 和首次观测时间，
            #         不进化；明确状态转变会在 reconciler 边界升级为 SUPERSEDE。
            # ------------------------------------------------
            if op.op in ("SUPERSEDE", "UPDATE"):
                target_id = op.memory_id
                if not target_id:
                    logger.warning(f"[reconciler] {op.op} op missing memory_id, skipped")
                    apply_errors.append(f"{op.op}: missing memory_id")
                    continue

                # UPDATE with no content = legacy DELETE mapping (shadow-only)
                if op.op == "UPDATE" and not op.content:
                    try:
                        updated = await vector_store.update_payload(
                            target_id,
                            {
                                "is_latest": False,
                                "status": MemoryStatus.SHADOW.value,
                            },
                        )
                        if not updated:
                            raise RuntimeError("vector store returned false")
                        logger.debug(f"[reconciler] shadow-only UPDATE: memory_id={target_id}")
                    except Exception as e:
                        logger.warning(f"[reconciler] failed to shadow node {target_id}: {e}")
                        apply_errors.append(f"UPDATE {target_id}: {type(e).__name__}: {e}")
                    continue

                content = op.content or ""
                layer = MemoryLayer.from_string(op.layer) if op.layer else MemoryLayer.L2_FACT

                # SUPERSEDE → 旧节点进演化链，status=SUPERSEDED（仍可被召回，
                #   命中后双向展开整链）；新节点 supersedes=[old_id]。
                # UPDATE → 旧节点不进链，status=SHADOW（等价逻辑删除，不召回）；
                #   新节点 supersedes=None，独立。
                is_supersede = op.op == "SUPERSEDE"
                old_status = (
                    MemoryStatus.SUPERSEDED.value if is_supersede
                    else MemoryStatus.SHADOW.value
                )

                # ------------------------------------------------
                # 多节点链折叠（仅 SUPERSEDE）：
                # op.memory_ids = [E0, E1, ...]（有序，旧→新）。把这些原本
                # 未成链的旧节点先连成一条链（E0 ← E1 ← ...），再让新 head
                # 节点 supersede 最新的那个（target_id = memory_ids[-1]）。
                # 每条旧节点都被标 SUPERSEDED（仍可召回+展开）；只有 head（新节点）
                # 保持 is_latest=True。
                # ------------------------------------------------
                chain_ids = list(getattr(op, "memory_ids", None) or [])
                if is_supersede and len(chain_ids) > 1:
                    for _prev_id, _cur_id in zip(chain_ids, chain_ids[1:]):
                        # _cur_id supersedes _prev_id：建立 supersedes / superseded_by 双向链
                        try:
                            _cur_node = await vector_store.get_by_id(_cur_id)
                            if _cur_node is not None:
                                _sup = list(_cur_node.supersedes or [])
                                if _prev_id not in _sup:
                                    _sup.append(_prev_id)
                                await vector_store.update_payload(
                                    _cur_id, {"supersedes": _sup}
                                )
                        except Exception as e:
                            logger.warning(
                                f"[reconciler] chain-link supersedes on {_cur_id} failed: {e}"
                            )
                        try:
                            _prev_node = await vector_store.get_by_id(_prev_id)
                            _sb = list(_prev_node.superseded_by or []) if _prev_node else []
                            if _cur_id not in _sb:
                                _sb.append(_cur_id)
                            await vector_store.update_payload(
                                _prev_id,
                                {
                                    "superseded_by": _sb,
                                    "is_latest": False,
                                    "status": MemoryStatus.SUPERSEDED.value,
                                },
                            )
                        except Exception as e:
                            logger.warning(
                                f"[reconciler] chain-link superseded_by on {_prev_id} failed: {e}"
                            )
                    logger.debug(
                        f"[reconciler] SUPERSEDE chain fold: {chain_ids} "
                        f"(head target={target_id})"
                    )

                # 标记链上最新的 target_id（新 head 直接取代它）：
                # SUPERSEDE → SUPERSEDED（进链，可召回）；UPDATE → SHADOW（逻辑删除）
                # ------------------------------------------------
                # UPDATE（有 content，非 SUPERSEDE）：原地更新旧节点
                # ------------------------------------------------
                # UPDATE 语义是「合并精炼同一主题」，本质是同一条记忆的延续，因此
                # 原地更新 target_id：换 content + embedding + tags + memory_at，
                # 天然保留 access_count / last_accessed_at（不打回冷启动），
                # 也不产生 SHADOW 垃圾节点。SUPERSEDE 仍走下面的 shadow+建新链。
                if not is_supersede:
                    new_emb = embeddings_map.get(op_idx) or await self.embed_service.embed_queued(content)
                    old_node = await vector_store.get_by_id(target_id)
                    if old_node is None:
                        logger.warning(f"[reconciler] UPDATE target not found: {target_id}")
                        apply_errors.append(f"UPDATE {target_id}: target not found")
                        continue
                    temporal = _temporal_kwargs(
                        _op_temporal_meta(op),
                        observed_fallback=request.memory_at,
                    )
                    # UPDATE is compatible enrichment, not a state transition.
                    # Preserve the first observation and the original validity
                    # start; keep the latest observation in custom metadata.
                    first_observed = old_node.observed_at or temporal["observed_at"]
                    old_valid_from = old_node.valid_from
                    new_valid_from = temporal["valid_from"]
                    if old_valid_from and new_valid_from:
                        merged_valid_from = min(
                            (old_valid_from, new_valid_from),
                            key=lambda value: value.timestamp(),
                        )
                    else:
                        merged_valid_from = old_valid_from or new_valid_from
                    custom = dict(old_node.custom or {})
                    merged_evidence = self._merge_evidence(
                        old_node.evidence_chain, op_turn_ids,
                    )
                    observations = list(custom.get("observation_times") or [])
                    latest_observed = temporal["observed_at"] or request.memory_at
                    if latest_observed:
                        observed_iso = latest_observed.isoformat()
                        if observed_iso not in observations:
                            observations.append(observed_iso)
                    if observations:
                        custom["observation_times"] = observations
                        custom["last_observed_at"] = observations[-1]
                    temporal_updates = {
                        "observed_at": int(first_observed.timestamp()) if first_observed else None,
                        "temporal_relation": temporal["temporal_relation"] or old_node.temporal_relation,
                        "event_time_text": temporal["event_time_text"] or old_node.event_time_text,
                        "event_start": int((temporal["event_start"] or old_node.event_start).timestamp())
                        if (temporal["event_start"] or old_node.event_start) else None,
                        "event_end": int((temporal["event_end"] or old_node.event_end).timestamp())
                        if (temporal["event_end"] or old_node.event_end) else None,
                        "normalization_confidence": (
                            temporal["normalization_confidence"]
                            if temporal["normalization_confidence"] is not None
                            else old_node.normalization_confidence
                        ),
                        "valid_from": int(merged_valid_from.timestamp()) if merged_valid_from else None,
                        "valid_until": int((temporal["valid_until"] or old_node.valid_until).timestamp())
                        if (temporal["valid_until"] or old_node.valid_until) else None,
                    }
                    try:
                        updated = await vector_store.update_payload(
                            target_id,
                            {
                                "content": content,
                                "embedding": new_emb,
                                "tags": list(op.tags or []),
                                "layer": layer.value,
                                # memory_at remains the original compatibility
                                # anchor; observed_at/custom track observations.
                                "memory_at": (int(old_node.memory_at.timestamp())
                                              if old_node.memory_at else None),
                                "gmt_modified": int(datetime.now().timestamp()),
                                "is_latest": True,
                                "status": MemoryStatus.ACTIVE.value,
                                "custom": custom,
                                "source_session_id": (
                                    old_node.source_session_id
                                    or request.session_id
                                    or "default_session"
                                ),
                                "source_turn_index": (
                                    old_node.source_turn_index
                                    if old_node.source_turn_index is not None
                                    else (op_turn_indices[0] if op_turn_indices else None)
                                ),
                                "evidence_chain": merged_evidence,
                                **temporal_updates,
                            },
                        )
                        if not updated:
                            raise RuntimeError("vector store returned false")
                    except Exception as e:
                        logger.warning(f"[reconciler] in-place UPDATE failed on {target_id}: {e}")
                        apply_errors.append(f"UPDATE {target_id}: {type(e).__name__}: {e}")
                        continue
                    stored_ids.append(target_id)
                    update_cnt += 1
                    updated_graph_node = await vector_store.get_by_id(target_id)
                    if updated_graph_node is not None:
                        await _publish_graph(updated_graph_node, op)
                    await self._emit_reconcile_apply(
                        request, vector_store=vector_store, op_type="UPDATE",
                        memory_id=target_id, layer=layer.value, content=content,
                    )
                    logger.debug(
                        f"[reconciler] UPDATE(in-place): {content[:80]} → node_id={target_id} "
                        f"layer={layer.value} reason={op.reason}"
                    )
                    # tag_index 惰性维护
                    if op.tags:
                        try:
                            await _tag_index_helper.ensure_tag_embeddings_for_node(
                                vector_store=vector_store,
                                embed_service=self.embed_service,
                                user_id=request.user_id,
                                tags=list(op.tags),
                            )
                        except Exception as e:
                            logger.debug(f"[tag-index] maintain failed (non-fatal): {e}")
                    # 写 memory_operations 记录
                    if self._cache:
                        try:
                            await self._cache.store_memory_operation(
                                request_id=req_id,
                                user_id=request.user_id,
                                agent_id=request.agent_id or "default_agent",
                                op=op.op,
                                memory_id=target_id,
                                content=content,
                                layer=layer.value,
                                reason=op.reason,
                            )
                        except Exception as e:
                            logger.warning(f"[reconciler] store_memory_operation(UPDATE) failed: {e}")
                    continue

                # ------------------------------------------------
                # SUPERSEDE：shadow（SUPERSEDED）旧节点 + 建新链头节点
                # ------------------------------------------------
                old_node = await vector_store.get_by_id(target_id)
                if old_node is None:
                    logger.warning(f"[reconciler] SUPERSEDE target not found: {target_id}")
                    apply_errors.append(f"SUPERSEDE {target_id}: target not found")
                    continue
                temporal = _temporal_kwargs(
                    _op_temporal_meta(op),
                    observed_fallback=request.memory_at,
                )
                transition_at = (
                    temporal["valid_from"]
                    or temporal["event_start"]
                    or temporal["observed_at"]
                    or request.memory_at
                )
                old_updates: Dict[str, Any] = {
                    "is_latest": False,
                    "status": old_status,
                }
                if transition_at and (
                    old_node.valid_until is None
                    or old_node.valid_until.timestamp() > transition_at.timestamp()
                ):
                    old_updates["valid_until"] = int(transition_at.timestamp())

                # 先创建并核验新 head，再一次性闭合旧节点。若旧节点
                # 更新失败，删除新 head 作为补偿，避免产生两个 latest。
                new_node = MemoryNode(
                    user_id=request.user_id,
                    agent_id=request.agent_id or "default_agent",
                    session_id=request.session_id or "default_session",
                    layer=layer,
                    content=content,
                    owner=getattr(op, "owner", None),
                    speculate=op.speculate,
                    supersedes=[target_id],
                    is_latest=True,
                    source_type=SourceType.INFERRED,
                    status=MemoryStatus.ACTIVE,
                    embedding=embeddings_map.get(op_idx) or await self.embed_service.embed_queued(content),
                    memory_at=request.memory_at,
                    tags=list(op.tags or []),
                    source_session_id=request.session_id or "default_session",
                    source_turn_index=(op_turn_indices[0] if op_turn_indices else None),
                    evidence_chain=self._merge_evidence(
                        old_node.evidence_chain, op_turn_ids,
                    ),
                    custom=self._build_custom(op, req_id),
                    **{
                        **temporal,
                        "valid_from": temporal["valid_from"] or transition_at,
                    },
                )
                nid = ""
                try:
                    nid = await vector_store.upsert(new_node)
                    persisted_new = await vector_store.get_by_id(nid)
                    if persisted_new is None:
                        raise RuntimeError("new head postcondition failed")

                    existing_superseded_by = list(old_node.superseded_by or [])
                    if nid not in existing_superseded_by:
                        existing_superseded_by.append(nid)
                    old_updates["superseded_by"] = existing_superseded_by

                    marked = await vector_store.update_payload(target_id, old_updates)
                    if not marked:
                        raise RuntimeError("old-node update returned false")
                    verified_old = await vector_store.get_by_id(target_id)
                    if (
                        verified_old is None
                        or verified_old.is_latest
                        or verified_old.status != MemoryStatus.SUPERSEDED
                        or nid not in (verified_old.superseded_by or [])
                    ):
                        raise RuntimeError("old-node evolution postcondition failed")
                    if transition_at and (
                        verified_old.valid_until is None
                        or int(verified_old.valid_until.timestamp())
                        != int(transition_at.timestamp())
                    ):
                        raise RuntimeError("old-node valid_until postcondition failed")
                except Exception as e:
                    cleanup_error = ""
                    try:
                        restored_id = await vector_store.upsert(old_node)
                        if restored_id != old_node.node_id:
                            cleanup_error += "; old-node rollback returned unexpected id"
                    except Exception as rollback_exc:
                        cleanup_error += (
                            f"; old-node rollback failed: "
                            f"{type(rollback_exc).__name__}: {rollback_exc}"
                        )
                    try:
                        if nid:
                            deleted = await vector_store.delete(nid)
                            if not deleted:
                                cleanup_error += "; compensation delete returned false"
                    except Exception as cleanup_exc:
                        cleanup_error += (
                            f"; compensation delete failed: "
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                        )
                    error_text = (
                        f"SUPERSEDE {target_id}: {type(e).__name__}: {e}{cleanup_error}"
                    )
                    logger.error(f"[reconciler] {error_text}")
                    apply_errors.append(error_text)
                    continue

                stored_ids.append(nid)
                await self._emit_reconcile_apply(
                    request, vector_store=vector_store, op_type="SUPERSEDE",
                    memory_id=nid, layer=layer.value, content=content,
                    supersedes=[target_id],
                )
                await self._maybe_index_entities(vector_store, new_node, request)
                await _publish_graph(
                    new_node,
                    op,
                    supersedes_fact_ids=[target_id],
                )

                supersede_cnt += 1

                logger.debug(
                    f"[reconciler] SUPERSEDE: {content[:80]} → node_id={nid} "
                    f"layer={layer.value} supersedes=[{target_id}] reason={op.reason}"
                )

                # tag_index 惰性维护
                if new_node.tags:
                    try:
                        await _tag_index_helper.ensure_tag_embeddings_for_node(
                            vector_store=vector_store,
                            embed_service=self.embed_service,
                            user_id=new_node.user_id,
                            tags=list(new_node.tags),
                        )
                    except Exception as e:
                        logger.debug(f"[tag-index] maintain failed (non-fatal): {e}")

                # 写 memory_operations 记录
                if self._cache:
                    try:
                        await self._cache.store_memory_operation(
                            request_id=req_id,
                            user_id=request.user_id,
                            agent_id=request.agent_id or "default_agent",
                            op=op.op,
                            memory_id=nid,
                            content=content,
                            layer=layer.value,
                            reason=op.supersede_reason or op.reason,
                            supersedes=[target_id] if is_supersede else [],
                        )
                    except Exception as e:
                        logger.warning(f"[reconciler] store_memory_operation({op.op}) failed: {e}")
                continue

            # ------------------------------------------------
            # ADD op：纯新增节点
            # ------------------------------------------------
            if op.op != "ADD":
                logger.warning(f"[reconciler] unknown op '{op.op}', skipped")
                continue

            content = op.content or ""
            layer = MemoryLayer.from_string(op.layer) if op.layer else MemoryLayer.L2_FACT

            new_node = MemoryNode(
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "default_session",
                layer=layer,
                content=content,
                owner=getattr(op, "owner", None),
                speculate=op.speculate,
                supersedes=None,
                is_latest=True,
                source_type=SourceType.INFERRED,
                status=MemoryStatus.ACTIVE,
                embedding=embeddings_map.get(op_idx) or await self.embed_service.embed_queued(content),
                memory_at=request.memory_at,
                tags=list(op.tags or []),
                source_session_id=request.session_id or "default_session",
                source_turn_index=(op_turn_indices[0] if op_turn_indices else None),
                evidence_chain=list(op_turn_ids),
                custom=self._build_custom(op, req_id),
                **_temporal_kwargs(
                    _op_temporal_meta(op),
                    observed_fallback=request.memory_at,
                ),
            )
            nid = await vector_store.upsert(new_node)
            stored_ids.append(nid)
            add_cnt += 1
            await self._emit_reconcile_apply(
                request, vector_store=vector_store, op_type="ADD",
                memory_id=nid, layer=layer.value, content=content,
            )
            await self._maybe_index_entities(vector_store, new_node, request)
            await _publish_graph(new_node, op)
            logger.debug(
                f"[reconciler] ADD: {content[:80]} → node_id={nid} "
                f"layer={layer.value} reason={op.reason}"
            )

            # tag_index 惰性维护（reader_hybrid_tag 路 B 依赖；失败静默降级）
            if new_node.tags:
                try:
                    await _tag_index_helper.ensure_tag_embeddings_for_node(
                        vector_store=vector_store,
                        embed_service=self.embed_service,
                        user_id=new_node.user_id,
                        tags=list(new_node.tags),
                    )
                except Exception as e:
                    logger.debug(f"[tag-index] maintain failed (non-fatal): {e}")

            if self._cache:
                try:
                    await self._cache.store_memory_operation(
                        request_id=req_id,
                        user_id=request.user_id,
                        agent_id=request.agent_id or "default_agent",
                        op="ADD",
                        memory_id=nid,
                        content=content,
                        layer=layer.value,
                        reason=op.reason,
                        supersedes=[],
                    )
                except Exception as e:
                    logger.warning(f"[reconciler] store_memory_operation failed: {e}")

        # DIGEST_SUMMARY log
        if self._cache and recon_result.ops:
            try:
                import json as _json2
                total_ops = len(recon_result.ops)
                summary_data = {
                    "success": not apply_errors,
                    "add_count": add_cnt,
                    "supersede_count": supersede_cnt,
                    "update_count": update_cnt,
                    "applied_ops": add_cnt + supersede_cnt + update_cnt,
                    "requested_ops": total_ops,
                    "failed_ops": len(apply_errors),
                    "errors": apply_errors,
                    "new_memories_input": len(new_memory_texts),
                    "graph_published": graph_counts["published"],
                    "graph_rejected": graph_counts["rejected"],
                    "graph_failures": graph_counts["failures"],
                }
                await self._cache.store_pipeline_log(
                    request_id=req_id,
                    user_id=request.user_id,
                    agent_id=request.agent_id or "default_agent",
                    session_id=request.session_id or "",
                    step="DIGEST_SUMMARY",
                    prompt="",
                    response=_json2.dumps(summary_data, ensure_ascii=False),
                    parsed=_json2.dumps(summary_data, ensure_ascii=False),
                    memory_ids=stored_ids,
                )
            except Exception as e:
                logger.warning(f"[write] store DIGEST_SUMMARY failed: {e}")

        ops_counts = {
            "add": add_cnt,
            "supersede": supersede_cnt,
            "update": update_cnt,
            "total": add_cnt + supersede_cnt + update_cnt,
            "graph_published": graph_counts["published"],
            "graph_rejected": graph_counts["rejected"],
            "graph_failures": graph_counts["failures"],
        }
        recon_tokens = {
            "prompt": recon_result.prompt_tokens,
            "completion": recon_result.completion_tokens,
            "total": recon_result.total_tokens,
        }
        error_message = "; ".join(apply_errors) if apply_errors else None
        return stored_ids, error_message, ops_counts, recon_tokens

    async def _direct_store(
        self,
        new_memories_meta: List[Dict],
        request: WriteRequest,
        vector_store: VectorStoreBase,
        req_id: str,
    ) -> Tuple[List[str], Optional[str], Dict[str, int], Dict[str, int]]:
        """
        跳过 reconcile，直接把 extractor 提取的 memories 插入 VDB。

        通过 RECONCILE_ENABLED=false 激活。适用于 eval 场景：
        不做去重/合并/演化，保留所有提取结果。
        """
        stored_ids: List[str] = []
        graph_counts = {"published": 0, "rejected": 0, "failures": 0}

        # 批量 embed
        contents = [m.get("content", "") for m in new_memories_meta if m.get("content")]
        if contents:
            batch_embeddings = await self.embed_service.embed_batch(contents)
        else:
            batch_embeddings = []

        emb_idx = 0
        for meta in new_memories_meta:
            content = meta.get("content", "")
            if not content:
                continue

            layer = MemoryLayer.from_string(meta.get("layer", "L2_FACT"))
            tags = meta.get("tags") or []
            speculate = meta.get("speculate")

            new_node = MemoryNode(
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "default_session",
                layer=layer,
                content=content,
                owner=_norm_owner(meta.get("owner")),
                speculate=speculate,
                is_latest=True,
                source_type=SourceType.INFERRED,
                status=MemoryStatus.ACTIVE,
                embedding=batch_embeddings[emb_idx] if emb_idx < len(batch_embeddings) else await self.embed_service.embed_queued(content),
                memory_at=request.memory_at,
                tags=list(tags),
                source_session_id=request.session_id or "default_session",
                source_turn_index=(meta.get("source_turn_indices") or [None])[0],
                evidence_chain=list(meta.get("source_turn_ids") or []),
                **_temporal_kwargs(
                    meta,
                    observed_fallback=request.memory_at,
                ),
            )
            emb_idx += 1
            nid = await vector_store.upsert(new_node)
            stored_ids.append(nid)
            await self._emit_reconcile_apply(
                request, vector_store=vector_store, op_type="ADD",
                memory_id=nid, layer=layer.value, content=content,
            )
            await self._maybe_index_entities(vector_store, new_node, request)
            graph_config = getattr(self.config, "graph_store", None)
            if getattr(graph_config, "graphrag_enabled", False):
                from ..models.graph_memory import sanitize_relation_candidates
                from ._retrieval.graphrag import publish_l2_fact_relations

                relations = sanitize_relation_candidates(
                    meta.get("relations") or [],
                    source_turn_ids=meta.get("source_turn_ids") or [],
                    min_confidence=float(
                        getattr(graph_config, "graphrag_min_confidence", 0.8)
                    ),
                )
                if relations:
                    try:
                        graph_summary = await publish_l2_fact_relations(
                            graph_store=self._graph_store,
                            fact_node=new_node,
                            relations=relations,
                        )
                    except Exception as graph_error:
                        graph_counts["failures"] += 1
                        logger.warning(
                            "[graphrag-write] direct publish failed for fact=%s: %s",
                            new_node.node_id,
                            graph_error,
                            exc_info=True,
                        )
                        continue
                    graph_counts["published"] += int(
                        graph_summary.get("published") or 0
                    )
                    graph_counts["rejected"] += int(
                        graph_summary.get("rejected") or 0
                    )
                    if not graph_summary.get("success"):
                        graph_counts["failures"] += 1

        logger.info(f"[direct_store] stored {len(stored_ids)} nodes (reconcile disabled)")
        n = len(stored_ids)
        return stored_ids, None, {
            "add": n,
            "supersede": 0,
            "update": 0,
            "total": n,
            "graph_published": graph_counts["published"],
            "graph_rejected": graph_counts["rejected"],
            "graph_failures": graph_counts["failures"],
        }, {"prompt": 0, "completion": 0, "total": 0}

    async def _store_summary(
        self,
        summary_content: str,
        source_raw_memory_id: str,
        request: WriteRequest,
        vector_store: VectorStoreBase,
    ) -> str:
        """存储 L3_SUMMARY 节点，返回 node_id。"""
        summary_node = MemoryNode(
            user_id=request.user_id,
            agent_id=request.agent_id or "default_agent",
            session_id=request.session_id or "default_session",
            layer=MemoryLayer.L3_SUMMARY,
            content=summary_content,
            source_type=SourceType.INFERRED,
            status=MemoryStatus.ACTIVE,
            is_latest=True,
            source_raw_memory_id=source_raw_memory_id,
            embedding=await self.embed_service.embed_queued(summary_content),
            memory_at=request.memory_at,
        )
        return await vector_store.upsert(summary_node)

    # ================================================================
    # Rolling Summary（topic-aware，buffer 攒够才 summary）
    # ================================================================

    def _summary_buffer_key(self, request: WriteRequest, bucket_date: str) -> str:
        """buffer key = user::agent::session::date（带 session，跨 session 不合并）。"""
        return "::".join([
            request.user_id or "",
            request.agent_id or "default_agent",
            request.session_id or "default_session",
            bucket_date,
        ])

    def _get_rolling_summarizer(self):
        """懒初始化 RollingSummarizer，复用 mem_agent 的 llm_provider。"""
        rs = getattr(self, "_rolling_summarizer", None)
        if rs is None:
            from ..agent.rolling_summary import RollingSummarizer
            rs = RollingSummarizer(self._mem_agent.llm_provider, self.config.llm)
            self._rolling_summarizer = rs
        return rs

    async def _maybe_rolling_summary(
        self,
        *,
        request: WriteRequest,
        raw_memory_id: str,
        vector_store: VectorStoreBase,
        req_id: str,
    ) -> List[str]:
        """
        把本次 messages 的 turn 追加进 buffer；达到 window 阈值则触发一次 rolling summary。

        返回本次新建的 L3_SUMMARY 节点 id 列表（未触发则空）。buffer 存 cache（无 cache 则跳过）。
        """
        if self._cache is None:
            return []

        # ── 当天日期（按 memory_at，无则今天）──
        _mem_at = request.memory_at
        bucket_date = (
            _mem_at.strftime("%Y-%m-%d") if _mem_at else datetime.now().strftime("%Y-%m-%d")
        )
        buffer_key = self._summary_buffer_key(request, bucket_date)
        window = int(getattr(self.config.llm, "summary_buffer_window", 10) or 10)

        # ── 本次 messages 转 turn（每条带 raw_id + 本次相对 turn_idx）──
        new_turns: List[Dict[str, Any]] = []
        new_user_count = 0
        for idx, m in enumerate(request.messages):
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "") or ""
            if not content.strip():
                continue
            new_turns.append({
                "raw_id": raw_memory_id,
                "turn_idx": idx,
                "role": role,
                "content": content,
            })
            if role == "user":
                new_user_count += 1

        if not new_turns:
            return []

        # ── 读已有 buffer，合并 ──
        existing = await self._cache.get_summary_buffer(buffer_key)
        pending_turns: List[Dict[str, Any]] = list(existing.get("pending_turns", [])) if existing else []
        pending_user_count = int(existing.get("pending_user_count", 0)) if existing else 0
        pending_turns.extend(new_turns)
        pending_user_count += new_user_count

        # ── 未达阈值：只攒不 summary ──
        if pending_user_count < window:
            await self._cache.upsert_summary_buffer(
                buffer_key,
                user_id=request.user_id or "",
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "default_session",
                bucket_date=bucket_date,
                pending_user_count=pending_user_count,
                pending_turns=pending_turns,
            )
            return []

        # ── 达到阈值：触发 rolling summary ──
        prev_summary = await self._latest_summary_text(request, vector_store)
        summarizer = self._get_rolling_summarizer()
        current_time = request.memory_at.isoformat(timespec="seconds") if request.memory_at else ""
        result = await summarizer.summarize(
            prev_summary=prev_summary,
            turns=pending_turns,
            current_time=current_time,
        )

        if not result.success:
            # 失败：保留 buffer 不动，等下次 add 再触发重试
            await self._cache.upsert_summary_buffer(
                buffer_key,
                user_id=request.user_id or "",
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "default_session",
                bucket_date=bucket_date,
                pending_user_count=pending_user_count,
                pending_turns=pending_turns,
            )
            return []

        # ── 落库：每条 summary 一个独立 L3 节点（无 chain）──
        new_ids: List[str] = []
        for item in result.summaries:
            raw_ids = self._lines_to_raw_ids(item.line_ids, pending_turns)
            sid = await self._store_rolling_summary(
                summary_content=item.content,
                topic=item.topic,
                source_raw_memory_ids=raw_ids,
                request=request,
                vector_store=vector_store,
            )
            new_ids.append(sid)

        # ── 更新 buffer：仅保留 keep 的 turn，其余（已 summary + 丢弃）清除 ──
        keep_set = set(result.keep_line_ids)
        kept_turns = [t for i, t in enumerate(pending_turns) if i in keep_set]
        kept_user_count = sum(1 for t in kept_turns if t.get("role") == "user")
        if kept_turns:
            await self._cache.upsert_summary_buffer(
                buffer_key,
                user_id=request.user_id or "",
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "default_session",
                bucket_date=bucket_date,
                pending_user_count=kept_user_count,
                pending_turns=kept_turns,
            )
        else:
            await self._cache.clear_summary_buffer(buffer_key)

        logger.info(
            f"[rolling-summary] req={req_id} key={buffer_key} "
            f"turns={len(pending_turns)} → summaries={len(new_ids)} kept={len(kept_turns)}"
        )
        return new_ids

    @staticmethod
    def _lines_to_raw_ids(
        line_ids: List[int], pending_turns: List[Dict[str, Any]]
    ) -> List[str]:
        """把展示行号映射回 raw_id，去重保序。"""
        out: List[str] = []
        seen: set = set()
        for li in line_ids:
            if 0 <= li < len(pending_turns):
                rid = pending_turns[li].get("raw_id") or ""
                if rid and rid not in seen:
                    seen.add(rid)
                    out.append(rid)
        return out

    async def _latest_summary_text(
        self, request: WriteRequest, vector_store: VectorStoreBase
    ) -> str:
        """取同 (user, agent, session) 当天最近一条 L3_SUMMARY 文本，作 prev_summary（可空）。"""
        try:
            nodes = await vector_store.list_by_user(
                user_id=request.user_id,
                agent_id=request.agent_id,
                layers=[MemoryLayer.L3_SUMMARY],
                status_filter=[MemoryStatus.ACTIVE],
                limit=50,
            )
        except Exception:
            return ""
        sess = request.session_id or "default_session"
        bucket_date = (
            request.memory_at.strftime("%Y-%m-%d")
            if request.memory_at else datetime.now().strftime("%Y-%m-%d")
        )
        candidates = []
        for n in nodes:
            if (n.session_id or "default_session") != sess:
                continue
            n_date = n.memory_at.strftime("%Y-%m-%d") if n.memory_at else ""
            if n_date and n_date != bucket_date:
                continue
            candidates.append(n)
        if not candidates:
            return ""
        candidates.sort(key=lambda n: n.gmt_created or datetime.min)
        return candidates[-1].content or ""

    async def _store_rolling_summary(
        self,
        *,
        summary_content: str,
        topic: str,
        source_raw_memory_ids: List[str],
        request: WriteRequest,
        vector_store: VectorStoreBase,
    ) -> str:
        """存一条 rolling L3_SUMMARY 节点（无 chain）；溯源用 raw_id list 存 custom。"""
        custom = {
            "topic": topic,
            "source_raw_memory_ids": source_raw_memory_ids,
            "summary_kind": "rolling",
        }
        summary_node = MemoryNode(
            user_id=request.user_id,
            agent_id=request.agent_id or "default_agent",
            session_id=request.session_id or "default_session",
            layer=MemoryLayer.L3_SUMMARY,
            content=summary_content,
            source_type=SourceType.INFERRED,
            status=MemoryStatus.ACTIVE,
            is_latest=True,
            source_raw_memory_id=(source_raw_memory_ids[0] if source_raw_memory_ids else ""),
            embedding=await self.embed_service.embed_queued(summary_content),
            memory_at=request.memory_at,
            custom=custom,
        )
        return await vector_store.upsert(summary_node)

    async def _emit_pipeline_step(
        self,
        request: WriteRequest,
        *,
        step: str,
        parsed: Union[str, Dict[str, Any], List[Any]],
        elapsed_ms: float = 0,
        response: str = "",
        prompt: str = "",
        memory_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """环节级 log/trace（经 client hook：文件始终 + DB 可关）。"""
        if not self._cache:
            return
        parsed_str = (
            parsed
            if isinstance(parsed, str)
            else json.dumps(parsed, ensure_ascii=False, default=str)
        )
        try:
            await self._cache.store_pipeline_log(
                request_id=request.request_id or get_request_id(),
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "",
                step=step,
                prompt=prompt,
                response=response,
                parsed=parsed_str,
                memory_ids=memory_ids,
                elapsed_ms=elapsed_ms,
                **kwargs,
            )
        except Exception as e:
            logger.debug(f"[write] pipeline step {step} failed: {e}")

    async def _emit_reconcile_apply(
        self,
        request: WriteRequest,
        *,
        vector_store,
        op_type: str,
        memory_id: str,
        layer: str,
        content: str,
        supersedes: Optional[List[str]] = None,
    ) -> None:
        """reconcile apply 级 pipeline log（RECONCILE_APPLY）。

        upsert/update_payload 之后立即回读校验该节点是否真落库，把结果写进
        pipeline trace。用于定位「reconcile 决定 ADD、账也记了，但 VDB 里查不到」
        这类问题——区分「没写」「写了没落库」「写后被删」。
        """
        verified = False
        verify_error = ""
        try:
            _node = await vector_store.get_by_id(memory_id)
            verified = _node is not None
            if not verified:
                verify_error = "get_by_id returned None (upsert returned id but not readable)"
        except Exception as e:
            verify_error = f"{type(e).__name__}: {e}"
        parsed = {
            "op": op_type,
            "memory_id": memory_id,
            "layer": layer,
            "content": (content or "")[:200],
            "supersedes": list(supersedes or []),
            "verified": verified,
        }
        if verify_error:
            parsed["verify_error"] = verify_error
        if not verified:
            logger.warning(
                f"[reconciler] RECONCILE_APPLY verify FAILED op={op_type} "
                f"mid={memory_id} layer={layer}: {verify_error}"
            )
        await self._emit_pipeline_step(
            request,
            step="RECONCILE_APPLY",
            parsed=parsed,
            memory_ids=[memory_id],
        )

    async def _emit_write_timeline(
        self,
        request: WriteRequest,
        tracer: PipelineTracer,
    ) -> None:
        """请求级 timeline 写入 Trace（SQLite），供 Inspector。"""
        if not self._cache or not is_pipeline_trace_enabled():
            return
        try:
            await self._cache.store_pipeline_log(
                request_id=request.request_id or get_request_id() or tracer.request_id,
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "",
                step="WRITE_TIMELINE",
                prompt="",
                response=tracer.to_summary_line(),
                parsed=json.dumps(tracer.to_dict(), ensure_ascii=False, default=str),
                elapsed_ms=0,
            )
        except Exception as e:
            logger.debug(f"[write] WRITE_TIMELINE failed: {e}")

    # ================================================================
    # 已抽取事实直写流程（跳过 extract，直接 reconcile）
    # ================================================================

    async def write_extracted(
        self,
        request: WriteRequest,
        ctx: Optional[PipelineContext] = None,
        tracer: Optional[PipelineTracer] = None,
    ) -> WriteResponse:
        """写入「agent 已抽取好的」记忆：跳过 extract，直接进 reconcile。

        适用于调用方（如 openclaw memory_add tool）已经把对话抽象成一条/多条
        结论性 memory content 的场景。跳过 LLM extract（省一次 LLM 调用），
        但仍走完整 reconcile：与已有记忆判重 / 合并 / 矛盾演化，并落 L2_FACT。

        与 write() 的区别：
        - 不写 L1_RAW 原始节点（content 本身即事实，无原始对话可留档）。
        - 不调 extractor（mem_agent.process_add）。
        - 直接把 content 列表喂给 _reconcile_and_store（或 RECONCILE_ENABLED=false
          时的 _direct_store）。

        输入约定：request.extra["extracted_contents"] = List[str]（每条一个事实）。
        为空时回退到 request.content（单条）。layer 统一 L2_FACT。
        """
        start_time = datetime.now()
        response = WriteResponse()

        if not request.request_id:
            request.request_id = get_request_id()

        if tracer is None:
            tracer = create_tracer(
                operation="write_extracted",
                pipeline_version="system1",
                uid=request.user_id,
                agent_id=request.agent_id,
                request_id=request.request_id,
                content_preview=request.content,
            )

        try:
            # 收集已抽取事实：优先 extra["extracted_contents"]，否则回退 content
            raw_contents = request.extra.get("extracted_contents")
            if not raw_contents:
                raw_contents = [request.content] if request.content else []
            contents = [c.strip() for c in raw_contents if isinstance(c, str) and c.strip()]

            if not contents:
                response.error_code = 400
                response.error_message = "extracted_contents or content is required"
                return response
            if not request.user_id:
                response.error_code = 400
                response.error_message = "user_id is required"
                return response

            if request.memory_at is None:
                request.memory_at = start_time

            _req_id = request.request_id or get_request_id()
            vector_store = await self._get_vector_store()

            # 直接构造 reconcile 输入（跳过 extract）。layer 统一 L2_FACT。
            new_memory_texts: List[str] = list(contents)
            new_memories_meta: List[Dict] = [
                {"content": c, "layer": "L2_FACT", "tags": []} for c in contents
            ]

            # 抽取好的多条之间互相判重（额外 embed 一次，丢弃重复项 + 记 DEDUP log）
            if len(new_memory_texts) >= 2:
                try:
                    new_memory_texts, new_memories_meta = await self._dedup_extracted(
                        new_memory_texts, new_memories_meta, request, _req_id,
                    )
                except Exception as e:
                    logger.warning(f"[write_extracted] dedup failed (non-fatal): {e}")

            _t_recon = datetime.now()
            if _RECONCILE_ENABLED:
                stored_ids, recon_error, ops_counts, recon_tokens = await self._reconcile_and_store(
                    new_memory_texts=new_memory_texts,
                    new_memories_meta=new_memories_meta,
                    request=request,
                    vector_store=vector_store,
                    req_id=_req_id,
                )
            else:
                stored_ids, recon_error, ops_counts, recon_tokens = await self._direct_store(
                    new_memories_meta=new_memories_meta,
                    request=request,
                    vector_store=vector_store,
                    req_id=_req_id,
                )
            _recon_ms = (datetime.now() - _t_recon).total_seconds() * 1000

            response.extra["reconcile_ops"] = ops_counts
            response.extra["reconcile_tokens"] = recon_tokens
            response.extra["extract_tokens"] = {"prompt": 0, "completion": 0, "total": 0}

            if recon_error is not None:
                response.success = False
                response.error_code = 502
                response.error_message = f"[RECONCILE_FAILED] {recon_error}"
                response.extra["agent_status"] = "failed"
                response.extra["agent_error"] = recon_error
                response.extra["agent_nodes"] = 0
                return response

            response.success = True
            response.layer = MemoryLayer.L2_FACT.value
            response.memory_id = stored_ids[0] if stored_ids else ""
            response.extra["agent_stored_ids"] = stored_ids
            response.extra["agent_status"] = "success"
            response.extra["agent_error"] = ""
            response.extra["agent_nodes"] = len(stored_ids)
            response.extra["agent_mode"] = "extracted"
            response.elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(
                f"[write_extracted] done: {len(stored_ids)} nodes stored from "
                f"{len(contents)} extracted contents, recon_ms={_recon_ms:.0f}"
            )
            return response

        except Exception as e:
            logger.error(f"[write_extracted] failed: {e}", exc_info=True)
            response.success = False
            response.error_code = 500
            response.error_message = str(e)
            return response
        finally:
            try:
                await self._emit_write_timeline(request, tracer)
            except Exception:
                pass

    # ================================================================
    # 主写入流程
    # ================================================================

    async def write(
        self,
        request: WriteRequest,
        ctx: Optional[PipelineContext] = None,
        tracer: Optional[PipelineTracer] = None,
    ) -> WriteResponse:
        """执行 Lite 写入流程。"""
        start_time = datetime.now()
        response = WriteResponse()

        # request_id 归一化：优先用 client 显式透传的 request.request_id（contextvar-immune），
        # 兜底用 contextvar（保护直接调用 write() 的测试 / 其它 caller）。
        # 此后整条链路落库统一读 request.request_id，不再依赖 contextvar。
        if not request.request_id:
            request.request_id = get_request_id()

        if tracer is None:
            tracer = create_tracer(
                operation="write",
                pipeline_version="system1",
                uid=request.user_id,
                agent_id=request.agent_id,
                request_id=request.request_id,
                content_preview=request.content,
            )

        try:
            # 参数校验
            content = request.content
            if not content and request.has_messages():
                content = request.get_flat_content()
                request.content = content

            if not content:
                response.error_code = 400
                response.error_message = "content or messages is required"
                return response
            if not request.user_id:
                response.error_code = 400
                response.error_message = "user_id is required"
                return response

            # ── 确保 memory_at 有值：未传则用接收到请求的时间 ──
            if request.memory_at is None:
                request.memory_at = start_time

            total_tokens = 0

            # 1. Layer 分配
            with tracer.span("layer_assign") as s:
                layer_str = request.extra.get("layer", "")
                if layer_str:
                    suggested_layer = layer_str
                    s.set_output({"layer": suggested_layer, "source": "explicit"})
                else:
                    suggested_layer = MemoryLayer.L1_RAW.value
                    s.set_output({"layer": suggested_layer, "source": "default_raw"})

            response.layer = suggested_layer

            # ── Timing: sys1_waiting_ms ──
            # start_time 是进入 write() 的时间，现在开始实际 I/O
            _t_process_start = datetime.now()
            _sys1_waiting_ms = (_t_process_start - start_time).total_seconds() * 1000
            _req_id_perf = request.request_id or get_request_id()
            logger.info(f"TRACE_PERF [{_req_id_perf}] S1_START waiting={_sys1_waiting_ms:.0f}ms user={request.user_id}")

            # Metrics: S1 开始
            from ..metrics import MetricsCollector
            MetricsCollector.get().sys1_start()

            # 2. 向量化 + 持久化（L1_RAW）
            _t_embed = datetime.now()
            with tracer.span("embed") as s:
                embedding = await self.embed_service.embed_queued(request.content)
                s.set_output({"dims": len(embedding), "content_len": len(request.content)})
            _embed_ms = (datetime.now() - _t_embed).total_seconds() * 1000
            response.extra["embedding"] = embedding
            await self._emit_pipeline_step(
                request,
                step="S1_EMBED",
                parsed={"dims": len(embedding), "content_len": len(request.content)},
                elapsed_ms=_embed_ms,
            )

            # 3. 持久化到向量库（L1_RAW）
            _t_l1 = datetime.now()
            memory_id = ""
            _l1_error: Optional[str] = None
            vector_store = None
            mem_node = None
            with tracer.span("qdrant_upsert") as s:
                try:
                    vector_store = await self._get_vector_store()
                    layer_enum = MemoryLayer.from_string(suggested_layer)
                    mem_node = MemoryNode(
                        user_id=request.user_id,
                        agent_id=request.agent_id or "default_agent",
                        session_id=request.session_id or "default_session",
                        layer=layer_enum,
                        content=request.content,
                        source_type=SourceType.EXPLICIT,
                        status=MemoryStatus.ACTIVE,
                        is_latest=True,
                        embedding=embedding,
                        memory_at=request.memory_at,
                    )
                    memory_id = await vector_store.upsert(mem_node)
                    response.memory_id = memory_id
                    s.set_output({"memory_id": memory_id, "layer": suggested_layer})
                    logger.debug(f"[write] persisted: id={memory_id} layer={suggested_layer}")
                except Exception as persist_err:
                    _l1_error = str(persist_err)
                    s.set_error(_l1_error)
                    logger.error(f"V1 Write: Persist failed (non-fatal): {persist_err}", exc_info=True)
            _l1_ms = (datetime.now() - _t_l1).total_seconds() * 1000
            # sparse 全文向量是否随本次 upsert 写入（tencent + BM25 可用时为 True）
            _sparse_enabled = bool(getattr(vector_store, "supports_fulltext", False))
            await self._emit_pipeline_step(
                request,
                step="S1_L1_UPSERT",
                parsed={
                    "memory_id": memory_id or None,
                    "layer": suggested_layer,
                    "content": (mem_node.content if mem_node else request.content),
                    "tags": list(mem_node.tags) if (mem_node and mem_node.tags) else [],
                    "sparse_enabled": _sparse_enabled,
                    "error": _l1_error,
                },
                elapsed_ms=_l1_ms,
                memory_ids=[memory_id] if memory_id else None,
            )

            # ── Timing: sys1_l1_process_ms ──
            _sys1_l1_process_ms = (datetime.now() - _t_process_start).total_seconds() * 1000
            logger.info(f"TRACE_PERF [{_req_id_perf}] S1_L1_DONE l1_ms={_sys1_l1_process_ms:.0f}ms")

            # 4. MemAgent 处理（可选）
            agent_mode = request.extra.get("agent_mode", "disabled")
            response.extra["agent_mode"] = agent_mode

            _t_workflow = datetime.now()
            if agent_mode == "full" and vector_store is not None and mem_node is not None:
                # ── 获取历史上下文（最近 k 轮对话，供 extractor 使用；k 由 config 配置）──
                _history_context = ""
                try:
                    _hist_turns = getattr(self.config.extractor, "history_turns", 5) or 5
                    _history_context = await self._get_recent_history(
                        vector_store, request.user_id, request.agent_id,
                        exclude_memory_id=memory_id,
                        max_turns=_hist_turns * 2,  # k 轮对话 = k*2 条 message
                    )
                except Exception as _hist_err:
                    logger.warning(f"[write] get_recent_history failed: {_hist_err}")

                with tracer.span("mem_agent") as s:
                    await self._run_agent(
                        request=request,
                        response=response,
                        vector_store=vector_store,
                        mem_node=mem_node,
                        memory_id=memory_id,
                        tracer_span=s,
                        history_context=_history_context,
                    )
                    total_tokens += response.extra.get("_agent_tokens", 0)
                if response.extra.get("agent_status") == "failed":
                    await self._emit_pipeline_step(
                        request,
                        step="S1_AGENT_FAILED",
                        parsed={
                            "error_code": response.extra.get("agent_error_code", ""),
                            "error": response.extra.get("agent_error", ""),
                        },
                    )
            elif agent_mode == "full":
                await self._emit_pipeline_step(
                    request,
                    step="S1_AGENT_SKIPPED",
                    parsed={"reason": "l1_upsert_failed_or_no_vector_store"},
                )

            # ── Timing: sys1_workflow_ms ──
            _sys1_workflow_ms = (datetime.now() - _t_workflow).total_seconds() * 1000
            logger.info(f"TRACE_PERF [{_req_id_perf}] S1_WORKFLOW_DONE workflow_ms={_sys1_workflow_ms:.0f}ms")

            # 5. 合并检测（可选）
            enable_merge_check = request.extra.get("enable_merge_check", False)
            should_merge = response.extra.get("should_merge", False)

            if enable_merge_check and not should_merge and request.existing_memories:
                with tracer.span("merge_check") as s:
                    existing_for_merge = [
                        {"memory_id": mem.get("memory_id", ""), "content": mem.get("content", "")}
                        for mem in request.existing_memories
                    ]
                    merge_result = self.merger.check_merge(
                        new_content=request.content,
                        existing_memories=existing_for_merge,
                    )
                    response.extra["should_merge"] = merge_result.should_merge
                    response.extra["merge_target_id"] = merge_result.target_memory_id or ""
                    s.set_output({
                        "should_merge": merge_result.should_merge,
                        "target_id": merge_result.target_memory_id or "",
                    })

            response.success = True
            response.tokens_used = total_tokens

            # ── 汇总 sys1 timing ──
            _ops_count = len(response.extra.get("agent_stored_ids", []))
            _ops_avg_ms = _sys1_workflow_ms / _ops_count if _ops_count > 0 else 0
            response.extra["timing"] = {
                "sys1_waiting_ms": round(_sys1_waiting_ms, 1),
                "sys1_l1_process_ms": round(_sys1_l1_process_ms, 1),
                "sys1_workflow_ms": round(_sys1_workflow_ms, 1),
                "sys1_ops_avg_ms": round(_ops_avg_ms, 1),
            }

            # Metrics: S1 完成 + VDB ops
            _mc = MetricsCollector.get()
            _mc.sys1_end(response.extra["timing"], success=True)
            if _ops_count > 0:
                for _ in range(_ops_count):
                    _mc.record_vdb_op(_ops_avg_ms)

        except Exception as e:
            logger.error(f"MemoryWriter.write failed: {e}", exc_info=True)
            response.error_code = 500
            response.error_message = str(e)
            tracer.set_error(str(e))
            # Metrics: S1 失败
            try:
                MetricsCollector.get().sys1_end({}, success=False)
            except Exception:
                pass
        finally:
            response.elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            tracer.set_output({
                "success": response.success,
                "layer": response.layer,
                "memory_id": response.memory_id,
                "entities_count": len(response.entities),
                "entities": [
                    {"name": e.get("name", ""), "type": e.get("type", "")}
                    if isinstance(e, dict) else {"name": str(e)}
                    for e in response.entities
                ],
                "summary": response.extra.get("summary", ""),
                "content_stored": request.content,
                "tokens_used": response.tokens_used,
                "pipeline_ms": response.elapsed_ms,
                "agent_status": response.extra.get("agent_status", ""),
            })
            await self._emit_write_timeline(request, tracer)
            tracer.finish(write_file=False)

        return response

    async def _get_recent_history(
        self,
        vector_store: VectorStoreBase,
        user_id: str,
        agent_id: Optional[str] = None,
        exclude_memory_id: str = "",
        max_turns: int = 20,
        max_chars_assistant: int = 500,
    ) -> str:
        """
        获取该用户最近的 L1_RAW 对话记录，按轮次拆分后拼成 history context。

        每条 L1_RAW 的 content 是多轮对话（[user]: ...\n[assistant]: ...）。
        拆成单条 message 后，user 消息完整保留，assistant 消息截取前 max_chars_assistant 字符。
        最终取最近 max_turns 条 message（注意单位是 message 条数，非对话轮数；
        调用方按 k 轮 × 2 传入）。

        Returns:
            格式化的历史上下文字符串，空字符串表示无历史。
        """
        try:
            nodes = await vector_store.list_by_user(
                user_id=user_id,
                agent_id=agent_id,
                layers=[MemoryLayer.L1_RAW],
                status_filter=[MemoryStatus.ACTIVE, MemoryStatus.SHADOW],
                limit=max(50, max_turns),
            )
            if not nodes:
                return ""

            # 排除当前正在处理的 memory（避免自引用）
            nodes = [n for n in nodes if n.node_id != exclude_memory_id]

            # 按 gmt_created 升序（时间正序）
            nodes.sort(key=lambda n: n.gmt_created or datetime.min)

            # 拆分每条 L1_RAW 内容为单条 messages。
            # L1_RAW content 格式: "[user]: xxx\n[assistant]: yyy\n[user]: zzz..."
            # 关键：单条 message 的 content 本身可能含换行（多行 user 输入），
            # 因此必须按「[role]: 前缀」识别消息边界，无前缀的行视为上一条消息的续行
            # 归属其下，而不是逐行当成独立 message——否则续行会被当成裸消息，且后续
            # 按行数截断时可能从一条多行消息中间切开，导致前缀行丢失、续行裸露在开头。
            import re as _re
            _role_prefix = _re.compile(r"^\[?(user|assistant|system|tool)\]?:\s?")

            # (role, content) 元组列表
            parsed: List[Tuple[str, str]] = []
            for node in nodes:
                content = node.content or ""
                for line in content.split("\n"):
                    m = _role_prefix.match(line)
                    if m:
                        role = m.group(1)
                        body = line[m.end():]
                        parsed.append((role, body))
                    elif parsed:
                        # 续行：拼回上一条 message（保留换行）
                        prev_role, prev_body = parsed[-1]
                        parsed[-1] = (prev_role, prev_body + "\n" + line)
                    else:
                        # 开头就是无前缀行（异常/老数据）：当成 user 消息兜底
                        parsed.append(("user", line))

            # 取最近 max_turns 条 message（按 message 截断，绝不从消息中间切开）
            recent = parsed[-max_turns:] if len(parsed) > max_turns else parsed
            if not recent:
                return ""

            # 格式化：assistant 整条截断，user 完整保留
            all_messages: List[str] = []
            for role, body in recent:
                body = body.strip()
                if role == "assistant" and len(body) > max_chars_assistant:
                    body = body[:max_chars_assistant] + "..."
                all_messages.append(f"[{role}]: {body}")

            return "\n".join(all_messages)
        except Exception as e:
            logger.debug(f"[write] _get_recent_history error: {e}")
            return ""

    async def _collect_existing_tags(
        self,
        vector_store: VectorStoreBase,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """
        收集该用户在 VDB 中已有的所有 tags（去重）。

        从 vector_store.list_by_user 的结果中提取所有 tags 字段，
        合并为唯一集合，供 extract prompt 引导 LLM 优先复用已有标签。
        失败静默返回空列表（不阻塞主流程）。
        """
        try:
            nodes = await vector_store.list_by_user(
                user_id=user_id,
                agent_id=agent_id,
                limit=10000,
            )
            tag_set: set = set()
            for node in nodes:
                if hasattr(node, "tags") and node.tags:
                    tag_set.update(node.tags)
            tags = sorted(tag_set)
            if tags:
                logger.debug(f"[write] collected {len(tags)} existing tags for user={user_id}")
            return tags
        except Exception as e:
            logger.warning(f"[write] _collect_existing_tags failed (non-fatal): {e}")
            return []

    async def _run_agent(
        self,
        request: WriteRequest,
        response: WriteResponse,
        vector_store: VectorStoreBase,
        mem_node: MemoryNode,
        memory_id: str,
        tracer_span,
        history_context: str = "",
    ) -> None:
        """
        MemAgent 完整处理流程：extract → reconcile & store → summary。
        结果写回 response.extra。
        """
        mode = ProcessMode.FULL
        existing_memories = []
        for mem in (request.existing_memories or []):
            existing_memories.append({
                "memory_id": mem.get("memory_id", ""),
                "content": mem.get("content", ""),
                "layer": mem.get("layer", ""),
                "embedding": mem.get("embedding", []),
            })

        try:
            # 收集该用户已有的所有 tags（供 extractor prompt 引导复用）
            existing_tags = await self._collect_existing_tags(
                vector_store, request.user_id, request.agent_id
            )

            # 构建 extractor 输入内容：优先用原始 messages dump
            if request.messages:
                import json as _json_msgs
                _extract_content = _json_msgs.dumps(
                    self._request_turn_records(request),
                    ensure_ascii=False,
                )
            else:
                _extract_content = request.content

            # 基础画像 schema 字段（{name: description}），由 extractor 渲染到 prompt
            _basic_profile_fields = self.config.basic_profile.effective_fields()

            # 本次有效 summary 开关（per-call > config）。summary 不再由 mem_agent 单次生成，
            # 改由下方 rolling buffer 流程处理（仅 messages），因此这里给 mem_agent 传 False，
            # 让老的「单次 add 即 summary」路径自然成为 no-op（零风险）。
            _effective_enable_summary = (
                request.enable_summary
                if request.enable_summary is not None
                else self.config.llm.enable_summary
            )

            agent_result = await self.mem_agent.process_add(
                content=_extract_content,
                context={"uid": request.user_id, "agent_id": request.agent_id},
                mode=mode,
                existing_memories=existing_memories,
                memory_at=request.memory_at,
                existing_tags=existing_tags,
                history_context=history_context,
                enable_summary=False,
                basic_profile_fields=_basic_profile_fields,
            )
        except Exception as agent_err:
            logger.error(f"[write] MemAgent process_add raised: {agent_err}", exc_info=True)
            response.extra["agent_status"] = "failed"
            response.extra["agent_error"] = str(agent_err)
            response.extra["agent_nodes"] = 0
            tracer_span.set_output({"success": False, "error": str(agent_err)})
            return

        if agent_result is None or not agent_result.success:
            await self._handle_agent_failure(request, response, agent_result, tracer_span)
            return

        # ── 提取成功 ──
        self._sanitize_extracted_provenance(
            agent_result.extracted_info or {}, request,
        )
        provenance_validation = await self._validate_extracted_provenance(
            agent_result.extracted_info or {}, request,
        )
        response.extra["provenance_validation"] = provenance_validation
        _perf_req_id = request.request_id or get_request_id()
        logger.info(
            f"TRACE_PERF [{_perf_req_id}] S1_EXTRACT_DONE "
            f"extract_ms={agent_result.extract_elapsed_ms:.0f} "
            f"summary_ms={agent_result.summary_elapsed_ms:.0f} "
            f"memory={len((agent_result.extracted_info or {}).get('memory', (agent_result.extracted_info or {}).get('facts', [])))}"
        )

        # ── basic_info: prompt-driven schema, no LLM function-calling ──
        # extractor 把 basic_info dict 放进 extracted_info；writer 在此 upsert L0_BASIC_INFO
        # 演化链。失败/无效都不抛错，落 tool_results_summary 走原 TOOL_CALLS pipeline log。
        basic_info_raw = None
        if isinstance(agent_result.extracted_info, dict):
            sanitized_basic_info = self._sanitize_basic_info(
                agent_result.extracted_info, request,
            )
            agent_result.extracted_info.pop("basic_info", None)
            basic_info_raw = sanitized_basic_info

        tool_calls_raw = None  # v0.1.5.13+ 不再有真实 LLM tool_calls；保留字段做兼容
        tool_results_summary: List[Dict[str, Any]] = []

        if isinstance(basic_info_raw, dict) and basic_info_raw:
            try:
                _bp_result = await upsert_basic_profile(
                    user_id=request.user_id,
                    agent_id=request.agent_id or "default_agent",
                    session_id=request.session_id or "default_session",
                    kv=basic_info_raw,
                    vector_store=vector_store,
                    embed_service=self.embed_service,
                    allowed_fields=list(_basic_profile_fields.keys()),
                )
                # 把 upsert 结果包成原 tool_results 形态（向后兼容 pipeline log 解析）
                tool_results_summary.append({
                    "tool": "basic_profile_upsert",  # 不再是 LLM tool name
                    "round": 1,
                    "success": _bp_result.success,
                    "data": _bp_result.to_dict(),
                    "error": _bp_result.error,
                })
                # input 侧记录 LLM 给出的 basic_info 原始值（trace 可追溯）
                tool_calls_raw = [{
                    "function": {
                        "name": "basic_profile_upsert",
                        "arguments": json.dumps(basic_info_raw, ensure_ascii=False, default=str),
                    }
                }]
            except Exception as bp_err:
                logger.error(f"[write] upsert_basic_profile raised: {bp_err}", exc_info=True)
                tool_results_summary.append({
                    "tool": "basic_profile_upsert",
                    "round": 1,
                    "success": False,
                    "error": str(bp_err),
                })

        response.extra["tool_results"] = tool_results_summary

        # 写 TOOL_CALLS pipeline log（可观测 tool 调用链路）
        if self._cache and (tool_calls_raw or tool_results_summary):
            try:
                import json as _json_tc
                _tc_req_id = request.request_id or get_request_id()
                tc_data = {
                    "tool_calls_input": (
                        [{"name": tc.get("function", {}).get("name", ""), "arguments": tc.get("function", {}).get("arguments", "")}
                         for tc in tool_calls_raw]
                        if isinstance(tool_calls_raw, list) else str(tool_calls_raw)[:500]
                    ) if tool_calls_raw else [],
                    "tool_results": tool_results_summary,
                    "tool_calls_only": bool(
                        tool_calls_raw
                        and not (agent_result.extracted_info or {}).get("memory")
                        and not (agent_result.extracted_info or {}).get("facts")
                        and not (agent_result.extracted_info or {}).get("identity")
                    ),
                }
                await self._cache.store_pipeline_log(
                    request_id=_tc_req_id,
                    user_id=request.user_id,
                    agent_id=request.agent_id or "default_agent",
                    session_id=request.session_id or "",
                    step="TOOL_CALLS",
                    prompt="",
                    response=_json_tc.dumps(tc_data, ensure_ascii=False, default=str),
                    parsed=_json_tc.dumps(tc_data, ensure_ascii=False, default=str),
                    memory_ids=[],
                )
            except Exception as e:
                logger.warning(f"[write] store TOOL_CALLS log failed: {e}")
        # 过滤实体
        _CATEGORY_BLACKLIST = {
            "locations", "location", "persons", "person",
            "organizations", "organization", "events", "event",
            "relations", "relation", "products", "product",
            "animals", "animal", "foods", "food",
            "technologies", "technology", "tech",
            "attributes", "attribute", "others", "other",
        }
        if agent_result.extracted_info and isinstance(agent_result.extracted_info, dict):
            for entity in (agent_result.extracted_info.get("entities") or []):
                if isinstance(entity, dict):
                    name = entity.get("name", "")
                    if name and name.lower() not in _CATEGORY_BLACKLIST:
                        response.entities.append(entity)
                elif isinstance(entity, str):
                    if entity.lower() not in _CATEGORY_BLACKLIST:
                        response.entities.append({"name": entity})

        response.extra["summary"] = agent_result.summary or ""
        if agent_result.suggested_layer:
            response.layer = agent_result.suggested_layer

        conflicts = []
        for conflict in (agent_result.conflicts or []):
            conflicts.append({
                "type": conflict.get("type", ""),
                "target_id": conflict.get("target_id", ""),
                "description": conflict.get("description", ""),
                "resolution": conflict.get("resolution", ""),
            })
        response.extra["conflicts"] = conflicts
        response.extra["should_merge"] = agent_result.should_merge
        response.extra["merge_target_id"] = agent_result.merge_target_id or ""
        response.extra["_agent_tokens"] = agent_result.tokens_used
        # extract 链路 LLM token 消耗（prompt/completion/total），供 client 透传给调用方
        response.extra["extract_tokens"] = {
            "prompt": getattr(agent_result, "extract_prompt_tokens", 0) or 0,
            "completion": getattr(agent_result, "extract_completion_tokens", 0) or 0,
            "total": getattr(agent_result, "extract_tokens_used", 0) or 0,
        }

        # 写 pipeline logs
        _req_id = request.request_id or get_request_id()
        await self._write_extract_log(request, agent_result, _req_id, tool_results=tool_results_summary)
        await self._write_summary_log(request, agent_result, _req_id)

        # Reconcile & store
        stored_ids: List[str] = []
        _vs_logger = logging.getLogger("hy_memory.data.vector_store_chroma")
        _vs_level = _vs_logger.level
        _vs_logger.setLevel(logging.INFO)
        try:
            new_memory_texts, new_memories_meta = self._collect_new_memories(
                agent_result.extracted_info or {}
            )

            # extractor 结果去重：新提取的多条之间互相判重（额外 embed 一次，
            # 还没入库故 delete_from_store=False，只丢弃重复项 + 记 DEDUP log）。
            if len(new_memory_texts) >= 2:
                try:
                    new_memory_texts, new_memories_meta = await self._dedup_extracted(
                        new_memory_texts, new_memories_meta, request, _req_id,
                    )
                except Exception as e:
                    logger.warning(f"[write] extractor dedup failed (non-fatal): {e}")

            if new_memory_texts:
                if _RECONCILE_ENABLED:
                    stored_ids, recon_error, ops_counts, recon_tokens = await self._reconcile_and_store(
                        new_memory_texts=new_memory_texts,
                        new_memories_meta=new_memories_meta,
                        request=request,
                        vector_store=vector_store,
                        req_id=_req_id,
                    )
                else:
                    stored_ids, recon_error, ops_counts, recon_tokens = await self._direct_store(
                        new_memories_meta=new_memories_meta,
                        request=request,
                        vector_store=vector_store,
                        req_id=_req_id,
                    )
                response.extra["reconcile_ops"] = ops_counts
                response.extra["reconcile_tokens"] = recon_tokens
                if recon_error is not None:
                    response.success = False
                    response.error_code = 502
                    response.error_message = f"[RECONCILE_FAILED] {recon_error}"
                    response.extra["agent_status"] = "failed"
                    response.extra["agent_error"] = recon_error
                    response.extra["agent_nodes"] = 0
                    logger.warning(f"[write] reconcile failed: {recon_error}")
                    return

            # Intentions → L7（直接 upsert，不走 reconcile）
            intentions = self._collect_intentions(agent_result.extracted_info or {})
            if intentions:
                intention_ids = await self._store_intentions(
                    intentions=intentions,
                    request=request,
                    vector_store=vector_store,
                    req_id=_req_id,
                )
                stored_ids.extend(intention_ids)
                logger.info(
                    f"TRACE_PERF [{_perf_req_id}] S1_INTENTION_DONE "
                    f"intentions={len(intention_ids)}"
                )

            # Summary → L3
            if agent_result.summary:
                sid = await self._store_summary(
                    summary_content=agent_result.summary,
                    source_raw_memory_id=memory_id,
                    request=request,
                    vector_store=vector_store,
                )
                stored_ids.append(sid)

            # Rolling summary（topic-aware，仅 messages + enable_summary 开启时）。
            # 独立阶段：累积 buffer，攒够 user 轮次再批量 summary。失败不影响主写入。
            if _effective_enable_summary and request.has_messages():
                try:
                    rolling_ids = await self._maybe_rolling_summary(
                        request=request,
                        raw_memory_id=memory_id,
                        vector_store=vector_store,
                        req_id=_req_id,
                    )
                    if rolling_ids:
                        stored_ids.extend(rolling_ids)
                except Exception as _rs_err:
                    logger.warning(f"[write] rolling summary failed (non-fatal): {_rs_err}")

            response.extra["agent_stored_ids"] = stored_ids
            response.extra["agent_status"] = "success"
            response.extra["agent_error"] = ""
            response.extra["agent_nodes"] = len(stored_ids)
            _vs_logger.setLevel(_vs_level)
            logger.info(
                f"TRACE_PERF [{_perf_req_id}] S1_PERSIST_DONE "
                f"ops={len(stored_ids)} nodes to vector store"
            )
            logger.debug(f"[agent] persisted {len(stored_ids)} nodes to vector store")

            # L1_RAW → SHADOW
            if stored_ids and memory_id:
                try:
                    mem_node.status = MemoryStatus.SHADOW
                    await vector_store.upsert(mem_node)
                    logger.debug(f"[agent] L1 raw {memory_id} status → SHADOW")
                except Exception as shadow_err:
                    logger.warning(f"[agent] failed to shadow L1 raw: {shadow_err}")

        except Exception as persist_err:
            _vs_logger.setLevel(_vs_level)
            logger.error(f"[agent] persist failed: {persist_err}", exc_info=True)
            response.extra["agent_stored_ids"] = stored_ids
            response.extra["agent_status"] = "failed"
            response.extra["agent_error"] = f"agent persist failed: {persist_err}"
            response.extra["agent_nodes"] = len(stored_ids)
            return

        tracer_span.set_output({
            "success": True,
            "entities_count": len(response.entities),
            "summary": agent_result.summary or "",
            "conflicts_count": len(conflicts),
            "tokens_used": agent_result.tokens_used,
            "should_merge": agent_result.should_merge,
            "agent_stored_count": len(stored_ids),
        })

    async def _handle_agent_failure(self, request, response, agent_result, tracer_span) -> None:
        """处理 agent extract 失败的情况。"""
        if agent_result is None:
            tracer_span.set_output({"success": False, "error": "agent processing failed"})
            return

        _error_code = getattr(agent_result, "error_code", "") or "AGENT_FAILED"
        _error_msg = agent_result.error or "agent processing failed"
        response.extra["agent_status"] = "failed"
        response.extra["agent_error"] = _error_msg
        response.extra["agent_error_code"] = _error_code
        response.extra["agent_nodes"] = 0
        response.success = False
        response.error_code = 502
        response.error_message = f"[{_error_code}] {_error_msg}"
        logger.warning(f"[write] agent extract failed: code={_error_code} error={_error_msg}")

        _raw_resp = getattr(agent_result, "extract_raw_response", "") or ""
        # 复原真实 prompt（[SYSTEM]/[USER]），与成功路径 _write_extract_log 一致，
        # 使失败（EMPTY_RESPONSE / JSON_PARSE_FAILED / LLM_ERROR）的 EXTRACT log 也能看到实际输入。
        _extract_prompt = ""
        _attempts = []
        _er = getattr(agent_result, "_extract_result", None)
        if _er is not None:
            if getattr(_er, "_actual_prompt", None):
                _sys = getattr(_er, "_actual_system_prompt", "") or ""
                _usr = getattr(_er, "_actual_prompt", "") or ""
                _extract_prompt = f"[SYSTEM]\n{_sys}\n\n[USER]\n{_usr}"
            _attempts = list(getattr(_er, "attempts", []) or [])
        _parsed = {"error": _error_code, "message": _error_msg}
        if _attempts:
            # 每一次 LLM 重试的失败明细（类型/消息/HTTP 状态码等），供排障
            _parsed["attempts"] = _attempts
        await self._emit_pipeline_step(
            request,
            step="EXTRACT",
            parsed=_parsed,
            prompt=_extract_prompt,
            response=_raw_resp,
            elapsed_ms=getattr(agent_result, "extract_elapsed_ms", 0) or 0,
            prompt_tokens=getattr(agent_result, "extract_prompt_tokens", 0) or 0,
            completion_tokens=getattr(agent_result, "extract_completion_tokens", 0) or 0,
            total_tokens=getattr(agent_result, "extract_tokens_used", 0) or 0,
        )

        tracer_span.set_output({"success": False, "error": _error_msg})

    async def _write_extract_log(self, request, agent_result, req_id: str, tool_results: List[Dict[str, Any]] = None) -> None:
        """写 EXTRACT pipeline log。"""
        if not self._cache:
            return
        try:
            import json as _json

            # 使用 extractor 返回的真实 prompt（如果有的话）
            _extract_prompt = ""
            _extract_result = getattr(agent_result, '_extract_result', None)
            if _extract_result and hasattr(_extract_result, '_actual_prompt') and _extract_result._actual_prompt:
                _sys = _extract_result._actual_system_prompt or ""
                _usr = _extract_result._actual_prompt or ""
                _extract_prompt = f"[SYSTEM]\n{_sys}\n\n[USER]\n{_usr}"
            else:
                # fallback: 用旧方式构造近似 prompt（不完全准确）
                from ..agent.extractor import EXTRACT_PROMPT
                _current_time = request.memory_at.isoformat(timespec="seconds") if request.memory_at else ""
                _extract_prompt = EXTRACT_PROMPT.format(content=request.content, current_date="", memory_at=_current_time or "", existing_tags="(see actual prompt)", last_messages="")

            # response 字段：优先用 raw_response（LLM 原始输出）
            _raw_resp = agent_result.extract_raw_response or ""

            # parsed 字段：包含 extracted_info + tool_results 元信息
            _parsed_data = dict(agent_result.extracted_info) if agent_result.extracted_info else {}
            if tool_results:
                _parsed_data["_tool_results"] = tool_results

            await self._cache.store_pipeline_log(
                request_id=req_id,
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "",
                step="EXTRACT",
                prompt=_extract_prompt,
                response=_raw_resp if _raw_resp else _json.dumps(_parsed_data, ensure_ascii=False, default=str),
                parsed=_json.dumps(_parsed_data, ensure_ascii=False, default=str) if _parsed_data else "",
                elapsed_ms=agent_result.extract_elapsed_ms,
                prompt_tokens=agent_result.extract_prompt_tokens,
                completion_tokens=agent_result.extract_completion_tokens,
                total_tokens=agent_result.extract_tokens_used,
            )
        except Exception as e:
            logger.warning(f"[write] store EXTRACT log failed: {e}")

    async def _write_summary_log(self, request, agent_result, req_id: str) -> None:
        """写 SUMMARY pipeline log（若有摘要）。"""
        if not self._cache or not agent_result.summary or not agent_result.summary_tokens_used:
            return
        try:
            import json as _json_s

            # 使用 summarizer 返回的真实 prompt
            _summary_result = getattr(agent_result, '_summary_result', None)
            if _summary_result and hasattr(_summary_result, '_actual_prompt') and _summary_result._actual_prompt:
                _summary_prompt = _summary_result._actual_prompt
            else:
                # fallback
                from ..agent.summarizer import SUMMARY_PROMPT
                from datetime import date as _date_now
                _current_time = request.memory_at.isoformat(timespec="seconds") if request.memory_at else ""
                _memory_date = _current_time[:10] if _current_time and len(_current_time) >= 10 else _date_now.today().isoformat()
                _current_date = _date_now.today().isoformat()
                _summary_prompt = SUMMARY_PROMPT.format(
                    content=request.content or "",
                    memory_date=_memory_date,
                    current_date=_current_date,
                )
            await self._cache.store_pipeline_log(
                request_id=req_id,
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "",
                step="SUMMARY",
                prompt=_summary_prompt,
                response=agent_result.summary,
                parsed=_json_s.dumps({"summary": agent_result.summary}, ensure_ascii=False),
                elapsed_ms=agent_result.summary_elapsed_ms,
                prompt_tokens=agent_result.summary_prompt_tokens,
                completion_tokens=agent_result.summary_completion_tokens,
                total_tokens=agent_result.summary_tokens_used,
            )
        except Exception as e:
            logger.warning(f"[write] store SUMMARY log failed: {e}")

    async def close(self) -> None:
        logger.debug("MemoryWriter closed")
