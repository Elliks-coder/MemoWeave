"""Entity-Fact GraphRAG 的写入发布与读取证据回查编排。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...models.graph_memory import sanitize_relation_candidates
from ...models.memory import MemoryLayer, MemoryNode, MemoryStatus


def collect_relations_for_op(
    op: Any,
    new_memories_meta: List[Dict[str, Any]],
    *,
    min_confidence: float,
) -> List[Dict[str, Any]]:
    """按 reconciler source_indices 把 L1 候选关系对齐到最终 L2 op。"""
    indices = list(getattr(op, "source_indices", None) or [])
    if not indices and len(new_memories_meta) == 1:
        indices = [0]

    accepted: List[Dict[str, Any]] = []
    seen = set()
    for raw_index in indices:
        try:
            meta = new_memories_meta[int(raw_index)]
        except (IndexError, TypeError, ValueError):
            continue
        sanitized = sanitize_relation_candidates(
            meta.get("relations") or [],
            source_turn_ids=meta.get("source_turn_ids") or [],
            min_confidence=min_confidence,
        )
        for relation in sanitized:
            key = (
                relation["subject_normalized"],
                relation["predicate"],
                relation["object_normalized"],
            )
            if key not in seen:
                seen.add(key)
                accepted.append(relation)
    return accepted


async def publish_l2_fact_relations(
    *,
    graph_store: Any,
    fact_node: MemoryNode,
    relations: List[Dict[str, Any]],
    supersedes_fact_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if graph_store is None:
        return {
            "available": False,
            "success": False,
            "published": 0,
            "rejected": len(relations or []),
            "errors": ["graph store was not injected"],
        }
    return await graph_store.publish_fact_relations(
        fact_node,
        relations,
        supersedes_fact_ids=supersedes_fact_ids or [],
    )


def _node_visible_at(node: MemoryNode, as_of: Optional[datetime]) -> bool:
    if as_of is None:
        return node.status == MemoryStatus.ACTIVE and bool(node.is_latest)
    if node.status not in {MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED}:
        return False
    point = as_of.timestamp()
    if node.valid_from is not None and node.valid_from.timestamp() > point:
        return False
    if node.valid_until is not None and point > node.valid_until.timestamp():
        return False
    return True


async def expand_graph_evidence(
    *,
    graph_store: Any,
    vector_store: Any,
    query: str,
    normal_results: List[Dict[str, Any]],
    tenant_keys: List[str],
    user_ids: List[str],
    as_of: Optional[datetime],
    max_hops: int,
    max_facts: int,
    max_degree: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """扩展图路径并回到 Chroma 读取 L2 原文，图本身不充当答案文本。"""
    anchors = [
        item for item in normal_results
        if item.get("node") is not None
        and getattr(item["node"], "layer", None) == MemoryLayer.L2_FACT
    ]
    anchor_ids = list(dict.fromkeys(
        str(item.get("node_id") or "") for item in anchors
        if str(item.get("node_id") or "")
    ))
    anchor_scores = {
        str(item.get("node_id")): float(item.get("score") or 0.0)
        for item in anchors
    }

    raw_candidates = await graph_store.expand_fact_relations(
        anchor_fact_ids=anchor_ids,
        query=query,
        tenant_keys=tenant_keys,
        user_ids=user_ids,
        as_of=as_of,
        max_hops=max_hops,
        max_facts=max_facts,
        max_degree=max_degree,
    )

    hits: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for candidate in raw_candidates:
        fact_id = str(candidate.get("fact_id") or "")
        if not fact_id or fact_id in anchor_scores:
            continue
        node = await vector_store.get_by_id(fact_id)
        if (
            node is None
            or node.layer != MemoryLayer.L2_FACT
            or not _node_visible_at(node, as_of)
        ):
            rejected.append({
                "fact_id": fact_id,
                "reason": "missing_or_not_visible_l2",
            })
            continue

        hop = max(1, int(candidate.get("hop") or 1))
        confidence = max(0.0, min(1.0, float(candidate.get("confidence") or 0.0)))
        anchor_id = str(candidate.get("anchor_fact_id") or "")
        base_score = anchor_scores.get(anchor_id, 0.62)
        graph_score = max(0.0, min(0.99, base_score * (0.9 ** hop) * confidence))
        hits.append({
            "node_id": fact_id,
            "node": node,
            "score": graph_score,
            "source": "entity_fact_graph",
            "graph_added": True,
            "graph_hop": hop,
            "graph_anchor_fact_id": anchor_id,
            "graph_origin": candidate.get("origin", ""),
            "graph_path": candidate.get("path") or [],
            "graph_confidence": confidence,
        })

    hits.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    hits = hits[:max_facts]
    summary = {
        "anchor_fact_ids": anchor_ids,
        "candidate_count": len(raw_candidates),
        "evidence_count": len(hits),
        "rejected": rejected,
        "paths": [
            {
                "fact_id": item["node_id"],
                "anchor_fact_id": item.get("graph_anchor_fact_id", ""),
                "hop": item.get("graph_hop", 0),
                "score": round(float(item.get("score", 0.0)), 6),
                "path": item.get("graph_path") or [],
            }
            for item in hits
        ],
    }
    return hits, summary
