"""Entity-Fact GraphRAG 的轻量领域模型与输入校验。

图关系来自 L1 extractor，但只有完成 L2 reconcile 后才会调用本模块清洗并
发布。这里刻意采用保守策略：只接受显式关系、精确实体名称和较高置信度；
宁可漏掉模糊关系，也不把同段共现误写成长期事实。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional


ALLOWED_PREDICATES = {
    "KNOWS",
    "FRIEND_OF",
    "PARTNER_OF",
    "FAMILY_OF",
    "PARENT_OF",
    "CHILD_OF",
    "SIBLING_OF",
    "COLLEAGUE_OF",
    "WORKS_AT",
    "STUDIES_AT",
    "MEMBER_OF",
    "LIVES_IN",
    "FROM",
    "LOCATED_IN",
    "LIKES",
    "DISLIKES",
    "OWNS",
    "USES",
    "CREATED",
    "PARTICIPATED_IN",
    "ATTENDED",
    "VISITED",
    "HAS_PET",
    "CARES_FOR",
    "INTERESTED_IN",
    "LEARNING",
    "RELATED_TO",
}

_PREDICATE_ALIASES = {
    "朋友": "FRIEND_OF",
    "伴侣": "PARTNER_OF",
    "家人": "FAMILY_OF",
    "父母": "PARENT_OF",
    "子女": "CHILD_OF",
    "兄弟姐妹": "SIBLING_OF",
    "同事": "COLLEAGUE_OF",
    "工作于": "WORKS_AT",
    "就读于": "STUDIES_AT",
    "成员": "MEMBER_OF",
    "居住于": "LIVES_IN",
    "来自": "FROM",
    "位于": "LOCATED_IN",
    "喜欢": "LIKES",
    "不喜欢": "DISLIKES",
    "拥有": "OWNS",
    "使用": "USES",
    "创建": "CREATED",
    "参加": "PARTICIPATED_IN",
    "访问": "VISITED",
    "宠物": "HAS_PET",
    "学习": "LEARNING",
}

_ENTITY_TYPE_ALIASES = {
    "person": "person",
    "people": "person",
    "人物": "person",
    "人": "person",
    "place": "place",
    "location": "place",
    "地点": "place",
    "organization": "organization",
    "organisation": "organization",
    "org": "organization",
    "组织": "organization",
    "event": "event",
    "事件": "event",
    "product": "product",
    "物品": "product",
    "activity": "activity",
    "活动": "activity",
    "animal": "animal",
    "pet": "animal",
    "动物": "animal",
    "concept": "concept",
    "概念": "concept",
    "other": "other",
    "其他": "other",
}

_USER_ALIASES = {"user", "the user", "用户", "本人", "我"}
_AGENT_ALIASES = {"assistant", "the assistant", "agent", "助手"}
_UNRESOLVED_PRONOUNS = {
    "he", "she", "they", "it", "him", "her", "them", "this", "that",
    "他", "她", "它", "他们", "她们", "它们", "这个", "那个",
}


def normalize_entity_name(value: Any) -> str:
    """生成保守、确定性的实体匹配键，不执行模糊实体合并。"""
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    if text in _USER_ALIASES:
        return "__user__"
    if text in _AGENT_ALIASES:
        return "__agent__"
    # 保留中英文、数字和内部空格；标点差异不应产生两个实体。
    return re.sub(r"[^\w\u3400-\u9fff ]+", "", text).strip()


def canonical_entity_name(value: Any) -> str:
    normalized = normalize_entity_name(value)
    if normalized == "__user__":
        return "用户"
    if normalized == "__agent__":
        return "assistant"
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def canonical_entity_type(value: Any) -> str:
    raw = str(value or "other").strip().casefold()
    return _ENTITY_TYPE_ALIASES.get(raw, "other")


def canonical_predicate(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw in _PREDICATE_ALIASES:
        return _PREDICATE_ALIASES[raw]
    normalized = re.sub(r"[^A-Z0-9]+", "_", raw.upper()).strip("_")
    return normalized if normalized in ALLOWED_PREDICATES else None


def sanitize_relation_candidates(
    raw_relations: Iterable[Dict[str, Any]],
    *,
    source_turn_ids: Iterable[str] = (),
    min_confidence: float = 0.8,
) -> List[Dict[str, Any]]:
    """校验 extractor 候选关系并输出后端可直接发布的规范结构。"""
    accepted: List[Dict[str, Any]] = []
    seen = set()
    inherited_turn_ids = [str(value) for value in source_turn_ids if str(value)]

    for raw in raw_relations or []:
        if not isinstance(raw, dict):
            continue
        evidence_type = str(raw.get("evidence_type") or "").strip().lower()
        if evidence_type != "explicit":
            continue
        try:
            confidence = float(raw.get("confidence"))
        except (TypeError, ValueError):
            continue
        if confidence < min_confidence or confidence > 1.0:
            continue

        subject = canonical_entity_name(raw.get("subject"))
        object_name = canonical_entity_name(raw.get("object"))
        subject_norm = normalize_entity_name(subject)
        object_norm = normalize_entity_name(object_name)
        predicate = canonical_predicate(raw.get("predicate"))

        if not subject_norm or not object_norm or not predicate:
            continue
        if subject_norm == object_norm:
            continue
        if subject_norm in _UNRESOLVED_PRONOUNS or object_norm in _UNRESOLVED_PRONOUNS:
            continue
        if len(subject_norm) < 2 or len(object_norm) < 2:
            continue

        # Nested relation provenance is inherited from the already sanitized
        # parent memory.  Never trust a second set of LLM-produced IDs here.
        relation_turn_ids = list(inherited_turn_ids)
        # 发布到 active graph 的关系必须有原始证据。
        if not relation_turn_ids:
            continue

        key = (subject_norm, predicate, object_norm)
        if key in seen:
            continue
        seen.add(key)
        accepted.append({
            "subject": subject,
            "subject_normalized": subject_norm,
            "subject_type": canonical_entity_type(raw.get("subject_type")),
            "predicate": predicate,
            "object": object_name,
            "object_normalized": object_norm,
            "object_type": canonical_entity_type(raw.get("object_type")),
            "confidence": confidence,
            "evidence_type": "explicit",
            "source_turn_ids": list(dict.fromkeys(relation_turn_ids)),
        })
    return accepted
