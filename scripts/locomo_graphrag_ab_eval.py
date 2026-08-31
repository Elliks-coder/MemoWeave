"""Paired LoCoMo A/B evaluation for three-channel recall vs GraphRAG fusion.

The runner ingests one complete conversation once with relation extraction
enabled.  Every selected question is then evaluated twice against the same
memory state:

* baseline: profile + proactive + normal recall, GraphRAG disabled at read time;
* graph: the same three channels with Entity-Fact graph evidence fused into the
  fixed-size normal top-k.

Checkpoints are written after every session and variant so API interruptions can
be resumed with ``--resume-run``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "hy-memory-1.2.21"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import locomo_ultra_eval as base


RESULTS_DIR = PROJECT_ROOT / "results"
RUNTIME_ROOT = PROJECT_ROOT / ".runtime" / "locomo-graphrag-ab"

RELATION_JUDGE_SYSTEM = """You audit an entity-relation memory graph.
For each item, decide whether the quoted original dialogue explicitly supports
the exact subject-predicate-object relation. Do not accept mere co-occurrence,
plausible inference, or a relation supported only by outside knowledge. The
special subject '用户' or 'The user' means the user-role speaker in the quoted
turn. Return JSON only:
{"items":[{"index":0,"supported":true,"reason":"brief reason"}]}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired LoCoMo A/B: three-channel recall vs three-channel + GraphRAG"
    )
    parser.add_argument("--dataset", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=len(base.SELECTED_QA_INDICES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--normal-min-score", type=float, default=0.4)
    parser.add_argument("--intention-min-score", type=float, default=0.4)
    parser.add_argument("--resume-run")
    parser.add_argument("--skip-digest", action="store_true")
    parser.add_argument(
        "--reaudit", action="store_true",
        help="Re-run relation grounding audit for an existing checkpoint",
    )
    parser.add_argument(
        "--audit-limit", type=int, default=100,
        help="Maximum active assertions independently checked by the LLM judge",
    )
    return parser.parse_args()


def configure_environment(run_id: str) -> None:
    base.load_dotenv(base.ENV_FILE, override=False)
    runtime_dir = RUNTIME_ROOT / run_id
    os.environ["MEMORY_MODE"] = "ultra"
    os.environ["MEMORY_VECTOR_STORE"] = "chroma"
    os.environ["MEMORY_COLLECTION_NAME"] = (
        f"locomo_graphrag_ab_{run_id.replace('-', '_')}"
    )
    os.environ["MEMORY_DATA_DIR"] = str(runtime_dir)
    os.environ["MEMORY_PERSIST_DIR"] = str(runtime_dir / "chroma")
    os.environ["MEMORY_GRAPH_PROVIDER"] = "kuzu"
    os.environ["MEMORY_GRAPH_DB_PATH"] = str(runtime_dir / "kuzu")
    os.environ["MEMORY_CACHE_BACKEND"] = "sqlite"
    os.environ["MEMORY_GRAPHRAG_ENABLED"] = "true"
    os.environ["MEMORY_GRAPHRAG_SHADOW_MODE"] = "true"
    os.environ.setdefault("MEMORY_GRAPHRAG_MAX_HOPS", "2")
    os.environ.setdefault("MEMORY_GRAPHRAG_MAX_FACTS", "5")
    os.environ.setdefault("MEMORY_GRAPHRAG_MAX_DEGREE", "25")
    os.environ.setdefault("MEMORY_GRAPHRAG_MIN_CONFIDENCE", "0.8")
    os.environ.setdefault("MEMORY_EMBEDDING_DIMS", "1024")

    required = (
        "MEMORY_LLM_MODEL",
        "MEMORY_LLM_API_KEY",
        "MEMORY_LLM_BASE_URL",
        "MEMORY_EMBEDDER_MODEL",
        "MEMORY_EMBEDDER_API_KEY",
        "MEMORY_EMBEDDER_BASE_URL",
    )
    missing = []
    for name in required:
        value = os.getenv(name, "").strip()
        if not value or "REPLACE_WITH" in value or "YOUR_WORKSPACE_ID" in value:
            missing.append(name)
    if missing:
        raise SystemExit("Missing model configuration: " + ", ".join(missing))


def save(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _turn_text_map(sample: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, turns in sample["conversation"].items():
        if not name.startswith("session_") or name.endswith("_date_time"):
            continue
        for turn in turns:
            dia_id = str(turn.get("dia_id") or "")
            if dia_id:
                text = str(turn.get("text") or "").strip()
                caption = turn.get("blip_caption")
                if caption:
                    if isinstance(caption, list):
                        caption = "; ".join(str(item) for item in caption)
                    text += f" [Shared image description: {caption}]"
                result[dia_id] = (
                    f"{turn.get('speaker', '')}: {text}"
                ).strip()
    return result


async def _read_assertions(graph_store: Any, tenant_key: str) -> list[dict[str, Any]]:
    rows = await graph_store._run(
        """
        MATCH (a:Assertion)-[:ASSERTION_SUBJECT]->(s:Entity)
        MATCH (a)-[:ASSERTION_OBJECT]->(o:Entity)
        MATCH (a)-[:ASSERTION_SUPPORTED_BY]->(f:VdbRef)
        WHERE a.isolation_key = $ik
        RETURN a.assertion_id, a.status, a.confidence, a.predicate,
               a.source_turn_ids, a.valid_from, a.valid_until,
               s.canonical_name, s.normalized_name,
               o.canonical_name, o.normalized_name, f.node_id;
        """,
        {"ik": tenant_key},
    )
    return [
        {
            "assertion_id": row[0],
            "status": row[1],
            "confidence": float(row[2] or 0.0),
            "predicate": row[3],
            "source_turn_ids": _parse_turn_ids(row[4]),
            "valid_from": str(row[5]) if row[5] is not None else None,
            "valid_until": str(row[6]) if row[6] is not None else None,
            "subject": row[7],
            "subject_normalized": row[8],
            "object": row[9],
            "object_normalized": row[10],
            "fact_id": row[11],
        }
        for row in (rows or [])
    ]


def _parse_turn_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
        return [str(item) for item in parsed if str(item)] if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _entity_mentioned(normalized: str, source: str) -> bool:
    if normalized in {"__user__", "__agent__"}:
        return True
    compact_source = " ".join(source.casefold().split())
    return str(normalized or "").casefold() in compact_source


async def audit_graph(
    client: Any,
    *,
    tenant_key: str,
    turn_texts: dict[str, str],
    user_speaker: str,
    audit_limit: int,
) -> dict[str, Any]:
    graph_store = client._graph_store
    stats = await client._loop_thread.run_async(graph_store.get_stats())
    assertions = await client._loop_thread.run_async(
        _read_assertions(graph_store, tenant_key)
    )
    active = [item for item in assertions if item["status"] == "active"]

    for item in assertions:
        source = "\n".join(
            turn_texts.get(turn_id, "[missing turn]")
            for turn_id in item["source_turn_ids"]
        )
        item["source_text"] = source
        item["provenance_resolved"] = bool(item["source_turn_ids"]) and all(
            turn_id in turn_texts for turn_id in item["source_turn_ids"]
        )
        item["entity_mentions_present"] = (
            _entity_mentioned(item["subject_normalized"], source)
            and _entity_mentioned(item["object_normalized"], source)
        )

    semantic_groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    functional_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    functional_predicates = {
        "LIVES_IN", "FROM", "WORKS_AT", "STUDIES_AT", "PARTNER_OF",
        "PARENT_OF", "CHILD_OF", "OWNS", "HAS_PET",
    }
    for item in active:
        semantic_groups[(
            item["subject_normalized"], item["predicate"], item["object_normalized"]
        )].append(item["assertion_id"])
        if item["predicate"] in functional_predicates:
            functional_groups[(
                item["subject_normalized"], item["predicate"]
            )].add(item["object_normalized"])

    duplicate_groups = [
        {"relation": list(key), "assertion_ids": values}
        for key, values in semantic_groups.items() if len(values) > 1
    ]
    competing_groups = [
        {"subject_predicate": list(key), "objects": sorted(values)}
        for key, values in functional_groups.items() if len(values) > 1
    ]

    judged: list[dict[str, Any]] = []
    judge_errors: list[str] = []
    to_judge = active[: max(0, audit_limit)]
    for offset in range(0, len(to_judge), 6):
        batch = to_judge[offset: offset + 6]
        payload = [
            {
                "index": offset + index,
                "relation": (
                    f"{item['subject']} --{item['predicate']}--> {item['object']}"
                ),
                "source": item["source_text"],
            }
            for index, item in enumerate(batch)
        ]
        audit_system = (
            RELATION_JUDGE_SYSTEM
            + f"\nFor this evaluation, '用户'/'The user' is exactly the speaker "
              f"named {user_speaker}."
        )
        raw = await base.llm_text(
            client,
            audit_system,
            json.dumps(payload, ensure_ascii=False),
            json_mode=True,
        )
        try:
            parsed = base.parse_json_object(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            parsed = {}
            judge_errors.append(
                f"batch {offset}-{offset + len(batch) - 1}: "
                f"{type(error).__name__}: {error}"
            )
        by_index = {
            int(entry.get("index")): entry
            for entry in parsed.get("items", [])
            if isinstance(entry, dict) and str(entry.get("index", "")).isdigit()
        }
        for index, assertion in enumerate(batch, start=offset):
            decision = by_index.get(index, {})
            # A malformed/truncated batch response must not abort a paid run.
            # Retry only the missing item with a much smaller response budget.
            if "supported" not in decision:
                single_payload = [{
                    "index": index,
                    "relation": (
                        f"{assertion['subject']} --{assertion['predicate']}--> "
                        f"{assertion['object']}"
                    ),
                    "source": assertion["source_text"],
                }]
                try:
                    single_raw = await base.llm_text(
                        client,
                        audit_system,
                        json.dumps(single_payload, ensure_ascii=False),
                        json_mode=True,
                    )
                    single_parsed = base.parse_json_object(single_raw)
                    single_items = single_parsed.get("items", [])
                    if single_items and isinstance(single_items[0], dict):
                        decision = single_items[0]
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    judge_errors.append(
                        f"assertion {assertion['assertion_id']}: "
                        f"{type(error).__name__}: {error}"
                    )
            supported = decision.get("supported")
            if not isinstance(supported, bool):
                supported = None
            judged.append({
                "assertion_id": assertion["assertion_id"],
                "relation": (
                    f"{assertion['subject']} --{assertion['predicate']}--> "
                    f"{assertion['object']}"
                ),
                "source_turn_ids": assertion["source_turn_ids"],
                "supported": supported,
                "reason": str(decision.get("reason") or "missing judge decision"),
            })

    decided = [item for item in judged if isinstance(item["supported"], bool)]

    return {
        "stats": base.compact(stats),
        "assertion_count_for_tenant": len(assertions),
        "active_count": len(active),
        "complete_edge_coverage": (
            round(len(assertions) / max(1, int(stats.get("assertion_nodes") or 0)), 4)
        ),
        "provenance_resolved_rate": round(
            sum(item["provenance_resolved"] for item in assertions)
            / max(1, len(assertions)),
            4,
        ),
        "entity_mention_rate": round(
            sum(item["entity_mentions_present"] for item in assertions)
            / max(1, len(assertions)),
            4,
        ),
        "duplicate_active_relation_groups": duplicate_groups,
        "competing_active_functional_groups": competing_groups,
        "llm_grounding_audit": {
            "n": len(judged),
            "decided": len(decided),
            "undecided": len(judged) - len(decided),
            "supported": sum(item["supported"] is True for item in decided),
            "precision": (
                round(
                    sum(item["supported"] is True for item in decided)
                    / len(decided), 4,
                )
                if decided else None
            ),
            "failures": [item for item in decided if item["supported"] is False],
            "errors": judge_errors,
        },
        "assertions": assertions,
    }


def _gold_turns(qa: dict[str, Any]) -> set[str]:
    return {str(value) for value in qa.get("evidence", []) if value}


async def evaluate_variant(
    client: Any,
    *,
    variant: str,
    qa: dict[str, Any],
    user_id: str,
    agent_id: str,
    benchmark_as_of: datetime,
    args: argparse.Namespace,
) -> dict[str, Any]:
    is_graph = variant == "graph"
    client._config.graph_store.graphrag_enabled = is_graph
    client._config.graph_store.graphrag_shadow_mode = not is_graph

    started = time.perf_counter()
    raw_search = base.compact(client.search(
        qa["question"],
        user_ids=[user_id],
        agent_ids=[agent_id],
        limit=args.top_k,
        min_score=args.normal_min_score,
        intention_min_score=args.intention_min_score,
        as_of=benchmark_as_of,
    ))
    search_ms = round((time.perf_counter() - started) * 1000, 2)
    memories = base.flatten_memories(raw_search)
    candidate = await base.llm_text(
        client,
        base.ANSWER_SYSTEM_PROMPT,
        (
            f"Retrieved memories:\n{base.memory_text(memories) or '(none)'}"
            f"\n\nQuestion: {qa['question']}"
        ),
    )
    category = int(qa["category"])
    reference = str(qa.get("answer") or "")
    judge_raw = await base.llm_text(
        client,
        base.JUDGE_SYSTEM_PROMPT,
        "\n".join([
            f"Question: {qa['question']}",
            f"Reference answer: {reference if reference else '[UNANSWERABLE]'}",
            f"Candidate answer: {candidate}",
        ]),
        json_mode=True,
    )
    judge = base.parse_json_object(judge_raw)

    gold = _gold_turns(qa)
    ids_at = {
        k: base.retrieved_evidence_ids(memories, k) for k in (1, 5, 10)
    }
    graph_memories = [
        item for item in memories
        if item.get("retrieval_source") == "entity_fact_graph"
    ]
    channels = Counter(item.get("bucket", "unknown") for item in memories)
    answerable = category != 5
    return {
        "candidate_answer": candidate,
        "judge": judge,
        "search_elapsed_ms": search_ms,
        "retrieved_count": len(memories),
        "channel_counts": dict(channels),
        "retrieved_memory_ids": [item.get("memory_id") for item in memories],
        "graph_returned_count": len(graph_memories),
        "graph_returned_memory_ids": [
            item.get("memory_id") for item in graph_memories
        ],
        "graph_summary": raw_search.get("graphrag"),
        "exact_hit_at_1": bool(gold.intersection(ids_at[1])) if answerable else None,
        "exact_hit_at_5": bool(gold.intersection(ids_at[5])) if answerable else None,
        "exact_hit_at_10": bool(gold.intersection(ids_at[10])) if answerable else None,
        "recall_at_1": base.evidence_recall(gold, ids_at[1]) if answerable else None,
        "recall_at_5": base.evidence_recall(gold, ids_at[5]) if answerable else None,
        "recall_at_10": base.evidence_recall(gold, ids_at[10]) if answerable else None,
        "retrieved_evidence_ids_at_10": sorted(ids_at[10]),
        "retrieved_memories": memories,
    }


def compute_metrics(report: dict[str, Any]) -> dict[str, Any]:
    questions = report.get("questions", [])

    def aggregate(items: list[dict[str, Any]], variant: str) -> dict[str, Any]:
        values = [item["variants"][variant] for item in items if variant in item["variants"]]
        answerable = [
            item["variants"][variant] for item in items
            if item["category"] != 5 and variant in item["variants"]
        ]
        if not values:
            return {"n": 0}
        result: dict[str, Any] = {
            "n": len(values),
            "accuracy": round(
                sum(bool(value.get("judge", {}).get("correct")) for value in values)
                / len(values), 4
            ),
            "mean_search_ms": round(statistics.mean(
                float(value.get("search_elapsed_ms") or 0) for value in values
            ), 2),
            "mean_retrieved_count": round(statistics.mean(
                int(value.get("retrieved_count") or 0) for value in values
            ), 2),
            "queries_with_graph_returned": sum(
                int(value.get("graph_returned_count") or 0) > 0 for value in values
            ),
            "mean_graph_returned": round(statistics.mean(
                int(value.get("graph_returned_count") or 0) for value in values
            ), 2),
        }
        for k in (1, 5, 10):
            hits = [bool(value.get(f"exact_hit_at_{k}")) for value in answerable]
            recalls = [
                float(value[f"recall_at_{k}"])
                for value in answerable if value.get(f"recall_at_{k}") is not None
            ]
            result[f"exact_hit_at_{k}"] = (
                round(sum(hits) / len(hits), 4) if hits else None
            )
            result[f"recall_at_{k}"] = (
                round(statistics.mean(recalls), 4) if recalls else None
            )
        return result

    variants: dict[str, Any] = {}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in questions:
        by_category[item["category_name"]].append(item)
    for variant in ("baseline", "graph"):
        variants[variant] = {
            "overall": aggregate(questions, variant),
            "by_category": {
                category: aggregate(items, variant)
                for category, items in by_category.items()
            },
        }

    paired = [
        item for item in questions
        if {"baseline", "graph"}.issubset(item.get("variants", {}))
    ]
    improved = []
    regressed = []
    retrieval_changed = []
    for item in paired:
        before = bool(item["variants"]["baseline"].get("judge", {}).get("correct"))
        after = bool(item["variants"]["graph"].get("judge", {}).get("correct"))
        if after and not before:
            improved.append(item["qa_index"])
        elif before and not after:
            regressed.append(item["qa_index"])
        if (
            item["variants"]["baseline"].get("retrieved_memory_ids")
            != item["variants"]["graph"].get("retrieved_memory_ids")
        ):
            retrieval_changed.append(item["qa_index"])

    return {
        "variants": variants,
        "paired": {
            "n": len(paired),
            "improved_qa_indices": improved,
            "regressed_qa_indices": regressed,
            "retrieval_changed_qa_indices": retrieval_changed,
            "unchanged_answer_outcome_count": len(paired) - len(improved) - len(regressed),
        },
    }


async def run(args: argparse.Namespace) -> Path:
    run_id = args.resume_run or datetime.now().strftime("%Y%m%d-%H%M%S")
    configure_environment(run_id)
    sample = base.load_sample(args.dataset.resolve(), args.sample_index)
    conversation = sample["conversation"]
    selected = base.SELECTED_QA_INDICES[: max(
        0, min(args.limit, len(base.SELECTED_QA_INDICES))
    )]
    result_path = RESULTS_DIR / f"locomo-graphrag-ab-{run_id}.json"
    user_id = f"locomo-graphrag-{sample['sample_id']}-{run_id}"
    agent_id = "locomo-graphrag-ab"

    if args.resume_run and result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        user_id = report["user_id"]
    else:
        report = {
            "run_id": run_id,
            "benchmark": "LoCoMo GraphRAG paired pilot",
            "sample_id": sample["sample_id"],
            "sample_index": args.sample_index,
            "user_id": user_id,
            "agent_id": agent_id,
            "models": {
                "llm": os.getenv("MEMORY_LLM_MODEL"),
                "embedder": os.getenv("MEMORY_EMBEDDER_MODEL"),
            },
            "selection": {
                "qa_indices": selected,
                "category_distribution": dict(Counter(
                    base.CATEGORY_NAMES[int(sample["qa"][index]["category"])]
                    for index in selected
                )),
            },
            "retrieval_policy": {
                "baseline": "profile + proactive + normal",
                "graph": "same three channels + Entity-Fact graph fusion",
                "fixed_normal_top_k": args.top_k,
                "normal_min_score": args.normal_min_score,
                "intention_min_score": args.intention_min_score,
                "graph_max_hops": int(os.getenv("MEMORY_GRAPHRAG_MAX_HOPS", "2")),
                "graph_max_facts": int(os.getenv("MEMORY_GRAPHRAG_MAX_FACTS", "5")),
                "graph_min_confidence": float(
                    os.getenv("MEMORY_GRAPHRAG_MIN_CONFIDENCE", "0.8")
                ),
            },
            "writes": [],
            "digest": None,
            "graph_audit": None,
            "questions": [],
        }
        save(report, result_path)

    sessions = sorted(
        (name for name in conversation if name.startswith("session_")
         and not name.endswith("_date_time")),
        key=base.session_number,
    )
    benchmark_as_of = max(
        base.parse_locomo_datetime(conversation[f"{name}_date_time"])
        for name in sessions
    )
    report["retrieval_policy"]["as_of"] = benchmark_as_of.isoformat()
    save(report, result_path)

    client = base.build_client()
    try:
        # Relation extraction and graph publishing remain enabled for ingestion.
        client._config.graph_store.graphrag_enabled = True
        client._config.graph_store.graphrag_shadow_mode = True
        completed_sessions = {
            item["session_id"] for item in report.get("writes", [])
            if item.get("success")
        }
        for name in sessions:
            if name in completed_sessions:
                continue
            messages = base.convert_session(
                conversation[name], conversation["speaker_a"]
            )
            started = time.perf_counter()
            result = base.compact(client.add(
                messages,
                user_id=user_id,
                agent_id=agent_id,
                session_id=name,
                memory_at=base.parse_locomo_datetime(
                    conversation[f"{name}_date_time"]
                ),
            ))
            entry = {
                "session_id": name,
                "turn_count": len(messages),
                "success": not (
                    isinstance(result, dict) and result.get("success") is False
                ),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "result": result,
            }
            report.setdefault("writes", []).append(entry)
            save(report, result_path)
            graph_counts = (
                result.get("ops_summary", {}) if isinstance(result, dict) else {}
            )
            print(
                f"write {name}: success={entry['success']} "
                f"graph_published={graph_counts.get('graph_published')} "
                f"elapsed={entry['elapsed_ms']}ms",
                flush=True,
            )
            if not entry["success"]:
                raise RuntimeError(f"write failed for {name}: {result}")

        if report.get("digest") is None and not args.skip_digest:
            started = time.perf_counter()
            result = base.compact(client.digest(user_id=user_id, agent_id=agent_id))
            report["digest"] = {
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "result": result,
            }
            save(report, result_path)

        if report.get("graph_audit") is None or args.reaudit:
            report["graph_audit"] = await audit_graph(
                client,
                tenant_key=f"{user_id}::{agent_id}",
                turn_texts=_turn_text_map(sample),
                user_speaker=conversation["speaker_a"],
                audit_limit=args.audit_limit,
            )
            save(report, result_path)
            audit = report["graph_audit"]
            print(
                "graph audit: "
                f"active={audit['active_count']} "
                f"grounding_precision={audit['llm_grounding_audit']['precision']} "
                f"duplicates={len(audit['duplicate_active_relation_groups'])}",
                flush=True,
            )

        questions_by_index = {
            int(item["qa_index"]): item for item in report.get("questions", [])
        }
        for question_position, qa_index in enumerate(selected):
            qa = sample["qa"][qa_index]
            entry = questions_by_index.get(qa_index)
            if entry is None:
                category = int(qa["category"])
                entry = {
                    "qa_index": qa_index,
                    "category": category,
                    "category_name": base.CATEGORY_NAMES[category],
                    "question": qa["question"],
                    "reference_answer": str(qa.get("answer") or ""),
                    "evidence_dialog_ids": qa.get("evidence", []),
                    "variants": {},
                }
                report.setdefault("questions", []).append(entry)
                questions_by_index[qa_index] = entry

            # Alternate execution order so embedding/vector cache warm-up does
            # not systematically make the second variant look faster.
            variant_order = (
                ("baseline", "graph")
                if question_position % 2 == 0
                else ("graph", "baseline")
            )
            for variant in variant_order:
                if variant in entry["variants"]:
                    continue
                value = await evaluate_variant(
                    client,
                    variant=variant,
                    qa=qa,
                    user_id=user_id,
                    agent_id=agent_id,
                    benchmark_as_of=benchmark_as_of,
                    args=args,
                )
                entry["variants"][variant] = value
                report["metrics"] = compute_metrics(report)
                save(report, result_path)
                print(
                    f"qa {qa_index} {entry['category_name']} {variant}: "
                    f"correct={bool(value['judge'].get('correct'))} "
                    f"recall@5={value['recall_at_5']} "
                    f"graph_returned={value['graph_returned_count']} "
                    f"search={value['search_elapsed_ms']}ms",
                    flush=True,
                )

        report["metrics"] = compute_metrics(report)
        report["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
        save(report, result_path)
        return result_path
    finally:
        client.close()
        await asyncio.sleep(0)


def main() -> None:
    args = parse_args()
    path = asyncio.run(run(args))
    print(f"LoCoMo GraphRAG A/B complete: {path}")


if __name__ == "__main__":
    main()
