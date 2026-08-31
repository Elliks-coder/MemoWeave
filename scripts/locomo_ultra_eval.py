"""Run a reproducible 25-question LoCoMo evaluation against Hy-Memory Ultra.

The runner ingests one complete LoCoMo conversation, evaluates retrieval
against the benchmark's annotated evidence, and uses the configured LLM to
answer questions from retrieved memories. It writes checkpoints after every
session and question so an interrupted API run can be resumed safely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_DATASET = PROJECT_ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNTIME_ROOT = PROJECT_ROOT / ".runtime" / "locomo-ultra"

# Stratified across all five official categories. The selected questions span
# early and late sessions, single- and multi-evidence items, temporal inference,
# open-domain reasoning, and adversarial speaker swaps.
SELECTED_QA_INDICES = [
    32, 39, 47, 56, 65,  # cat 1: multi-hop
    0, 28, 41, 57, 79,  # cat 2: temporal
    2, 14, 50, 59, 81,  # cat 3: open-domain reasoning
    85, 91, 116, 134, 139,  # cat 4: single-hop
    153, 164, 171, 185, 191,  # cat 5: adversarial/unanswerable
]

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a long-term memory QA benchmark.
Compare the candidate with the reference answer. Ignore wording differences, but require
all important facts from the reference and reject material contradictions. For an
unanswerable question, the candidate is correct only if it clearly says the information
is unknown or unavailable. Return JSON only:
{"correct": true, "reason": "brief reason"}"""

ANSWER_PROMPT_VERSION = "balanced-evidence-v2"

ANSWER_SYSTEM_PROMPT = """Answer the question from the retrieved memories. Treat them as
evidence to reason over, not merely sentences to quote.

Use this balanced evidence policy:
1. Give a direct answer when a memory states it explicitly.
2. Combine multiple relevant memories when the answer follows from them together.
3. Make a reasonable, best-supported inference when the question asks about likelihood,
   disposition, implications, causes, or a counterfactual. The answer need not appear
   verbatim in one memory. You may use ordinary commonsense as the connecting rule, but
   never as a source of missing person-specific facts.
4. Calibrate the wording: state direct facts plainly; mark inferred conclusions with
   "likely", "probably", or an equally brief qualifier.
5. Abstain only when there is no relevant evidence, when materially different answers are
   equally supported, or when the question requires an exact name, date, quantity, or list
   that the memories do not provide. Do not abstain merely because synthesis is required.

Ground every person-specific claim in memories about that same person. Do not transfer a
fact from one person to another. If memories conflict, prefer the latest temporally valid
state and avoid silently combining incompatible states. Never invent events or precise
details. Preserve supported names, dates, quantities, and list items. Give the shortest
answer that fully addresses the question; include a brief reason only when it clarifies an
inference. If abstention is required, use exactly this sentence: I don't know based on the
available memories."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hy-Memory Ultra on 25 LoCoMo questions")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=len(SELECTED_QA_INDICES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--normal-min-score", "--min-score", dest="normal_min_score",
        type=float, default=0.4,
        help="Normal retrieval score threshold (default: 0.4)",
    )
    parser.add_argument(
        "--intention-min-score", type=float, default=0.4,
        help="Proactive/L7 retrieval score threshold (default: 0.4)",
    )
    parser.add_argument("--resume-run", help="Resume a previous run id")
    parser.add_argument("--skip-digest", action="store_true")
    return parser.parse_args()


def configure_environment(run_id: str) -> None:
    load_dotenv(ENV_FILE, override=False)
    runtime_dir = RUNTIME_ROOT / run_id
    os.environ["MEMORY_MODE"] = "ultra"
    os.environ["MEMORY_VECTOR_STORE"] = "chroma"
    os.environ["MEMORY_COLLECTION_NAME"] = f"locomo_ultra_{run_id.replace('-', '_')}"
    os.environ["MEMORY_DATA_DIR"] = str(runtime_dir)
    os.environ["MEMORY_PERSIST_DIR"] = str(runtime_dir / "chroma")
    os.environ["MEMORY_GRAPH_PROVIDER"] = "kuzu"
    os.environ["MEMORY_GRAPH_DB_PATH"] = str(runtime_dir / "kuzu")
    os.environ["MEMORY_CACHE_BACKEND"] = "sqlite"
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


def compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return compact(value.model_dump())
    return str(value)


def build_client():
    from hy_memory import HyMemoryClient
    from hy_memory.agent.llm_provider import LLMProvider
    from hy_memory.config import MemoryConfig

    config = MemoryConfig()
    if (config.llm.model or "").lower().startswith("deepseek-v4"):
        config.llm.extra_body = {"thinking": {"type": "disabled"}}
    client = HyMemoryClient(config=config, mode="ultra")
    client._benchmark_llm = LLMProvider(config)
    return client


def load_sample(dataset_path: Path, sample_index: int) -> dict[str, Any]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not 0 <= sample_index < len(data):
        raise SystemExit(f"sample-index must be between 0 and {len(data) - 1}")
    return data[sample_index]


def session_number(name: str) -> int:
    return int(name.split("_")[1])


def parse_locomo_datetime(value: str) -> datetime:
    # Example: "1:56 pm on 8 May, 2023"
    normalized = re.sub(r"\s+on\s+", " ", value.strip(), flags=re.IGNORECASE)
    return datetime.strptime(normalized, "%I:%M %p %d %B, %Y").replace(tzinfo=timezone.utc)


def convert_session(turns: list[dict[str, Any]], speaker_a: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        role = "user" if turn.get("speaker") == speaker_a else "assistant"
        text = str(turn.get("text", "")).strip()
        caption = turn.get("blip_caption")
        if caption:
            if isinstance(caption, list):
                caption = "; ".join(str(item) for item in caption)
            text += f" [Shared image description: {caption}]"
        if text:
            messages.append({
                "role": role,
                "content": text,
                "turn_id": str(turn.get("dia_id") or ""),
            })
    return messages


def evidence_map(sample: dict[str, Any]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    conversation = sample["conversation"]
    for name, turns in conversation.items():
        if not re.fullmatch(r"session_\d+", name):
            continue
        number = session_number(name)
        for turn in turns:
            dia_id = turn.get("dia_id")
            if dia_id:
                result[number].add(str(dia_id))
    return result


def flatten_memories(search_result: dict[str, Any]) -> list[dict[str, Any]]:
    buckets = search_result.get("memories", {}) if isinstance(search_result, dict) else {}
    memories: list[dict[str, Any]] = []
    for bucket in ("profile", "proactive", "normal"):
        for item in buckets.get(bucket, []) or []:
            record = dict(item)
            record["bucket"] = bucket
            memories.append(record)
    memories.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
    return memories


def retrieved_evidence_ids(
    memories: list[dict[str, Any]],
    k: int,
) -> set[str]:
    """Union exact source-turn IDs carried by the first k ranked memories."""
    ids: set[str] = set()
    for item in memories[:k]:
        chain = item.get("evidence_chain") or []
        if isinstance(chain, (str, int)):
            chain = [chain]
        if isinstance(chain, list):
            ids.update(str(value) for value in chain if str(value))
    return ids


def evidence_recall(gold_ids: set[str], retrieved_ids: set[str]) -> float | None:
    """Strict evidence recall: matched annotated dia_ids / all annotated dia_ids."""
    if not gold_ids:
        return None
    return len(gold_ids.intersection(retrieved_ids)) / len(gold_ids)


def memory_text(memories: list[dict[str, Any]]) -> str:
    def _date(value: Any) -> str:
        if not isinstance(value, (int, float)):
            return ""
        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d")

    lines = []
    for index, item in enumerate(memories, start=1):
        metadata = []
        layer = str(item.get("layer") or "").strip()
        if layer:
            metadata.append(f"layer={layer}")
        observed = _date(item.get("observed_at") or item.get("memory_at"))
        if observed:
            metadata.append(f"observed={observed}")
        relation = str(item.get("temporal_relation") or "").strip()
        if relation and relation not in {"unknown", "atemporal"}:
            metadata.append(f"time_relation={relation}")
        event_start = _date(item.get("event_start"))
        event_end = _date(item.get("event_end"))
        if event_start:
            event_range = event_start
            if event_end and event_end != event_start:
                event_range += f"..{event_end}"
            metadata.append(f"event={event_range}")
        header = " ".join(metadata)
        content = str(item.get("content") or "").strip()
        lines.append(
            f"[{index}] {header} | {content}" if header else f"[{index}] {content}"
        )
    return "\n".join(lines)


async def llm_text(client: Any, system: str, user: str, *, json_mode: bool = False) -> str:
    kwargs: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = await client._benchmark_llm.chat(**kwargs)
    return str(response.content).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(match.group(0)) if match else {"correct": False, "reason": "invalid judge output"}


def save_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_metrics(report: dict[str, Any]) -> dict[str, Any]:
    questions = report.get("questions", [])
    if not questions:
        return {}

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in questions:
        by_category[item["category_name"]].append(item)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        answerable = [item for item in items if item["category"] != 5]
        hit1 = [bool(item.get("evidence_hit_at_1")) for item in items if item["category"] != 5]
        hit5 = [bool(item.get("evidence_hit_at_5")) for item in items if item["category"] != 5]
        hit10 = [bool(item.get("evidence_hit_at_10")) for item in items if item["category"] != 5]
        exact_hit1 = [
            bool(item["exact_evidence_hit_at_1"])
            for item in answerable if item.get("exact_evidence_hit_at_1") is not None
        ]
        exact_hit5 = [
            bool(item["exact_evidence_hit_at_5"])
            for item in answerable if item.get("exact_evidence_hit_at_5") is not None
        ]
        exact_hit10 = [
            bool(item["exact_evidence_hit_at_10"])
            for item in answerable if item.get("exact_evidence_hit_at_10") is not None
        ]
        recall1 = [
            float(item["evidence_recall_at_1"])
            for item in answerable if item.get("evidence_recall_at_1") is not None
        ]
        recall5 = [
            float(item["evidence_recall_at_5"])
            for item in answerable if item.get("evidence_recall_at_5") is not None
        ]
        recall10 = [
            float(item["evidence_recall_at_10"])
            for item in answerable if item.get("evidence_recall_at_10") is not None
        ]
        provenance_coverage = [
            float(item["l2_l7_provenance_coverage"])
            for item in items if item.get("l2_l7_provenance_coverage") is not None
        ]
        judged = [bool(item.get("judge", {}).get("correct")) for item in items]
        latencies = [float(item.get("search_elapsed_ms") or 0) for item in items]
        return {
            "n": len(items),
            "answer_accuracy": round(sum(judged) / len(judged), 4) if judged else None,
            "evidence_hit_at_1": round(sum(hit1) / len(hit1), 4) if hit1 else None,
            "evidence_hit_at_5": round(sum(hit5) / len(hit5), 4) if hit5 else None,
            "evidence_hit_at_10": round(sum(hit10) / len(hit10), 4) if hit10 else None,
            "exact_evidence_hit_at_1": round(sum(exact_hit1) / len(exact_hit1), 4) if exact_hit1 else None,
            "exact_evidence_hit_at_5": round(sum(exact_hit5) / len(exact_hit5), 4) if exact_hit5 else None,
            "exact_evidence_hit_at_10": round(sum(exact_hit10) / len(exact_hit10), 4) if exact_hit10 else None,
            "evidence_recall_at_1": round(statistics.mean(recall1), 4) if recall1 else None,
            "evidence_recall_at_5": round(statistics.mean(recall5), 4) if recall5 else None,
            "evidence_recall_at_10": round(statistics.mean(recall10), 4) if recall10 else None,
            "mean_l2_l7_provenance_coverage": (
                round(statistics.mean(provenance_coverage), 4)
                if provenance_coverage else None
            ),
            "mean_search_ms": round(statistics.mean(latencies), 2) if latencies else None,
        }

    return {
        "overall": aggregate(questions),
        "by_category": {name: aggregate(items) for name, items in by_category.items()},
        "write_success_rate": round(
            sum(bool(item.get("success")) for item in report.get("writes", []))
            / max(1, len(report.get("writes", []))),
            4,
        ),
        "memory_layers": dict(Counter(
            item.get("layer", "unknown")
            for item in report.get("memory_inventory", {}).get("vdb", {}).get("memories", [])
        )),
    }


async def run(args: argparse.Namespace) -> Path:
    run_id = args.resume_run or datetime.now().strftime("%Y%m%d-%H%M%S")
    configure_environment(run_id)
    sample = load_sample(args.dataset.resolve(), args.sample_index)
    conversation = sample["conversation"]
    selected_indices = SELECTED_QA_INDICES[: max(0, min(args.limit, len(SELECTED_QA_INDICES)))]
    user_id = f"locomo-{sample['sample_id']}-{run_id}"
    agent_id = "locomo-ultra-eval"
    result_path = RESULTS_DIR / f"locomo-ultra-{run_id}.json"

    if args.resume_run and result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        user_id = report["user_id"]
    else:
        report = {
            "run_id": run_id,
            "benchmark": "LoCoMo (ACL 2024)",
            "source": "https://github.com/snap-research/locomo",
            "dataset_file": str(args.dataset.resolve()),
            "sample_id": sample["sample_id"],
            "speakers": [conversation["speaker_a"], conversation["speaker_b"]],
            "user_id": user_id,
            "agent_id": agent_id,
            "mode": "ultra",
            "models": {
                "llm": os.getenv("MEMORY_LLM_MODEL"),
                "embedder": os.getenv("MEMORY_EMBEDDER_MODEL"),
                "embedding_dims": os.getenv("MEMORY_EMBEDDING_DIMS"),
            },
            "answer_policy": {
                "version": ANSWER_PROMPT_VERSION,
                "system_prompt": ANSWER_SYSTEM_PROMPT,
            },
            "selection": {
                "qa_indices": selected_indices,
                "category_distribution": dict(Counter(
                    CATEGORY_NAMES[int(sample["qa"][index]["category"])] for index in selected_indices
                )),
            },
            "writes": [],
            "digest": None,
            "questions": [],
        }
        save_report(report, result_path)

    # Refresh selection metadata when resuming with an updated, role-compatible
    # selection. Completed question records (if any) remain the source of truth.
    report["selection"] = {
        "qa_indices": selected_indices,
        "category_distribution": dict(Counter(
            CATEGORY_NAMES[int(sample["qa"][index]["category"])] for index in selected_indices
        )),
        "subject": conversation["speaker_a"],
        "role_note": "Questions target the speaker mapped to Hy-Memory's user role; category 5 tests speaker-swap abstention.",
    }
    # A resumed run may contain answers produced by an older prompt.  Keep the
    # per-question version authoritative and label this as the policy used for
    # newly generated answers instead of rewriting historical provenance.
    report["answer_policy_current"] = {
        "version": ANSWER_PROMPT_VERSION,
        "system_prompt": ANSWER_SYSTEM_PROMPT,
    }
    report["questions"] = [
        item for item in report.get("questions", []) if item.get("qa_index") in selected_indices
    ]
    report["metrics"] = compute_metrics(report)
    save_report(report, result_path)

    sessions = sorted(
        (name for name in conversation if re.fullmatch(r"session_\d+", name)),
        key=session_number,
    )
    benchmark_as_of = max(
        parse_locomo_datetime(conversation[f"{name}_date_time"])
        for name in sessions
    )
    report["retrieval_policy"] = {
        "top_k": args.top_k,
        "normal_min_score": args.normal_min_score,
        "intention_min_score": args.intention_min_score,
        "profile_min_score": 0.4,
        "as_of": benchmark_as_of.isoformat(),
        "as_of_source": "latest conversation session",
    }
    save_report(report, result_path)
    existing_sessions = {item["session_id"] for item in report.get("writes", []) if item.get("success")}
    client = build_client()
    try:
        for name in sessions:
            if name in existing_sessions:
                continue
            date_text = conversation[f"{name}_date_time"]
            messages = convert_session(conversation[name], conversation["speaker_a"])
            started = time.perf_counter()
            result = compact(client.add(
                messages,
                user_id=user_id,
                agent_id=agent_id,
                session_id=name,
                memory_at=parse_locomo_datetime(date_text),
            ))
            entry = {
                "session_id": name,
                "session_date": date_text,
                "turn_count": len(messages),
                "success": not (isinstance(result, dict) and result.get("success") is False),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "result": result,
            }
            report.setdefault("writes", []).append(entry)
            save_report(report, result_path)
            print(f"write {name}: success={entry['success']} elapsed={entry['elapsed_ms']}ms", flush=True)
            if not entry["success"]:
                raise RuntimeError(f"write failed for {name}: {result}")

        if report.get("digest") is None and not args.skip_digest:
            started = time.perf_counter()
            digest_result = compact(client.digest(user_id=user_id, agent_id=agent_id))
            report["digest"] = {
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "result": digest_result,
            }
            save_report(report, result_path)

        dia_to_session: dict[str, str] = {}
        for number, ids in evidence_map(sample).items():
            for dia_id in ids:
                dia_to_session[dia_id] = f"session_{number}"

        existing_questions = {item["qa_index"] for item in report.get("questions", [])}
        for qa_index in selected_indices:
            if qa_index in existing_questions:
                continue
            qa = sample["qa"][qa_index]
            category = int(qa["category"])
            search_started = time.perf_counter()
            raw_search = compact(client.search(
                qa["question"],
                user_ids=[user_id],
                agent_ids=[agent_id],
                limit=args.top_k,
                min_score=args.normal_min_score,
                intention_min_score=args.intention_min_score,
                as_of=benchmark_as_of,
            ))
            search_ms = round((time.perf_counter() - search_started) * 1000, 2)
            memories = flatten_memories(raw_search)
            context = memory_text(memories)
            candidate = await llm_text(
                client,
                ANSWER_SYSTEM_PROMPT,
                f"Retrieved memories:\n{context or '(none)'}\n\nQuestion: {qa['question']}",
            )
            reference = str(qa.get("answer") or "")
            unanswerable = category == 5
            judge_raw = await llm_text(
                client,
                JUDGE_SYSTEM_PROMPT,
                "\n".join([
                    f"Question: {qa['question']}",
                    f"Reference answer: {reference if reference else '[UNANSWERABLE]'}",
                    f"Candidate answer: {candidate}",
                ]),
                json_mode=True,
            )
            judge = parse_json_object(judge_raw)

            evidence_sessions = sorted({
                dia_to_session[evidence]
                for evidence in qa.get("evidence", [])
                if evidence in dia_to_session
            })
            # Session-level relevance is inferred from each memory's memory_at date.
            # This is approximate because Hy-Memory search does not return session_id.
            session_dates = {
                name: parse_locomo_datetime(conversation[f"{name}_date_time"]).date().isoformat()
                for name in sessions
            }
            evidence_dates = {session_dates[name] for name in evidence_sessions}
            ranked_dates = []
            for item in memories:
                timestamp = item.get("memory_at")
                ranked_dates.append(
                    datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
                    if isinstance(timestamp, (int, float)) else None
                )
            gold_evidence_ids = {str(value) for value in qa.get("evidence", []) if value}
            retrieved_ids_at_1 = retrieved_evidence_ids(memories, 1)
            retrieved_ids_at_5 = retrieved_evidence_ids(memories, 5)
            retrieved_ids_at_10 = retrieved_evidence_ids(memories, 10)
            l2_l7_items = [
                item for item in memories
                if item.get("layer") in {"l2_fact", "l7_intention"}
            ]
            provenance_coverage = (
                sum(bool(item.get("evidence_chain")) for item in l2_l7_items)
                / len(l2_l7_items)
                if l2_l7_items else None
            )
            entry = {
                "qa_index": qa_index,
                "category": category,
                "category_name": CATEGORY_NAMES[category],
                "question": qa["question"],
                "reference_answer": reference,
                "unanswerable": unanswerable,
                "evidence_dialog_ids": qa.get("evidence", []),
                "evidence_sessions": evidence_sessions,
                "candidate_answer": candidate,
                "answer_prompt_version": ANSWER_PROMPT_VERSION,
                "judge": judge,
                "search_elapsed_ms": search_ms,
                "retrieved_count": len(memories),
                # Backward-compatible approximate metric: session date only.
                "evidence_hit_at_1": bool(evidence_dates.intersection(ranked_dates[:1])) if not unanswerable else None,
                "evidence_hit_at_5": bool(evidence_dates.intersection(ranked_dates[:5])) if not unanswerable else None,
                "evidence_hit_at_10": bool(evidence_dates.intersection(ranked_dates[:10])) if not unanswerable else None,
                # Strict provenance metrics against annotated LoCoMo dia_id values.
                "exact_evidence_hit_at_1": bool(gold_evidence_ids.intersection(retrieved_ids_at_1)) if not unanswerable else None,
                "exact_evidence_hit_at_5": bool(gold_evidence_ids.intersection(retrieved_ids_at_5)) if not unanswerable else None,
                "exact_evidence_hit_at_10": bool(gold_evidence_ids.intersection(retrieved_ids_at_10)) if not unanswerable else None,
                "evidence_recall_at_1": evidence_recall(gold_evidence_ids, retrieved_ids_at_1) if not unanswerable else None,
                "evidence_recall_at_5": evidence_recall(gold_evidence_ids, retrieved_ids_at_5) if not unanswerable else None,
                "evidence_recall_at_10": evidence_recall(gold_evidence_ids, retrieved_ids_at_10) if not unanswerable else None,
                "retrieved_evidence_dialog_ids_at_1": sorted(retrieved_ids_at_1),
                "retrieved_evidence_dialog_ids_at_5": sorted(retrieved_ids_at_5),
                "retrieved_evidence_dialog_ids_at_10": sorted(retrieved_ids_at_10),
                "l2_l7_provenance_coverage": provenance_coverage,
                "retrieved_memories": memories,
                "search_rewrite": raw_search.get("rewrite") if isinstance(raw_search, dict) else None,
            }
            report.setdefault("questions", []).append(entry)
            report["metrics"] = compute_metrics(report)
            save_report(report, result_path)
            print(
                f"qa {qa_index} {CATEGORY_NAMES[category]}: correct={bool(judge.get('correct'))} "
                f"recall@5={entry['evidence_recall_at_5']} search={search_ms}ms",
                flush=True,
            )

        report["memory_inventory"] = compact(client.list_memories(
            user_id=user_id,
            agent_id=agent_id,
            limit=1000,
        ))
        report["metrics"] = compute_metrics(report)
        report["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
        save_report(report, result_path)
        return result_path
    finally:
        client.close()
        # Give the package's metrics tasks one event-loop tick to finish cleanly.
        await asyncio.sleep(0)


def main() -> None:
    args = parse_args()
    path = asyncio.run(run(args))
    print(f"LoCoMo Ultra evaluation complete: {path}")


if __name__ == "__main__":
    main()
