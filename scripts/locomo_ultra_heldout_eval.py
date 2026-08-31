"""Run a resumable, role-compatible LoCoMo Ultra held-out evaluation.

The fixed manifest contains 500 questions from the nine conversations that
were not used by the conv-26 development regression.  Phase 1 contains 300
questions; phase 2 extends the same run to 500 without ingesting memories
again.  Checkpoints are written after every session, digest, and question.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from locomo_ultra_eval import (
    ANSWER_PROMPT_VERSION,
    ANSWER_SYSTEM_PROMPT,
    CATEGORY_NAMES,
    DEFAULT_DATASET,
    JUDGE_SYSTEM_PROMPT,
    PROJECT_ROOT,
    RESULTS_DIR,
    build_client,
    compact,
    compute_metrics as compute_question_metrics,
    configure_environment,
    convert_session,
    evidence_map,
    evidence_recall,
    flatten_memories,
    memory_text,
    parse_json_object,
    parse_locomo_datetime,
    retrieved_evidence_ids,
    save_report,
    session_number,
)


MANIFEST_PATH = PROJECT_ROOT / "benchmarks" / "locomo" / "heldout_500_seed_20260816.json"
MANIFEST_SEED = 20260816
DEV_SAMPLE_ID = "conv-26"
AGENT_ID = "locomo-ultra-heldout"

# Phase 1 is exactly 300 questions. Phase 2 extends the same frozen manifest
# to 500. Open-domain uses every strict role-compatible held-out question.
PHASE1_TARGETS = {1: 60, 2: 60, 3: 26, 4: 96, 5: 58}
FULL_TARGETS = {1: 100, 2: 100, 3: 44, 4: 160, 5: 96}

INPUT_PRICE_PER_MILLION = 12.0
OUTPUT_PRICE_PER_MILLION = 24.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Hy-Memory Ultra on a 300/500-question LoCoMo held-out split"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--question-limit", type=int, choices=(300, 500), default=300)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--normal-min-score", type=float, default=0.4)
    parser.add_argument("--intention-min-score", type=float, default=0.4)
    parser.add_argument("--resume-run", help="Resume a previous held-out run id")
    parser.add_argument("--skip-digest", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dialog_speaker_map(sample: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    conversation = sample["conversation"]
    for name, turns in conversation.items():
        if not re.fullmatch(r"session_\d+", name):
            continue
        for turn in turns:
            dia_id = str(turn.get("dia_id") or "")
            if dia_id:
                result[dia_id] = str(turn.get("speaker") or "")
    return result


def _eligible_questions(data: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    pools: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample_index, sample in enumerate(data):
        if sample.get("sample_id") == DEV_SAMPLE_ID:
            continue
        conversation = sample["conversation"]
        speaker_a = str(conversation["speaker_a"])
        speaker_b = str(conversation["speaker_b"])
        speaker_map = _dialog_speaker_map(sample)
        subject_pattern = re.compile(rf"\b{re.escape(speaker_a)}\b", re.IGNORECASE)

        for qa_index, qa in enumerate(sample.get("qa") or []):
            category = int(qa["category"])
            if category not in FULL_TARGETS:
                continue
            # Hy-Memory maps speaker_a to the user. Keep answerable questions
            # about that user, and category-5 speaker-swap questions that name
            # the user while their annotated evidence belongs to speaker_b.
            if not subject_pattern.search(str(qa.get("question") or "")):
                continue
            evidence_speakers = {
                speaker_map.get(str(dia_id), "")
                for dia_id in (qa.get("evidence") or [])
            }
            role_compatible = (
                speaker_b in evidence_speakers
                if category == 5
                else speaker_a in evidence_speakers
            )
            if not role_compatible:
                continue
            pools[category].append({
                "sample_index": sample_index,
                "sample_id": sample["sample_id"],
                "qa_index": qa_index,
                "category": category,
                "category_name": CATEGORY_NAMES[category],
                "question_key": f"{sample['sample_id']}:{qa_index}",
            })
    return pools


def _round_robin_sample(
    candidates: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        groups[item["sample_id"]].append(item)
    for values in groups.values():
        rng.shuffle(values)
    sample_ids = sorted(groups)
    rng.shuffle(sample_ids)

    chosen: list[dict[str, Any]] = []
    while len(chosen) < count:
        progressed = False
        for sample_id in sample_ids:
            if groups[sample_id] and len(chosen) < count:
                chosen.append(groups[sample_id].pop())
                progressed = True
        if not progressed:
            break
    if len(chosen) != count:
        raise RuntimeError(f"requested {count} questions but only selected {len(chosen)}")
    return chosen


def build_manifest(data: list[dict[str, Any]], dataset_path: Path) -> dict[str, Any]:
    rng = random.Random(MANIFEST_SEED)
    pools = _eligible_questions(data)
    phase1: list[dict[str, Any]] = []
    phase2: list[dict[str, Any]] = []

    for category in sorted(FULL_TARGETS):
        full_count = FULL_TARGETS[category]
        first_count = PHASE1_TARGETS[category]
        if len(pools[category]) < full_count:
            raise RuntimeError(
                f"category {category} has {len(pools[category])} eligible questions, "
                f"needs {full_count}"
            )
        selected = _round_robin_sample(pools[category], full_count, rng)
        for item in selected[:first_count]:
            phase1.append({**item, "phase": 1})
        for item in selected[first_count:]:
            phase2.append({**item, "phase": 2})

    # Grouping by sample makes ingestion/evaluation efficient while the phase
    # boundary preserves the exact 300 -> 500 incremental contract.
    sort_key = lambda item: (item["sample_index"], item["category"], item["qa_index"])
    questions = sorted(phase1, key=sort_key) + sorted(phase2, key=sort_key)
    raw = dataset_path.read_bytes()
    return {
        "name": "LoCoMo Ultra held-out 500",
        "seed": MANIFEST_SEED,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": hashlib.sha256(raw).hexdigest().upper(),
        "development_sample_excluded": DEV_SAMPLE_ID,
        "eligibility": (
            "question explicitly names speaker_a; answerable evidence includes speaker_a; "
            "category-5 evidence includes speaker_b"
        ),
        "phase1_targets": {CATEGORY_NAMES[k]: v for k, v in PHASE1_TARGETS.items()},
        "full_targets": {CATEGORY_NAMES[k]: v for k, v in FULL_TARGETS.items()},
        "questions": questions,
    }


def ensure_manifest(data: list[dict[str, Any]], dataset_path: Path, path: Path) -> dict[str, Any]:
    expected = build_manifest(data, dataset_path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(
                f"frozen manifest differs from regenerated selection: {path}"
            )
        return existing
    save_report(expected, path)
    return expected


def _llm_usage(response: Any) -> dict[str, Any]:
    prompt = int(getattr(response, "prompt_tokens", 0) or 0)
    completion = int(getattr(response, "completion_tokens", 0) or 0)
    return {
        "model": str(getattr(response, "model", "") or ""),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(getattr(response, "tokens_used", 0) or prompt + completion),
        "estimated_cny": round(
            prompt / 1_000_000 * INPUT_PRICE_PER_MILLION
            + completion / 1_000_000 * OUTPUT_PRICE_PER_MILLION,
            6,
        ),
    }


async def llm_call(
    client: Any, system: str, user: str, *, json_mode: bool = False
) -> tuple[str, dict[str, Any]]:
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
    return str(response.content).strip(), _llm_usage(response)


def _classify_failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    quota_markers = (
        "insufficient_quota", "quota", "balance", "billing", "payment required",
        "arrears", "欠费", "余额不足", "额度不足", "402",
    )
    return "quota_or_balance" if any(marker in text for marker in quota_markers) else "runtime_error"


def _usage_summary(report: dict[str, Any]) -> dict[str, Any]:
    answer_prompt = answer_completion = judge_prompt = judge_completion = 0
    for item in report.get("questions", []):
        answer = item.get("answer_usage") or {}
        judge = item.get("judge_usage") or {}
        answer_prompt += int(answer.get("prompt_tokens") or 0)
        answer_completion += int(answer.get("completion_tokens") or 0)
        judge_prompt += int(judge.get("prompt_tokens") or 0)
        judge_completion += int(judge.get("completion_tokens") or 0)
    prompt = answer_prompt + judge_prompt
    completion = answer_completion + judge_completion
    return {
        "answer_prompt_tokens": answer_prompt,
        "answer_completion_tokens": answer_completion,
        "judge_prompt_tokens": judge_prompt,
        "judge_completion_tokens": judge_completion,
        "qa_prompt_tokens": prompt,
        "qa_completion_tokens": completion,
        "qa_estimated_cny": round(
            prompt / 1_000_000 * INPUT_PRICE_PER_MILLION
            + completion / 1_000_000 * OUTPUT_PRICE_PER_MILLION,
            4,
        ),
    }


def _inventory_summary(value: Any) -> dict[str, Any]:
    inventory = compact(value)
    vdb_memories = ((inventory.get("vdb") or {}).get("memories") or [])
    layers = Counter(str(item.get("layer") or "unknown") for item in vdb_memories)
    graph = inventory.get("graph") or []
    if isinstance(graph, dict):
        graph_count = int(graph.get("total") or len(graph.get("memories") or []))
    else:
        graph_count = len(graph)
    return {
        "vdb_total": len(vdb_memories),
        "layers": dict(sorted(layers.items())),
        "graph_total": graph_count,
    }


def compute_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Extend question metrics with held-out multi-sample write/inventory data."""
    metrics = compute_question_metrics(report)
    sample_items = list((report.get("samples") or {}).items())
    samples = [sample for _, sample in sample_items]
    writes = [
        write
        for sample in samples
        for write in (sample.get("writes") or [])
    ]
    layers: Counter[str] = Counter()
    graph_total = 0
    digest_successes = 0
    digest_failures: list[str] = []
    for sample_id, sample in sample_items:
        inventory = sample.get("memory_inventory") or {}
        layers.update({
            str(layer): int(count or 0)
            for layer, count in (inventory.get("layers") or {}).items()
        })
        graph_total += int(inventory.get("graph_total") or 0)
        digest = sample.get("digest")
        if digest is None:
            continue
        result = digest.get("result") or {}
        if result.get("success") is False:
            digest_failures.append(str(sample_id))
        else:
            digest_successes += 1

    metrics["write_success_rate"] = round(
        sum(bool(item.get("success")) for item in writes) / max(1, len(writes)),
        4,
    )
    metrics["write_sessions"] = {
        "total": len(writes),
        "successful": sum(bool(item.get("success")) for item in writes),
        "failed": sum(not bool(item.get("success")) for item in writes),
    }
    metrics["memory_layers"] = dict(sorted(layers.items()))
    metrics["graph_total"] = graph_total
    digest_total = digest_successes + len(digest_failures)
    metrics["digest"] = {
        "total": digest_total,
        "successful": digest_successes,
        "failed": len(digest_failures),
        "success_rate": round(digest_successes / max(1, digest_total), 4),
        "failed_sample_ids": digest_failures,
    }
    return metrics


async def run(args: argparse.Namespace) -> Path:
    dataset_path = args.dataset.resolve()
    data = load_dataset(dataset_path)
    manifest = ensure_manifest(data, dataset_path, args.manifest.resolve())
    if args.prepare_only:
        print(f"Held-out manifest ready: {args.manifest.resolve()}")
        raise SystemExit(0)

    selected = manifest["questions"][: args.question_limit]
    selected_keys = {item["question_key"] for item in selected}
    run_id = args.resume_run or datetime.now().strftime("%Y%m%d-%H%M%S")
    configure_environment(f"heldout-{run_id}")
    result_path = RESULTS_DIR / f"locomo-ultra-heldout-{run_id}.json"

    if args.resume_run and result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        report = {
            "run_id": run_id,
            "benchmark": "LoCoMo (ACL 2024)",
            "split": "held-out; conv-26 excluded from final score",
            "dataset_file": str(dataset_path),
            "manifest_file": str(args.manifest.resolve()),
            "manifest_seed": manifest["seed"],
            "question_limit": args.question_limit,
            "models": {},
            "answer_policy": {
                "version": ANSWER_PROMPT_VERSION,
                "system_prompt": ANSWER_SYSTEM_PROMPT,
            },
            "retrieval_policy": {
                "top_k": args.top_k,
                "normal_min_score": args.normal_min_score,
                "profile_min_score": 0.4,
                "intention_min_score": args.intention_min_score,
                "zero_result_fallback_min_score": 0.3,
            },
            "selection": {},
            "samples": {},
            "questions": [],
        }

    report["question_limit"] = args.question_limit
    report["selection"] = {
        "question_keys": [item["question_key"] for item in selected],
        "category_distribution": dict(Counter(item["category_name"] for item in selected)),
        "sample_distribution": dict(Counter(item["sample_id"] for item in selected)),
    }
    # A 300-question checkpoint can be resumed at 500; never discard completed
    # phase-2 records if a user inspects it again with a lower limit.
    completed_keys = {item.get("question_key") for item in report.get("questions", [])}
    report["selected_completed"] = len(selected_keys.intersection(completed_keys))
    save_report(report, result_path)

    entries_by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        entries_by_sample[int(item["sample_index"])].append(item)

    client = build_client()
    report["models"] = {
        "llm": client._config.llm.model,
        "embedder": client._config.embedder.model,
        "embedding_dims": client._config.embedder.embedding_dims,
    }
    save_report(report, result_path)

    try:
        for sample_index in sorted(entries_by_sample):
            sample = data[sample_index]
            sample_id = str(sample["sample_id"])
            conversation = sample["conversation"]
            user_id = f"locomo-heldout-{sample_id}-{run_id}"
            state = report.setdefault("samples", {}).setdefault(sample_id, {
                "sample_index": sample_index,
                "speakers": [conversation["speaker_a"], conversation["speaker_b"]],
                "user_id": user_id,
                "agent_id": AGENT_ID,
                "writes": [],
                "digest": None,
            })
            user_id = state["user_id"]
            sessions = sorted(
                (name for name in conversation if re.fullmatch(r"session_\d+", name)),
                key=session_number,
            )
            benchmark_as_of = max(
                parse_locomo_datetime(conversation[f"{name}_date_time"])
                for name in sessions
            )
            state["as_of"] = benchmark_as_of.isoformat()

            existing_sessions = {
                item["session_id"] for item in state.get("writes", []) if item.get("success")
            }
            for name in sessions:
                if name in existing_sessions:
                    continue
                date_text = conversation[f"{name}_date_time"]
                messages = convert_session(conversation[name], conversation["speaker_a"])
                started = time.perf_counter()
                result = compact(client.add(
                    messages,
                    user_id=user_id,
                    agent_id=AGENT_ID,
                    session_id=name,
                    memory_at=parse_locomo_datetime(date_text),
                ))
                write = {
                    "session_id": name,
                    "session_date": date_text,
                    "turn_count": len(messages),
                    "success": not (isinstance(result, dict) and result.get("success") is False),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "result": result,
                }
                state.setdefault("writes", []).append(write)
                save_report(report, result_path)
                print(
                    f"[{sample_id}] write {name}: success={write['success']} "
                    f"elapsed={write['elapsed_ms']}ms",
                    flush=True,
                )
                if not write["success"]:
                    raise RuntimeError(f"write failed for {sample_id}/{name}: {result}")

            if state.get("digest") is None and not args.skip_digest:
                started = time.perf_counter()
                digest = compact(client.digest(user_id=user_id, agent_id=AGENT_ID))
                state["digest"] = {
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "result": digest,
                }
                save_report(report, result_path)

            dia_to_session: dict[str, str] = {}
            for number, ids in evidence_map(sample).items():
                for dia_id in ids:
                    dia_to_session[dia_id] = f"session_{number}"
            session_dates = {
                name: parse_locomo_datetime(conversation[f"{name}_date_time"]).date().isoformat()
                for name in sessions
            }
            existing_questions = {
                item.get("question_key") for item in report.get("questions", [])
            }

            for selection in entries_by_sample[sample_index]:
                question_key = selection["question_key"]
                if question_key in existing_questions:
                    continue
                qa_index = int(selection["qa_index"])
                qa = sample["qa"][qa_index]
                category = int(qa["category"])

                search_started = time.perf_counter()
                raw_search = compact(client.search(
                    qa["question"],
                    user_ids=[user_id],
                    agent_ids=[AGENT_ID],
                    limit=args.top_k,
                    min_score=args.normal_min_score,
                    intention_min_score=args.intention_min_score,
                    as_of=benchmark_as_of,
                ))
                search_ms = round((time.perf_counter() - search_started) * 1000, 2)
                memories = flatten_memories(raw_search)
                context = memory_text(memories)
                candidate, answer_usage = await llm_call(
                    client,
                    ANSWER_SYSTEM_PROMPT,
                    f"Retrieved memories:\n{context or '(none)'}\n\nQuestion: {qa['question']}",
                )
                reference = str(qa.get("answer") or "")
                judge_raw, judge_usage = await llm_call(
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
                unanswerable = category == 5
                evidence_sessions = sorted({
                    dia_to_session[evidence]
                    for evidence in qa.get("evidence", [])
                    if evidence in dia_to_session
                })
                evidence_dates = {session_dates[name] for name in evidence_sessions}
                ranked_dates = []
                for item in memories:
                    timestamp = item.get("memory_at")
                    ranked_dates.append(
                        datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
                        if isinstance(timestamp, (int, float)) else None
                    )
                gold_ids = {str(value) for value in qa.get("evidence", []) if value}
                ids1 = retrieved_evidence_ids(memories, 1)
                ids5 = retrieved_evidence_ids(memories, 5)
                ids10 = retrieved_evidence_ids(memories, 10)
                l2_l7 = [
                    item for item in memories
                    if item.get("layer") in {"l2_fact", "l7_intention"}
                ]
                provenance = (
                    sum(bool(item.get("evidence_chain")) for item in l2_l7) / len(l2_l7)
                    if l2_l7 else None
                )
                entry = {
                    "question_key": question_key,
                    "sample_index": sample_index,
                    "sample_id": sample_id,
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
                    "answer_usage": answer_usage,
                    "judge": judge,
                    "judge_usage": judge_usage,
                    "search_elapsed_ms": search_ms,
                    "retrieved_count": len(memories),
                    "zero_result_fallback": raw_search.get("fallback"),
                    "evidence_hit_at_1": bool(evidence_dates.intersection(ranked_dates[:1])) if not unanswerable else None,
                    "evidence_hit_at_5": bool(evidence_dates.intersection(ranked_dates[:5])) if not unanswerable else None,
                    "evidence_hit_at_10": bool(evidence_dates.intersection(ranked_dates[:10])) if not unanswerable else None,
                    "exact_evidence_hit_at_1": bool(gold_ids.intersection(ids1)) if not unanswerable else None,
                    "exact_evidence_hit_at_5": bool(gold_ids.intersection(ids5)) if not unanswerable else None,
                    "exact_evidence_hit_at_10": bool(gold_ids.intersection(ids10)) if not unanswerable else None,
                    "evidence_recall_at_1": evidence_recall(gold_ids, ids1) if not unanswerable else None,
                    "evidence_recall_at_5": evidence_recall(gold_ids, ids5) if not unanswerable else None,
                    "evidence_recall_at_10": evidence_recall(gold_ids, ids10) if not unanswerable else None,
                    "retrieved_evidence_dialog_ids_at_1": sorted(ids1),
                    "retrieved_evidence_dialog_ids_at_5": sorted(ids5),
                    "retrieved_evidence_dialog_ids_at_10": sorted(ids10),
                    "l2_l7_provenance_coverage": provenance,
                    "retrieved_memories": memories,
                    "search_rewrite": raw_search.get("rewrite"),
                }
                report.setdefault("questions", []).append(entry)
                existing_questions.add(question_key)
                report["metrics"] = compute_metrics(report)
                report["usage"] = _usage_summary(report)
                report["selected_completed"] = len(
                    selected_keys.intersection(existing_questions)
                )
                save_report(report, result_path)
                print(
                    f"[{sample_id}] qa {qa_index} {CATEGORY_NAMES[category]}: "
                    f"correct={bool(judge.get('correct'))} "
                    f"fallback={bool((raw_search.get('fallback') or {}).get('used'))} "
                    f"recall@5={entry['evidence_recall_at_5']} search={search_ms}ms",
                    flush=True,
                )

            state["memory_inventory"] = _inventory_summary(client.list_memories(
                user_id=user_id,
                agent_id=AGENT_ID,
                limit=1000,
            ))
            save_report(report, result_path)

        report["metrics"] = compute_metrics(report)
        report["usage"] = _usage_summary(report)
        report["selected_completed"] = len(selected_keys.intersection({
            item.get("question_key") for item in report.get("questions", [])
        }))
        if report["selected_completed"] == len(selected):
            report["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
            report.pop("halted", None)
        save_report(report, result_path)
        return result_path
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            report["halted"] = {
                "at": datetime.now(tz=timezone.utc).isoformat(),
                "kind": _classify_failure(exc),
                "error_type": type(exc).__name__,
                "error": str(exc)[:4000],
                "selected_completed": report.get("selected_completed", 0),
            }
            report["metrics"] = compute_metrics(report)
            report["usage"] = _usage_summary(report)
            save_report(report, result_path)
        raise
    finally:
        client.close()
        await asyncio.sleep(0)


def main() -> None:
    args = parse_args()
    path = asyncio.run(run(args))
    print(f"LoCoMo Ultra held-out evaluation complete: {path}")


if __name__ == "__main__":
    main()
