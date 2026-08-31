from datetime import datetime, timezone

from scripts.locomo_ultra_eval import (
    ANSWER_PROMPT_VERSION,
    ANSWER_SYSTEM_PROMPT,
    memory_text,
    evidence_recall,
    retrieved_evidence_ids,
)


def test_balanced_answer_policy_allows_grounded_inference_without_benchmark_overfit():
    prompt = " ".join(ANSWER_SYSTEM_PROMPT.lower().split())

    assert ANSWER_PROMPT_VERSION == "balanced-evidence-v2"
    assert "reasonable, best-supported inference" in prompt
    assert "ordinary commonsense as the connecting rule" in prompt
    assert "do not abstain merely because synthesis is required" in prompt
    assert "never as a source of missing person-specific facts" in prompt
    assert "do not transfer a fact from one person to another" in prompt
    assert "using only the retrieved memories" not in prompt

    # Guard against tuning the policy to this particular LoCoMo sample or its
    # failed questions.  The prompt must describe general evidence behavior.
    benchmark_specific_terms = {
        "caroline",
        "counseling",
        "political",
        "religious",
        "adoption",
        "home country",
        "lgbtq",
    }
    assert not {term for term in benchmark_specific_terms if term in prompt}


def test_memory_context_exposes_generic_temporal_evidence_fields():
    def epoch(value: str) -> float:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()

    rendered = memory_text(
        [
            {
                "layer": "l7_intention",
                "content": "The user plans to research an organization.",
                "observed_at": epoch("2023-05-25T13:14:00"),
                "temporal_relation": "future",
                "event_start": epoch("2023-06-01T00:00:00"),
                "event_end": epoch("2023-08-31T23:59:59"),
            }
        ]
    )

    assert "layer=l7_intention" in rendered
    assert "observed=2023-05-25" in rendered
    assert "time_relation=future" in rendered
    assert "event=2023-06-01..2023-08-31" in rendered
    assert "The user plans to research an organization." in rendered


def test_strict_evidence_recall_uses_original_dialog_ids():
    memories = [
        {"evidence_chain": ["D1:2", "D1:3"]},
        {"evidence_chain": ["D4:1"]},
    ]
    assert retrieved_evidence_ids(memories, 1) == {"D1:2", "D1:3"}
    assert retrieved_evidence_ids(memories, 5) == {"D1:2", "D1:3", "D4:1"}
    assert evidence_recall({"D1:2", "D4:1"}, retrieved_evidence_ids(memories, 1)) == 0.5
    assert evidence_recall({"D1:2", "D4:1"}, retrieved_evidence_ids(memories, 5)) == 1.0
