from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from locomo_ultra_heldout_eval import compute_metrics


def test_multi_sample_metrics_aggregate_writes_inventory_and_digest_failures() -> None:
    report = {
        "questions": [],
        "samples": {
            "conv-a": {
                "writes": [{"success": True}, {"success": True}],
                "memory_inventory": {
                    "layers": {"l2_fact": 3, "l7_intention": 1},
                    "graph_total": 2,
                },
                "digest": {"result": {"success": True}},
            },
            "conv-b": {
                "writes": [{"success": False}],
                "memory_inventory": {
                    "layers": {"l2_fact": 2},
                    "graph_total": 0,
                },
                "digest": {"result": {"success": False}},
            },
        },
    }

    metrics = compute_metrics(report)

    assert metrics["write_success_rate"] == 0.6667
    assert metrics["write_sessions"] == {
        "total": 3,
        "successful": 2,
        "failed": 1,
    }
    assert metrics["memory_layers"] == {"l2_fact": 5, "l7_intention": 1}
    assert metrics["graph_total"] == 2
    assert metrics["digest"] == {
        "total": 2,
        "successful": 1,
        "failed": 1,
        "success_rate": 0.5,
        "failed_sample_ids": ["conv-b"],
    }
