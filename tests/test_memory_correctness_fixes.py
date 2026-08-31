import json
import inspect
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "hy-memory-1.2.21"
sys.path.insert(0, str(SOURCE_ROOT))

from hy_memory.agent.reconciler import MemoryReconciler, ReconcileOp, ReconcileResult
from hy_memory.agent.extractor import EXTRACT_PROMPT, EXTRACT_SYSTEM_PROMPT
from hy_memory.data.graph_store_kuzu import KuzuGraphStore
from hy_memory.data.vector_store_chroma import ChromaVectorStore
from hy_memory.models.memory import MemoryLayer, MemoryNode, MemoryStatus
from hy_memory.metrics import MetricsCollector
from hy_memory.client import HyMemoryClient, _serialize_search_memory
from hy_memory.config import MemoryConfig
from hy_memory.pipelines.base import ChatMessage, ReadRequest, WriteRequest
from hy_memory.pipelines.cross_domain_sweeper import CrossDomainSweeper
from hy_memory.pipelines.reader_legacy import (
    LegacyReadPipeline,
    ZERO_RESULT_FALLBACK_MIN_SCORE,
)
from hy_memory.pipelines.system2_agent import run_system2_agent
from hy_memory.pipelines.writer import MemoryWriter, _temporal_kwargs
from hy_memory.pipelines._retrieval.reconcile_retrieval import ReconcileRetrievalConfig


class _MetricsCacheStub:
    def __init__(self):
        self.stored = []

    async def store_metrics_minute(self, minute_key, data):
        self.stored.append((minute_key, data))


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


@pytest.mark.asyncio
async def test_metrics_background_tasks_are_cancelled_and_final_bucket_is_flushed():
    collector = MetricsCollector()
    cache = _MetricsCacheStub()
    collector.bind_cache(cache)
    collector.sys1_start()
    collector.sys1_end({}, success=True)

    await collector.start_background_tasks()
    tasks = list(collector._background_tasks)
    assert len(tasks) == 2
    assert all(not task.done() for task in tasks)

    await collector.stop_background_tasks()

    assert collector._background_started is False
    assert collector._background_tasks == []
    assert all(task.done() for task in tasks)
    assert len(cache.stored) == 1


def test_l7_recall_is_enabled_by_default_without_query_routing():
    """Default chat search must recall L7 without requiring a route decision."""
    assert ReadRequest().intention_limit == 10
    assert ReadRequest().intention_min_score == 0.4
    assert (
        inspect.signature(HyMemoryClient.search)
        .parameters["intention_limit"]
        .default
        == 10
    )
    assert inspect.signature(HyMemoryClient.search).parameters["intention_min_score"].default == 0.4
    assert inspect.signature(HyMemoryClient.async_search).parameters["intention_min_score"].default == 0.4


@pytest.mark.asyncio
async def test_legacy_reader_retries_all_empty_buckets_once_at_fixed_point_three():
    node = MemoryNode(
        node_id="fallback-hit",
        user_id="u",
        agent_id="a",
        layer=MemoryLayer.L2_FACT,
        content="The user supports equal rights.",
    )

    class _Embedder:
        async def embed(self, _text):
            return [1.0, 0.0]

    class _Store:
        _client = object()

        def __init__(self):
            self.thresholds = []

        async def search(self, **kwargs):
            threshold = kwargs.get("score_threshold")
            self.thresholds.append(threshold)
            layers = kwargs.get("layers") or []
            if threshold == ZERO_RESULT_FALLBACK_MIN_SCORE and MemoryLayer.L2_FACT in layers:
                return [{"node_id": node.node_id, "score": 0.35, "node": node}]
            return []

    store = _Store()
    reader = LegacyReadPipeline(
        MemoryConfig(),
        embed_service=_Embedder(),
        vector_store=store,
    )
    await reader.initialize()

    response = await reader.read(ReadRequest(
        query="What is the user's political leaning?",
        user_ids=["u"],
        agent_ids=["a"],
        min_score=0.4,
        profile_min_score=0.4,
        intention_min_score=0.4,
    ))

    assert response.success is True
    assert response.total_found == 1
    assert response.memories[0]["memory_id"] == "fallback-hit"
    assert response.extra["zero_result_fallback"] == {
        "used": True,
        "min_score": 0.3,
    }
    assert 0.4 in store.thresholds
    assert 0.3 in store.thresholds


def test_chat_message_accepts_external_turn_id_aliases():
    normalized = HyMemoryClient._normalize_message({
        "role": "user",
        "content": "hello",
        "dia_id": "D1:2",
    })
    assert normalized[0].turn_id == "D1:2"
    assert normalized[0].to_dict()["turn_id"] == "D1:2"
    assert ChatMessage.from_dict({"message_id": "external-7"}).turn_id == "external-7"
    assert (
        inspect.signature(HyMemoryClient.async_search)
        .parameters["intention_limit"]
        .default
        == 10
    )


def test_search_response_preserves_temporal_fields():
    payload = {
        "memory_id": "temporal-memory",
        "content": "The user attended a support group.",
        "score": 0.81234,
        "layer": "l2_fact",
        "observed_at": 1683525360,
        "temporal_relation": "past",
        "event_time_text": "yesterday",
        "event_start": 1683388800,
        "event_end": 1683475199,
        "normalization_confidence": 0.9,
        "valid_from": 1683388800,
        "valid_until": None,
        "source_session_id": "session_1",
        "source_turn_index": 3,
        "evidence_chain": ["D1:3", "D1:4"],
    }

    serialized = _serialize_search_memory(payload)

    assert serialized["score"] == 0.8123
    for field_name in (
        "observed_at",
        "temporal_relation",
        "event_time_text",
        "event_start",
        "event_end",
        "normalization_confidence",
        "valid_from",
        "valid_until",
    ):
        assert field_name in serialized
        assert serialized[field_name] == payload[field_name]
    assert serialized["source_session_id"] == "session_1"
    assert serialized["source_turn_index"] == 3
    assert serialized["evidence_chain"] == ["D1:3", "D1:4"]


@pytest.mark.asyncio
async def test_intention_expiry_uses_injected_as_of_and_handles_timezone():
    from hy_memory.pipelines._retrieval.intention import recall_intentions

    node = MemoryNode(
        node_id="summer-plan",
        user_id="u",
        agent_id="a",
        layer=MemoryLayer.L7_INTENTION,
        content="The user plans a summer trip.",
        valid_until=_dt("2023-08-31T23:59:59"),
    )

    class _Store:
        def __init__(self):
            self.updates = []
            self.search_kwargs = None

        async def search(self, **kwargs):
            self.search_kwargs = kwargs
            return [{"node_id": node.node_id, "score": 0.8, "node": node}]

        async def update_payload(self, node_id, payload):
            self.updates.append((node_id, payload))

    before_store = _Store()
    before = await recall_intentions(
        before_store,
        [0.1],
        limit=10,
        now=datetime(2023, 8, 1, tzinfo=timezone.utc),
    )
    assert [item["node_id"] for item in before] == ["summer-plan"]
    assert before_store.updates == []
    assert before_store.search_kwargs["score_threshold"] == 0.4

    after_store = _Store()
    after = await recall_intentions(
        after_store,
        [0.1],
        limit=10,
        now=datetime(2023, 9, 1, tzinfo=timezone.utc),
    )
    assert after == []
    assert after_store.updates == [
        ("summer-plan", {"layer": MemoryLayer.L2_FACT.value})
    ]


def test_l2_l7_temporal_schema_round_trip_and_legacy_fallback():
    observed = _dt("2023-05-25T13:14:00")
    node = MemoryNode(
        node_id="future-plan",
        user_id="u",
        agent_id="a",
        layer=MemoryLayer.L7_INTENTION,
        content="用户计划在暑假研究收养机构。",
        memory_at=observed,
        observed_at=observed,
        temporal_relation="future",
        event_time_text="this summer",
        event_start=_dt("2023-06-01"),
        event_end=_dt("2023-08-31T23:59:59"),
        normalization_confidence=0.75,
        valid_from=_dt("2023-05-25"),
        valid_until=_dt("2023-08-31T23:59:59"),
    )

    restored = MemoryNode.from_dict(node.to_dict())
    assert restored.content == node.content
    assert restored.observed_at == observed
    assert restored.temporal_relation == "future"
    assert restored.event_time_text == "this summer"
    assert restored.event_start == _dt("2023-06-01")
    assert restored.event_end == _dt("2023-08-31T23:59:59")
    assert restored.normalization_confidence == 0.75
    assert restored.valid_from == _dt("2023-05-25")
    assert restored.valid_until == _dt("2023-08-31T23:59:59")

    legacy = MemoryNode.from_dict(
        {
            "node_id": "legacy",
            "user_id": "u",
            "agent_id": "a",
            "layer": "l2_fact",
            "content": "legacy payload",
            "memory_at": int(observed.timestamp()),
        }
    )
    assert legacy.observed_at == observed


@pytest.mark.asyncio
async def test_chroma_persists_l2_temporal_schema():
    import chromadb

    config = SimpleNamespace(
        vector_store=SimpleNamespace(
            collection_name="temporal_test",
            embedding_dims=4,
            persist_directory="",
            host=None,
            port=None,
        )
    )
    store = ChromaVectorStore(config)
    store._client = chromadb.EphemeralClient()
    store._collection = store._client.get_or_create_collection(
        name=store._collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    try:
        node = MemoryNode(
            node_id="fact",
            user_id="u",
            agent_id="a",
            layer=MemoryLayer.L2_FACT,
            content="用户计划在暑假研究收养机构。",
            embedding=[0.1, 0.2, 0.3, 0.4],
            memory_at=_dt("2023-05-25T13:14:00"),
            observed_at=_dt("2023-05-25T13:14:00"),
            temporal_relation="future",
            event_time_text="this summer",
            event_start=_dt("2023-06-01"),
            event_end=_dt("2023-08-31T23:59:59"),
            normalization_confidence=0.75,
            valid_from=_dt("2023-05-25"),
            valid_until=_dt("2023-08-31T23:59:59"),
            source_session_id="session_3",
            source_turn_index=4,
            evidence_chain=["D3:4", "D3:5"],
        )
        await store.upsert(node)
        restored = await store.get_by_id("fact")
        assert restored.observed_at == _dt("2023-05-25T13:14:00")
        assert restored.temporal_relation == "future"
        assert restored.event_time_text == "this summer"
        assert restored.event_start == _dt("2023-06-01")
        assert restored.event_end == _dt("2023-08-31T23:59:59")
        assert restored.normalization_confidence == 0.75
        assert restored.valid_from == _dt("2023-05-25")
        assert restored.valid_until == _dt("2023-08-31T23:59:59")
        assert restored.source_session_id == "session_3"
        assert restored.source_turn_index == 4
        assert restored.evidence_chain == ["D3:4", "D3:5"]
    finally:
        await store.close()


def test_writer_sanitizes_and_maps_turn_provenance_to_reconcile_ops():
    request = WriteRequest(
        session_id="session_2",
        messages=[
            ChatMessage(role="assistant", content="Where?", turn_id="D2:1"),
            ChatMessage(role="user", content="Berlin.", turn_id="D2:2"),
        ],
    )
    extracted = {
        "memory": [{
            "content": "The user is moving to Berlin.",
            "source_turn_ids": ["D2:1", "invented", "D2:2"],
        }],
        "intentions": [{
            "content": "The user plans to move to Berlin.",
            "source_turn_indices": [1],
        }],
    }

    MemoryWriter._sanitize_extracted_provenance(extracted, request)
    assert extracted["memory"][0]["source_turn_ids"] == ["D2:1", "D2:2"]
    assert extracted["memory"][0]["source_turn_indices"] == [0, 1]
    assert extracted["intentions"][0]["source_turn_ids"] == ["D2:2"]

    texts, metas = MemoryWriter._collect_new_memories(extracted)
    assert texts == ["The user is moving to Berlin."]
    op = ReconcileOp(op="ADD", content=texts[0], source_indices=[0])
    turn_ids, turn_indices = MemoryWriter._op_provenance(op, metas)
    assert turn_ids == ["D2:1", "D2:2"]
    assert turn_indices == [0, 1]


class _SemanticProvenanceEmbedder:
    async def embed_batch(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "counseling" in lowered or "mental health" in lowered:
                vectors.append([1.0, 0.0])
            elif "adoption" in lowered or "adopt" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors


@pytest.mark.asyncio
async def test_writer_rejects_real_but_semantically_unrelated_turn_ids(monkeypatch):
    monkeypatch.delenv("MEMORY_PROVENANCE_VALIDATION_ENABLED", raising=False)
    monkeypatch.delenv("MEMORY_PROVENANCE_MIN_SCORE", raising=False)
    writer = MemoryWriter(
        config=SimpleNamespace(recall=SimpleNamespace(entity_store_enabled=False)),
        embed_service=_SemanticProvenanceEmbedder(),
    )
    request = WriteRequest(
        session_id="session_2",
        messages=[ChatMessage(
            role="user",
            content="I am researching adoption agencies to build a family.",
            turn_id="D2:8",
        )],
    )
    extracted = {
        "memory": [
            {
                "content": "The user plans a career in counseling and mental health.",
                "source_turn_ids": ["D2:8"],
            },
            {
                "content": "The user is researching adoption agencies to build a family.",
                "source_turn_ids": ["D2:8"],
            },
        ],
    }

    MemoryWriter._sanitize_extracted_provenance(extracted, request)
    summary = await writer._validate_extracted_provenance(extracted, request)

    assert summary == {
        "enabled": True,
        "checked": 2,
        "accepted": 1,
        "rejected": 1,
        "min_score": 0.45,
    }
    assert extracted["memory"][0]["source_turn_ids"] == []
    assert extracted["memory"][0]["source_turn_indices"] == []
    assert extracted["memory"][1]["source_turn_ids"] == ["D2:8"]


@pytest.mark.asyncio
async def test_provenance_validation_fails_open_when_embedder_is_unavailable(monkeypatch):
    class _FailingEmbedder:
        async def embed_batch(self, texts):
            raise RuntimeError("embedder offline")

    monkeypatch.delenv("MEMORY_PROVENANCE_VALIDATION_ENABLED", raising=False)
    writer = MemoryWriter(
        config=SimpleNamespace(recall=SimpleNamespace(entity_store_enabled=False)),
        embed_service=_FailingEmbedder(),
    )
    request = WriteRequest(
        messages=[ChatMessage(role="user", content="I live in Berlin.", turn_id="T1")],
    )
    extracted = {
        "memory": [{"content": "The user lives in Berlin.", "source_turn_ids": ["T1"]}],
    }
    MemoryWriter._sanitize_extracted_provenance(extracted, request)

    summary = await writer._validate_extracted_provenance(extracted, request)

    assert summary["enabled"] is False
    assert "embedder offline" in summary["error"]
    assert extracted["memory"][0]["source_turn_ids"] == ["T1"]


def test_basic_profile_location_requires_explicit_current_residence():
    origin_request = WriteRequest(messages=[ChatMessage(
        role="user",
        content="I moved away from my home country, Norway, four years ago.",
        turn_id="T1",
    )])
    origin = {"basic_info": {"location": "Norway", "name": "Ari"}}
    assert MemoryWriter._sanitize_basic_info(origin, origin_request) == {"name": "Ari"}

    residence_request = WriteRequest(messages=[ChatMessage(
        role="user",
        content="I currently live in Berlin.",
        turn_id="T2",
    )])
    residence = {"basic_info": {"location": "Berlin"}}
    assert MemoryWriter._sanitize_basic_info(residence, residence_request) == {
        "location": "Berlin",
    }


def test_extractor_prompt_has_generic_claim_level_provenance_and_profile_rules():
    prompt = " ".join((EXTRACT_SYSTEM_PROMPT + EXTRACT_PROMPT).lower().split())
    assert "every factual clause" in prompt
    assert "topically different turn" in prompt
    assert "current primary residence" in prompt
    assert "home country" in prompt
    assert "desired or former job" in prompt
    assert "caroline" not in prompt


def test_temporal_date_only_values_inherit_aware_conversation_timezone():
    anchor = _dt("2023-10-22T09:55:00+00:00")
    values = _temporal_kwargs(
        {
            "observed_at": "2023-10-22T09:55:00",
            "temporal_relation": "past",
            "event_start": "2023-10-20",
            "event_end": "2023-10-20",
            "valid_from": "2023-10-20",
        },
        observed_fallback=anchor,
    )

    assert values["observed_at"].tzinfo == timezone.utc
    assert values["event_start"] == _dt("2023-10-20T00:00:00+00:00")
    assert values["event_end"] == _dt("2023-10-20T23:59:59+00:00")
    assert datetime.fromtimestamp(
        values["event_start"].timestamp(), tz=timezone.utc,
    ).date().isoformat() == "2023-10-20"
    assert datetime.fromtimestamp(
        values["event_end"].timestamp(), tz=timezone.utc,
    ).date().isoformat() == "2023-10-20"


def test_reconcile_candidate_threshold_is_separate_and_high_recall():
    assert ReconcileRetrievalConfig().min_score == 0.3
    assert MemoryReconciler.HYBRID_MIN_SCORE == 0.3


def test_reconciler_promotes_explicit_state_transition_to_supersede():
    reconciler = object.__new__(MemoryReconciler)
    ops = reconciler._parse_ops(
        json.dumps(
            [
                {
                    "op": "UPDATE",
                    "memory_id": "old",
                    "content": "用户现在住在上海。",
                    "state_transition": True,
                    "source_indices": [0],
                    "observed_at": "2023-05-25T13:14:00",
                    "temporal_relation": "present",
                    "valid_from": "2023-05-25",
                }
            ],
            ensure_ascii=False,
        )
    )

    assert len(ops) == 1
    assert ops[0].op == "SUPERSEDE"
    assert ops[0].memory_id == "old"
    assert ops[0].source_indices == [0]
    assert ops[0].observed_at == "2023-05-25T13:14:00"


class _FakeLLMProvider:
    operations = []

    def __init__(self, config):
        self.config = config

    async def complete_messages(self, **kwargs):
        return SimpleNamespace(
            content=json.dumps(self.operations, ensure_ascii=False),
            prompt_tokens=10,
            completion_tokens=5,
        )


class _FakeSystem2Executor:
    def __init__(self, fail_edge=False):
        self.fail_edge = fail_edge
        self.created = 0
        self.executed = []
        self.tool_call_log = []

    async def execute(self, tool_name, args):
        self.executed.append((tool_name, dict(args)))
        if tool_name == "create_graph_node":
            self.created += 1
            result = {
                "success": True,
                "verified": True,
                "node_id": f"00000000-0000-0000-0000-{self.created:012d}",
            }
        elif tool_name == "add_edge":
            result = {
                "success": not self.fail_edge,
                "verified": not self.fail_edge,
            }
            if self.fail_edge:
                result["error"] = "postcondition failed"
        else:
            result = {"success": True}
        self.tool_call_log.append({"tool": tool_name, "args": args, "result": result})
        return json.dumps(result)


@pytest.mark.asyncio
async def test_system2_resolves_schema_refs_to_real_ids_and_propagates_failure(monkeypatch):
    from hy_memory.agent import llm_provider as llm_provider_module

    monkeypatch.setattr(llm_provider_module, "LLMProvider", _FakeLLMProvider)
    _FakeLLMProvider.operations = [
        {"op": "create_schema", "ref": "schema_1", "content": "A"},
        {"op": "create_schema", "ref": "schema_2", "content": "B"},
        {
            "op": "add_edge",
            "source_ref": "schema_1",
            "target_ref": "schema_2",
            "reason": "related",
        },
    ]
    config = SimpleNamespace(
        llm=SimpleNamespace(agent_max_tokens=4000, temperature=0.0)
    )

    ok_executor = _FakeSystem2Executor()
    ok = await run_system2_agent(
        {"facts": [{"content": "测试"}]}, ok_executor, config
    )
    edge_args = ok_executor.executed[-1][1]
    assert ok["success"] is True
    assert edge_args["source_id"] == ok["ref_map"]["schema_1"]
    assert edge_args["target_id"] == ok["ref_map"]["schema_2"]
    assert edge_args["source_id"].startswith("00000000-")

    failed_executor = _FakeSystem2Executor(fail_edge=True)
    failed = await run_system2_agent(
        {"facts": [{"content": "测试"}]}, failed_executor, config
    )
    assert failed["success"] is False
    assert len(failed["failed_operations"]) == 1
    assert failed["failed_operations"][0]["tool"] == "add_edge"


@pytest.mark.asyncio
async def test_kuzu_add_edge_rejects_zero_row_match():
    store = object.__new__(KuzuGraphStore)
    store._available = True

    async def no_rows(*args, **kwargs):
        return []

    store._run = no_rows
    assert await store.add_edge("missing-a", "missing-b", "RELATED_TO") is False


@pytest.mark.asyncio
async def test_kuzu_add_edge_real_database_is_verified_and_idempotent():
    with tempfile.TemporaryDirectory(
        prefix="hy-memory-kuzu-test-", dir=str(SOURCE_ROOT.parent)
    ) as temp_dir:
        temp_path = Path(temp_dir)
        config = SimpleNamespace(
            graph_store=SimpleNamespace(db_path=str(temp_path / "graph")),
            vector_store=SimpleNamespace(
                embedding_dims=4,
                persist_directory=str(temp_path / "vdb"),
            ),
            embedder=SimpleNamespace(embedding_dims=4),
        )
        store = KuzuGraphStore(config)
        await store.initialize()
        assert store._available is True
        try:
            for node_id in ("node-a", "node-b"):
                node = MemoryNode(
                    node_id=node_id,
                    user_id="u",
                    agent_id="a",
                    layer=MemoryLayer.L6_SCHEMA,
                    content=node_id,
                )
                node._graph_embedding = [0.1, 0.2, 0.3, 0.4]
                await store.upsert_memory_node(node)

            assert await store.update_node(
                "node-a", {"custom_json": json.dumps({"verified": True})}
            ) is True
            assert await store.update_node(
                "missing", {"custom_json": "{}"}
            ) is False
            assert await store.update_embedding(
                "node-a", beh_embedding=[0.4, 0.3, 0.2, 0.1]
            ) is True
            assert await store.update_embedding(
                "missing", beh_embedding=[0.4, 0.3, 0.2, 0.1]
            ) is False
            assert await store.add_edge("node-a", "node-b", "RELATED_TO") is True
            assert await store.add_edge("node-a", "node-b", "RELATED_TO") is True
            assert await store.add_edge("node-a", "missing", "RELATED_TO") is False
            rows = await store._run(
                "MATCH (:Memory)-[r:RELATED_TO]->(:Memory) RETURN count(r);"
            )
            assert rows[0][0] == 2

            intention = MemoryNode(
                node_id="intention",
                user_id="u",
                agent_id="a",
                layer=MemoryLayer.L7_INTENTION,
                content="用户计划在暑假研究收养机构。",
                memory_at=_dt("2023-05-25T13:14:00"),
                observed_at=_dt("2023-05-25T13:14:00"),
                temporal_relation="future",
                event_time_text="this summer",
                event_start=_dt("2023-06-01"),
                event_end=_dt("2023-08-31T23:59:59"),
                normalization_confidence=0.75,
                valid_from=_dt("2023-05-25"),
                valid_until=_dt("2023-08-31T23:59:59"),
            )
            intention._graph_embedding = [0.4, 0.3, 0.2, 0.1]
            await store.upsert_memory_node(intention)
            restored = await store.get_node("intention")
            assert restored.observed_at == _dt("2023-05-25T13:14:00")
            assert restored.temporal_relation == "future"
            assert restored.event_time_text == "this summer"
            assert restored.event_start == _dt("2023-06-01")
            assert restored.event_end == _dt("2023-08-31T23:59:59")
            assert restored.normalization_confidence == 0.75
        finally:
            await store.close()


class _TraceCache:
    def __init__(self):
        self.logs = []

    async def store_pipeline_log(self, **kwargs):
        self.logs.append(kwargs)


class _SweeperGraph:
    def __init__(self):
        self.nodes = [
            MemoryNode(
                node_id=f"n{i}",
                user_id="u",
                agent_id="a",
                layer=MemoryLayer.L6_SCHEMA,
                content=f"schema {i}",
            )
            for i in range(5)
        ]

    async def get_all_nodes(self, **kwargs):
        return self.nodes


@pytest.mark.asyncio
async def test_sweeper_cluster_failure_has_truthful_failure_summary():
    cache = _TraceCache()
    sweeper = CrossDomainSweeper(
        graph_store=_SweeperGraph(),
        embed_service=object(),
        llm_call=None,
        user_id="u",
        agent_id="a",
        request_id="req",
        cache=cache,
    )

    async def ensure(_basics):
        return 5

    async def scan(_basics):
        return [["n0", "n1"]], {"total_with_beh": 5, "pairs": []}

    async def fail_cluster(_cluster_ids, _basics):
        raise RuntimeError("graph write failed")

    sweeper._ensure_beh_embeddings = ensure
    sweeper._scan_collisions = scan
    sweeper._process_cluster = fail_cluster

    result = await sweeper.sweep()
    assert result["success"] is False
    assert result["stage"] == "cluster_processing"
    assert result["collisions"] == 1
    summary = json.loads(cache.logs[-1]["parsed"])
    assert summary["success"] is False
    assert summary["stage"] == "cluster_processing"


@pytest.mark.asyncio
async def test_sweeper_behavior_failure_is_not_swallowed():
    cache = _TraceCache()

    async def fail_llm(_prompt):
        raise RuntimeError("model unavailable")

    sweeper = CrossDomainSweeper(
        graph_store=_SweeperGraph(),
        embed_service=object(),
        llm_call=fail_llm,
        user_id="u",
        agent_id="a",
        request_id="req",
        cache=cache,
    )
    result = await sweeper.sweep()

    assert result["success"] is False
    assert result["stage"] == "behavior_embedding"
    assert result["new_beh_embeddings"] is None
    summary = json.loads(cache.logs[-1]["parsed"])
    assert summary["success"] is False
    assert "model unavailable" in summary["error"]


class _FakeEmbedService:
    async def embed_batch(self, contents):
        return [[float(i + 1), 0.5] for i, _ in enumerate(contents)]

    async def embed_queued(self, content):
        return [1.0, 0.5]


class _FakeVectorStore:
    def __init__(self, nodes):
        self.nodes = {node.node_id: MemoryNode.from_dict(node.to_dict()) for node in nodes}
        self.fail_updates_for = set()

    async def get_by_id(self, node_id):
        node = self.nodes.get(node_id)
        return MemoryNode.from_dict(node.to_dict()) if node else None

    async def upsert(self, node):
        self.nodes[node.node_id] = MemoryNode.from_dict(node.to_dict())
        return node.node_id

    async def update_payload(self, node_id, updates):
        if node_id in self.fail_updates_for:
            return False
        node = self.nodes.get(node_id)
        if node is None:
            return False
        payload = node.to_dict()
        payload.update(updates)
        self.nodes[node_id] = MemoryNode.from_dict(payload)
        return True

    async def delete(self, node_id):
        return self.nodes.pop(node_id, None) is not None


class _FakeReconciler:
    def __init__(self, op):
        self.op = op

    async def reconcile(self, **kwargs):
        return ReconcileResult(ops=[self.op], success=True)


def _writer_for(op):
    config = SimpleNamespace(
        recall=SimpleNamespace(entity_store_enabled=False),
    )
    writer = MemoryWriter(config=config, embed_service=_FakeEmbedService())
    writer._reconciler = _FakeReconciler(op)
    return writer


@pytest.mark.asyncio
async def test_supersede_closes_old_interval_and_verifies_bidirectional_chain():
    old = MemoryNode(
        node_id="old",
        user_id="u",
        agent_id="a",
        session_id="s",
        layer=MemoryLayer.L2_FACT,
        content="用户住在北京。",
        memory_at=_dt("2023-01-01T09:00:00"),
        observed_at=_dt("2023-01-01T09:00:00"),
        temporal_relation="present",
        valid_from=_dt("2023-01-01"),
        evidence_chain=["D1:1"],
    )
    store = _FakeVectorStore([old])
    op = ReconcileOp(
        op="SUPERSEDE",
        memory_id="old",
        content="用户现在住在上海。",
        layer="L2_FACT",
        observed_at="2023-05-25T13:14:00",
        temporal_relation="present",
        valid_from="2023-05-25",
        state_transition=True,
        source_indices=[0],
    )
    writer = _writer_for(op)
    request = WriteRequest(
        user_id="u",
        agent_id="a",
        session_id="s",
        memory_at=_dt("2023-05-25T13:14:00"),
    )

    stored_ids, error, counts, _ = await writer._reconcile_and_store(
        [op.content], [{
            "source_turn_ids": ["D2:2"],
            "source_turn_indices": [1],
        }], request, store, "req"
    )

    assert error is None
    assert counts["supersede"] == 1
    assert len(stored_ids) == 1
    new_id = stored_ids[0]
    old_after = await store.get_by_id("old")
    new_after = await store.get_by_id(new_id)
    assert old_after.status == MemoryStatus.SUPERSEDED
    assert old_after.is_latest is False
    assert old_after.valid_until == _dt("2023-05-25")
    assert old_after.superseded_by == [new_id]
    assert new_after.supersedes == ["old"]
    assert new_after.valid_from == _dt("2023-05-25")
    assert new_after.observed_at == _dt("2023-05-25T13:14:00")
    assert new_after.evidence_chain == ["D1:1", "D2:2"]
    assert new_after.source_turn_index == 1


@pytest.mark.asyncio
async def test_update_merges_old_and_new_turn_provenance_in_place():
    old = MemoryNode(
        node_id="old",
        user_id="u",
        agent_id="a",
        session_id="session_1",
        layer=MemoryLayer.L2_FACT,
        content="The user enjoys hiking.",
        source_session_id="session_1",
        source_turn_index=0,
        evidence_chain=["D1:1"],
    )
    store = _FakeVectorStore([old])
    op = ReconcileOp(
        op="UPDATE",
        memory_id="old",
        content="The user enjoys hiking and goes every weekend.",
        layer="L2_FACT",
        source_indices=[0],
    )
    writer = _writer_for(op)
    request = WriteRequest(user_id="u", agent_id="a", session_id="session_2")

    stored_ids, error, counts, _ = await writer._reconcile_and_store(
        [op.content], [{
            "source_turn_ids": ["D2:3"],
            "source_turn_indices": [2],
        }], request, store, "req"
    )

    assert error is None
    assert counts["update"] == 1
    assert stored_ids == ["old"]
    updated = await store.get_by_id("old")
    assert updated.evidence_chain == ["D1:1", "D2:3"]
    assert updated.source_session_id == "session_1"
    assert updated.source_turn_index == 0


@pytest.mark.asyncio
async def test_supersede_failure_rolls_back_new_head_and_propagates_error():
    old = MemoryNode(
        node_id="old",
        user_id="u",
        agent_id="a",
        layer=MemoryLayer.L2_FACT,
        content="old state",
        memory_at=_dt("2023-01-01"),
        valid_from=_dt("2023-01-01"),
    )
    store = _FakeVectorStore([old])
    store.fail_updates_for.add("old")
    op = ReconcileOp(
        op="SUPERSEDE",
        memory_id="old",
        content="new state",
        layer="L2_FACT",
        observed_at="2023-05-25T13:14:00",
        valid_from="2023-05-25",
    )
    writer = _writer_for(op)
    request = WriteRequest(
        user_id="u", agent_id="a", memory_at=_dt("2023-05-25T13:14:00")
    )

    stored_ids, error, counts, _ = await writer._reconcile_and_store(
        [op.content], [{}], request, store, "req"
    )

    assert stored_ids == []
    assert error and "old-node update returned false" in error
    assert counts["supersede"] == 0
    assert list(store.nodes) == ["old"]
    restored_old = await store.get_by_id("old")
    assert restored_old.status == MemoryStatus.ACTIVE
    assert restored_old.is_latest is True
    assert restored_old.superseded_by is None
