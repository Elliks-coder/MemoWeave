import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "hy-memory-1.2.21"
sys.path.insert(0, str(SOURCE_ROOT))

from hy_memory.agent.reconciler import ReconcileOp
from hy_memory.data.graph_store_kuzu import KuzuGraphStore
from hy_memory.models.graph_memory import sanitize_relation_candidates
from hy_memory.models.memory import MemoryLayer, MemoryNode
from hy_memory.pipelines._retrieval.graphrag import (
    collect_relations_for_op,
    expand_graph_evidence,
)
from hy_memory.pipelines.base import ReadRequest
from hy_memory.pipelines.base import WriteRequest
from hy_memory.pipelines.reader_legacy import LegacyReadPipeline
from hy_memory.pipelines.writer import MemoryWriter


def _relation(subject, predicate, object_name, *, confidence=0.95):
    return {
        "subject": subject,
        "subject_type": "person",
        "predicate": predicate,
        "object": object_name,
        "object_type": "person" if object_name == "小林" else "place",
        "confidence": confidence,
        "evidence_type": "explicit",
    }


def test_relation_candidates_are_conservative_and_require_provenance():
    raw = [
        _relation("用户", "COLLEAGUE_OF", "小林"),
        {**_relation("用户", "FRIEND_OF", "小林"), "evidence_type": "inferred"},
        _relation("用户", "LIKES", "咖啡", confidence=0.4),
        _relation("她", "LIVES_IN", "杭州"),
        _relation("用户", "UNKNOWN_PREDICATE", "小林"),
    ]

    accepted = sanitize_relation_candidates(
        raw,
        source_turn_ids=["D1:turn:2"],
        min_confidence=0.8,
    )

    assert len(accepted) == 1
    assert accepted[0]["subject_normalized"] == "__user__"
    assert accepted[0]["predicate"] == "COLLEAGUE_OF"
    assert accepted[0]["object_normalized"] == "小林"
    assert accepted[0]["source_turn_ids"] == ["D1:turn:2"]
    assert sanitize_relation_candidates(raw[:1], source_turn_ids=[]) == []


def test_reconciler_source_indices_govern_relation_publication():
    metas = [
        {
            "content": "小林是用户的同事。",
            "source_turn_ids": ["D1:turn:1"],
            "relations": [_relation("用户", "COLLEAGUE_OF", "小林")],
        },
        {
            "content": "阿杰是用户的朋友。",
            "source_turn_ids": ["D2:turn:4"],
            "relations": [_relation("用户", "FRIEND_OF", "阿杰")],
        },
    ]
    op = ReconcileOp(op="ADD", content=metas[1]["content"], source_indices=[1])

    relations = collect_relations_for_op(op, metas, min_confidence=0.8)

    assert [(item["predicate"], item["object"]) for item in relations] == [
        ("FRIEND_OF", "阿杰")
    ]
    assert relations[0]["source_turn_ids"] == ["D2:turn:4"]


class _NodeStore:
    def __init__(self, nodes):
        self.nodes = {node.node_id: node for node in nodes}

    async def get_by_id(self, node_id):
        return self.nodes.get(node_id)


class _ReaderVectorStore(_NodeStore):
    def __init__(self, anchor, graph_fact):
        super().__init__([anchor, graph_fact])
        self.anchor = anchor

    async def initialize(self):
        return None

    async def search(self, **kwargs):
        return [{"node_id": self.anchor.node_id, "node": self.anchor, "score": 0.9}]


class _ReaderGraphStore:
    async def expand_fact_relations(self, **kwargs):
        assert kwargs["anchor_fact_ids"] == ["anchor-fact"]
        assert kwargs["tenant_keys"] == ["u1::a1"]
        return [{
            "fact_id": "graph-fact",
            "anchor_fact_id": "anchor-fact",
            "origin": "vector_fact",
            "hop": 1,
            "confidence": 0.95,
            "path": [{
                "subject": "小林",
                "predicate": "COLLEAGUE_OF",
                "object": "用户",
                "fact_id": "graph-fact",
            }],
        }]


class _ReaderEmbedder:
    async def embed(self, query):
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_batch(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _WriterVectorStore:
    def __init__(self):
        self.nodes = {}

    async def upsert(self, node):
        self.nodes[node.node_id] = node
        return node.node_id


class _WriterGraphStore:
    def __init__(self):
        self.calls = []

    async def publish_fact_relations(
        self, fact_node, relations, supersedes_fact_ids=None,
    ):
        self.calls.append((fact_node, relations, supersedes_fact_ids or []))
        return {
            "available": True,
            "success": True,
            "published": len(relations),
            "rejected": 0,
            "errors": [],
        }


@pytest.mark.asyncio
async def test_writer_publishes_only_after_l2_store_succeeds():
    vector_store = _WriterVectorStore()
    graph_store = _WriterGraphStore()
    config = SimpleNamespace(
        recall=SimpleNamespace(entity_store_enabled=False),
        graph_store=SimpleNamespace(
            graphrag_enabled=True,
            graphrag_min_confidence=0.8,
        ),
    )
    writer = MemoryWriter(
        config=config,
        embed_service=_ReaderEmbedder(),
        vector_store=vector_store,
        graph_store=graph_store,
    )
    request = WriteRequest(
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        memory_at=datetime.fromisoformat("2026-01-01T10:00:00"),
    )
    meta = {
        "content": "小林是用户的同事。",
        "layer": "L2_FACT",
        "source_turn_ids": ["D1:turn:1"],
        "source_turn_indices": [1],
        "relations": [_relation("用户", "COLLEAGUE_OF", "小林")],
    }

    stored, error, counts, _ = await writer._direct_store(
        [meta], request, vector_store, "req-1"
    )

    assert error is None
    assert len(stored) == 1
    assert stored[0] in vector_store.nodes
    assert counts["graph_published"] == 1
    assert counts["graph_failures"] == 0
    assert len(graph_store.calls) == 1
    fact_node, relations, supersedes = graph_store.calls[0]
    assert fact_node.node_id == stored[0]
    assert relations[0]["predicate"] == "COLLEAGUE_OF"
    assert relations[0]["source_turn_ids"] == ["D1:turn:1"]
    assert supersedes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("shadow_mode, expected_count", [(True, 1), (False, 2)])
async def test_legacy_reader_graph_expansion_respects_shadow_mode(
    shadow_mode, expected_count,
):
    anchor = MemoryNode(
        node_id="anchor-fact",
        user_id="u1",
        agent_id="a1",
        layer=MemoryLayer.L2_FACT,
        content="小林搬到了杭州。",
    )
    graph_fact = MemoryNode(
        node_id="graph-fact",
        user_id="u1",
        agent_id="a1",
        layer=MemoryLayer.L2_FACT,
        content="小林是用户的同事。",
        evidence_chain=["D1:turn:1"],
    )
    config = SimpleNamespace(
        recall=SimpleNamespace(strength_enabled=False),
        graph_store=SimpleNamespace(
            graphrag_enabled=True,
            graphrag_shadow_mode=shadow_mode,
            graphrag_max_hops=2,
            graphrag_max_facts=5,
            graphrag_max_degree=25,
        ),
    )
    reader = LegacyReadPipeline(
        config=config,
        embed_service=_ReaderEmbedder(),
        vector_store=_ReaderVectorStore(anchor, graph_fact),
        graph_store=_ReaderGraphStore(),
    )
    await reader.initialize()

    response = await reader.read(ReadRequest(
        query="用户哪位同事搬去了杭州？",
        user_id="u1",
        agent_id="a1",
        profile_limit=0,
        intention_limit=0,
        limit=10,
    ))

    assert response.success is True
    assert response.total_found == expected_count
    assert response.extra["graphrag"]["evidence_count"] == 1
    assert response.extra["graphrag"]["added_count"] == (0 if shadow_mode else 1)
    if shadow_mode:
        assert {item["memory_id"] for item in response.memories} == {"anchor-fact"}
    else:
        graph_entry = next(
            item for item in response.memories if item["memory_id"] == "graph-fact"
        )
        assert graph_entry["retrieval_source"] == "entity_fact_graph"
        assert graph_entry["graph_hop"] == 1
        assert graph_entry["evidence_chain"] == ["D1:turn:1"]


@pytest.mark.asyncio
async def test_kuzu_entity_fact_graph_publishes_expands_and_evolves():
    with tempfile.TemporaryDirectory(
        prefix="memo-graphrag-kuzu-", dir=str(SOURCE_ROOT.parent)
    ) as temp_dir:
        temp_path = Path(temp_dir)
        config = SimpleNamespace(
            graph_store=SimpleNamespace(
                db_path=str(temp_path / "graph"),
                graphrag_enabled=True,
            ),
            vector_store=SimpleNamespace(
                embedding_dims=4,
                persist_directory=str(temp_path / "vdb"),
            ),
            embedder=SimpleNamespace(embedding_dims=4),
        )
        store = KuzuGraphStore(config)
        await store.initialize()
        assert store._available is True
        assert store._graphrag_available is True

        colleague_fact = MemoryNode(
            node_id="fact-colleague",
            user_id="u1",
            agent_id="a1",
            session_id="s1",
            layer=MemoryLayer.L2_FACT,
            content="小林是用户的同事。",
            memory_at=datetime.fromisoformat("2026-01-01T10:00:00"),
            valid_from=datetime.fromisoformat("2026-01-01T10:00:00"),
            source_session_id="s1",
            evidence_chain=["D1:turn:1"],
        )
        hangzhou_fact = MemoryNode(
            node_id="fact-hangzhou",
            user_id="u1",
            agent_id="a1",
            session_id="s2",
            layer=MemoryLayer.L2_FACT,
            content="小林已经搬到杭州。",
            memory_at=datetime.fromisoformat("2026-02-01T10:00:00"),
            valid_from=datetime.fromisoformat("2026-02-01T10:00:00"),
            source_session_id="s2",
            evidence_chain=["D2:turn:3"],
        )

        colleague_rel = sanitize_relation_candidates(
            [_relation("用户", "COLLEAGUE_OF", "小林")],
            source_turn_ids=colleague_fact.evidence_chain,
        )
        lives_rel = sanitize_relation_candidates(
            [_relation("小林", "LIVES_IN", "杭州")],
            source_turn_ids=hangzhou_fact.evidence_chain,
        )

        first = await store.publish_fact_relations(colleague_fact, colleague_rel)
        second = await store.publish_fact_relations(hangzhou_fact, lives_rel)
        duplicate = await store.publish_fact_relations(hangzhou_fact, lives_rel)
        assert first["success"] is True and first["published"] == 1
        assert second["success"] is True and second["published"] == 1
        assert duplicate["success"] is True and duplicate["published"] == 1

        assertions = await store._run("MATCH (a:Assertion) RETURN count(a);")
        assert assertions[0][0] == 2

        expanded = await store.expand_fact_relations(
            anchor_fact_ids=[hangzhou_fact.node_id],
            query="用户哪位同事搬去了杭州？",
            tenant_keys=["u1::a1"],
            user_ids=["u1"],
            max_hops=2,
            max_facts=5,
            max_degree=25,
        )
        assert colleague_fact.node_id in {item["fact_id"] for item in expanded}
        colleague_path = next(
            item for item in expanded if item["fact_id"] == colleague_fact.node_id
        )
        assert colleague_path["hop"] <= 2
        assert colleague_path["path"]

        # Storage API fails closed when a future caller forgets tenant scope.
        unscoped = await store.expand_fact_relations(
            anchor_fact_ids=[hangzhou_fact.node_id],
            query="用户哪位同事搬去了杭州？",
            tenant_keys=[],
            user_ids=[],
            max_hops=2,
            max_facts=5,
            max_degree=25,
        )
        assert unscoped == []

        # 检索层必须回到 VDB 获取 L2 原文，而不是把裸图当作答案。
        vector_store = _NodeStore([colleague_fact, hangzhou_fact])
        hits, summary = await expand_graph_evidence(
            graph_store=store,
            vector_store=vector_store,
            query="用户哪位同事搬去了杭州？",
            normal_results=[{
                "node_id": hangzhou_fact.node_id,
                "node": hangzhou_fact,
                "score": 0.9,
            }],
            tenant_keys=["u1::a1"],
            user_ids=["u1"],
            as_of=None,
            max_hops=2,
            max_facts=5,
            max_degree=25,
        )
        assert [hit["node_id"] for hit in hits] == [colleague_fact.node_id]
        assert hits[0]["node"].content == "小林是用户的同事。"
        assert hits[0]["graph_path"]
        assert summary["evidence_count"] == 1

        suzhou_fact = MemoryNode(
            node_id="fact-suzhou",
            user_id="u1",
            agent_id="a1",
            session_id="s3",
            layer=MemoryLayer.L2_FACT,
            content="小林现在居住在苏州。",
            memory_at=datetime.fromisoformat("2026-03-01T10:00:00"),
            valid_from=datetime.fromisoformat("2026-03-01T10:00:00"),
            source_session_id="s3",
            evidence_chain=["D3:turn:2"],
        )
        suzhou_rel = sanitize_relation_candidates(
            [_relation("小林", "LIVES_IN", "苏州")],
            source_turn_ids=suzhou_fact.evidence_chain,
        )
        evolved = await store.publish_fact_relations(
            suzhou_fact,
            suzhou_rel,
            supersedes_fact_ids=[hangzhou_fact.node_id],
        )
        assert evolved["success"] is True
        old_status = await store._run(
            """
            MATCH (a:Assertion)-[:ASSERTION_SUPPORTED_BY]->
                  (f:VdbRef {node_id: $fid})
            RETURN a.status, a.valid_until;
            """,
            {"fid": hangzhou_fact.node_id},
        )
        assert old_status[0][0] == "superseded"
        supersede_edges = await store._run(
            "MATCH (:Assertion)-[r:ASSERTION_SUPERSEDES]->(:Assertion) RETURN count(r);"
        )
        assert supersede_edges[0][0] == 1

        stats = await store.get_stats()
        assert stats["entity_nodes"] >= 4
        assert stats["assertion_nodes"] == 3
        assert stats["active_assertions"] == 2
        assert stats["episode_refs"] == 3

        await store.delete_by_metadata("u1", agent_id="a1")
        purged = await store.get_stats()
        assert purged["entity_nodes"] == 0
        assert purged["assertion_nodes"] == 0
        assert purged["episode_refs"] == 0

        await store.close()
