"""Hy-Memory Ultra 端到端冒烟测试。

默认执行三次历史会话写入、一次 System 2 digest、两次召回，并把结果写入
results/。使用 --init-only 时只验证 Chroma/Kuzu/SQLite 初始化，不调用模型。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Hy-Memory Ultra smoke test")
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize local Ultra backends; do not call LLM/Embedding APIs.",
    )
    return parser.parse_args()


def configure_environment(init_only: bool) -> None:
    load_dotenv(ENV_FILE, override=False)

    runtime_dir = PROJECT_ROOT / ".runtime" / "ultra"
    os.environ.setdefault("MEMORY_MODE", "ultra")
    os.environ.setdefault("MEMORY_VECTOR_STORE", "chroma")
    os.environ.setdefault("MEMORY_COLLECTION_NAME", "hy_memory_ultra")
    os.environ.setdefault("MEMORY_DATA_DIR", str(runtime_dir))
    os.environ.setdefault("MEMORY_PERSIST_DIR", str(runtime_dir / "chroma"))
    os.environ.setdefault("MEMORY_GRAPH_PROVIDER", "kuzu")
    os.environ.setdefault("MEMORY_GRAPH_DB_PATH", str(runtime_dir / "kuzu"))
    os.environ.setdefault("MEMORY_CACHE_BACKEND", "sqlite")
    os.environ.setdefault("MEMORY_EMBEDDING_DIMS", "1024")

    if init_only:
        # Client 初始化不会发出模型请求，以下值只用于通过配置构造。
        os.environ.setdefault("MEMORY_LLM_MODEL", "init-only")
        os.environ.setdefault("MEMORY_LLM_API_KEY", "init-only")
        os.environ.setdefault("MEMORY_LLM_BASE_URL", "http://127.0.0.1:9/v1")
        os.environ.setdefault("MEMORY_EMBEDDER_MODEL", "init-only")
        os.environ.setdefault("MEMORY_EMBEDDER_API_KEY", "init-only")
        os.environ.setdefault("MEMORY_EMBEDDER_BASE_URL", "http://127.0.0.1:9/v1")
        return

    required = (
        "MEMORY_LLM_MODEL",
        "MEMORY_LLM_API_KEY",
        "MEMORY_LLM_BASE_URL",
        "MEMORY_EMBEDDER_MODEL",
        "MEMORY_EMBEDDER_API_KEY",
        "MEMORY_EMBEDDER_BASE_URL",
        "MEMORY_EMBEDDING_DIMS",
    )
    missing = []
    for name in required:
        value = os.getenv(name, "").strip()
        if not value or "REPLACE_WITH" in value or "YOUR_WORKSPACE_ID" in value:
            missing.append(name)

    if missing:
        print("配置未完成：" + ", ".join(missing), file=sys.stderr)
        print("请编辑项目根目录的 .env，填写真实密钥和接口地址。", file=sys.stderr)
        raise SystemExit(2)


def compact(value: Any) -> Any:
    """Convert results to JSON-safe values without leaking configuration secrets."""
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
    """Build an Ultra client and disable costly DeepSeek thinking for extraction."""
    from hy_memory import HyMemoryClient
    from hy_memory.config import MemoryConfig

    config = MemoryConfig()
    if (config.llm.model or "").lower().startswith("deepseek-v4"):
        # V4 defaults to thinking mode. Memory extraction is structured JSON work;
        # non-thinking mode is faster, cheaper, and avoids unused reasoning tokens.
        config.llm.extra_body = {"thinking": {"type": "disabled"}}
    return HyMemoryClient(config=config, mode="ultra")


def run_init_only() -> None:
    client = build_client()
    try:
        print(json.dumps({"initialized": True, "mode": client.mode}, ensure_ascii=False))
    finally:
        client.close()


def run_end_to_end() -> None:
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    user_id = f"resume-demo-{run_stamp}"
    agent_id = "ultra-smoke"
    sessions = [
        (
            "profile-2026-06",
            datetime(2026, 6, 3, tzinfo=timezone.utc),
            [
                {"role": "user", "content": "我住在杭州，是后端开发工程师，平时特别喜欢川菜和重辣口味。"},
                {"role": "assistant", "content": "记住了：你在杭州从事后端开发，并偏爱重辣川菜。"},
            ],
        ),
        (
            "schedule-2026-07",
            datetime(2026, 7, 10, tzinfo=timezone.utc),
            [
                {"role": "user", "content": "我正在准备秋招，每周三和周六晚上固定刷算法题，不要在这两个晚上安排娱乐。"},
                {"role": "assistant", "content": "好的，周三和周六晚上优先保留给算法训练。"},
            ],
        ),
        (
            "preference-update-2026-08",
            datetime(2026, 8, 8, tzinfo=timezone.utc),
            [
                {"role": "user", "content": "体检后医生让我少吃辣，所以我现在不再选重辣川菜，外出吃饭优先清淡粤菜。"},
                {"role": "assistant", "content": "明白，以后饮食建议以清淡粤菜为主，并避免重辣。"},
            ],
        ),
    ]

    client = build_client()
    try:
        writes = []
        for session_id, memory_at, messages in sessions:
            result = client.add(
                messages,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_at=memory_at,
            )
            writes.append(compact(result))
            if isinstance(result, dict) and result.get("success") is False:
                raise RuntimeError(f"write failed for {session_id}: {result}")

        digest = compact(client.digest(user_id=user_id, agent_id=agent_id))
        food_recall = compact(
            client.search(
                "用户当前喜欢什么口味，聚餐应该选什么菜？",
                user_ids=[user_id],
                agent_ids=[agent_id],
                limit=10,
                min_score=0.2,
            )
        )
        schedule_recall = compact(
            client.search(
                "用户每周哪两个晚上固定学习，不能安排娱乐？",
                user_ids=[user_id],
                agent_ids=[agent_id],
                limit=10,
                min_score=0.2,
            )
        )
        memories = compact(client.list_memories(user_id=user_id, agent_id=agent_id, limit=100))

        report = {
            "run_id": run_stamp,
            "user_id": user_id,
            "mode": client.mode,
            "models": {
                "llm": os.getenv("MEMORY_LLM_MODEL"),
                "embedder": os.getenv("MEMORY_EMBEDDER_MODEL"),
                "embedding_dims": os.getenv("MEMORY_EMBEDDING_DIMS"),
            },
            "writes": writes,
            "digest": digest,
            "recall": {
                "food_preference": food_recall,
                "study_schedule": schedule_recall,
            },
            "memories": memories,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result_path = RESULTS_DIR / f"ultra-smoke-{run_stamp}.json"
        result_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Ultra 端到端测试完成：{result_path}")
    finally:
        client.close()


def main() -> None:
    args = parse_args()
    configure_environment(args.init_only)
    if args.init_only:
        run_init_only()
    else:
        run_end_to_end()


if __name__ == "__main__":
    main()
