# -*- coding: utf-8 -*-
"""
SharedRuntime — 多 client 部署（例如多租户 server）的共享基础设施。

## 背景

`HyMemoryClient` 默认每个实例自带：
  - 一个 `_LoopThread`（持久 event loop）
  - 一份 `MysqlCache` / `SqliteCache` pool
  - 一份 pipeline observability hook 包装

而 `MetricsCollector` 是进程级 singleton。当 app 因为租户间 embedder dims
不同而**起多个 client** 时：

```
Client-A._LoopThread (loop_A)   →  Pool-A bound to loop_A
Client-B._LoopThread (loop_B)   →  Pool-B bound to loop_B

MetricsCollector (singleton)
  ├── bind_cache(Pool-A)  ← Client-A init 时
  ├── bind_cache(Pool-B)  ← Client-B init 时（覆盖！）
  └── flush coroutine    ← ensure_future on loop_A，但访问 Pool-B
                          → 跨 loop 用 aiomysql → 连接污染 → KeyError
```

## 设计

把"进程级资源"显式抽出来：

| 归属 SharedRuntime（一进程一份） | 归属每个 HyMemoryClient（per tenant） |
|---|---|
| `_LoopThread`（共享 loop）         | `_embed_service`（不同 dims）         |
| `MysqlCache` / `SqliteCache` pool | `_vector_store`（不同 collection）     |
| `MetricsCollector` 单例 + 唯一 bind | `_graph_store`                        |
| pipeline observability hook       | `_history_store` / `_coding_store`    |
|                                   | `_registry` + 所有 pipeline           |

## 使用

```python
from hy_memory import HyMemoryClient, SharedRuntime, MemoryConfig

# App 启动时一次性创建
base_config = MemoryConfig.from_env()  # 提供 cache 配置
runtime = await SharedRuntime.create(base_config)

# 每个租户一个 lightweight client，共用 runtime
clients = {
    "default":      HyMemoryClient(default_config,       runtime=runtime),
    "hunyuan_test": HyMemoryClient(hunyuan_test_config,  runtime=runtime),
}

# App 退出
for c in clients.values():
    c.close()              # 只关 per-tenant 资源
await runtime.aclose()     # 关 shared 基础设施
```

## 兼容性

不传 `runtime=` 的旧用法**完全不变**，client 自己持有所有资源（solo 模式）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import MemoryConfig
    from .data.cache_base import CacheBase

logger = logging.getLogger(__name__)


class SharedRuntime:
    """
    多 client 共享的基础设施。

    用 `await SharedRuntime.create(config)` 构造（不要直接 `SharedRuntime(config)`，
    需要 async init 才能建 cache pool 并 bind metrics）。
    """

    def __init__(self, config: "MemoryConfig"):
        # 配置只用来建 cache 和决定 loop 策略；vector_store / embedder 等
        # 由各个 client 自己管。
        self._config = config
        self._loop_thread = None  # type: Optional[object]  # _LoopThread
        self._cache = None        # type: Optional[CacheBase]
        self._metrics = None      # type: Optional[object]  # MetricsCollector
        self._observability_installed = False
        self._closed = False

    # ------------------------------------------------------------
    # 工厂 / 生命周期
    # ------------------------------------------------------------

    @classmethod
    async def create(cls, config: "MemoryConfig") -> "SharedRuntime":
        """
        创建并初始化一个 SharedRuntime。

        会立即：
          1. 起一个 _LoopThread（持久 event loop）
          2. 在该 loop 上 init cache pool
          3. 把 cache 绑给 MetricsCollector 单例
          4. 在该 loop 上启动 metrics 后台 flush + cleanup
          5. 把 pipeline observability hook 安装到 cache 上

        步骤 2-5 都在同一个 loop（即 _LoopThread 的 loop）上执行，
        保证后续 metrics flush 用的就是这个 loop 上的 pool，不会跨 loop。
        """
        rt = cls(config)
        await rt._initialize()
        return rt

    async def _initialize(self) -> None:
        # 注意：本方法可能从外部 loop 调到（如 app 的 uvicorn loop），
        # 但内部所有"绑定到 loop"的操作必须在 _LoopThread 的 loop 上完成。
        # 因此把真正的 init 委托过去。
        from .client import _LoopThread

        self._loop_thread = _LoopThread()
        # 在 _LoopThread 的 loop 上做所有需要 loop affinity 的 init
        await self._loop_thread.run_async(self._init_on_loop())

    async def _init_on_loop(self) -> None:
        """在 _LoopThread 的 loop 上执行的初始化。"""
        # 1. cache pool（pool 会绑定到当前 loop）
        from .data.cache import create_cache

        self._cache = create_cache(self._config)
        await self._cache.initialize()
        logger.info(
            f"[runtime] Cache ready (backend={self._config.cache.backend})"
        )

        # 2. MetricsCollector 单例：唯一一次 bind_cache + start_background_tasks
        #    flush 协程从此固定在本 loop 上跑，永远不会跨 loop。
        from .metrics import MetricsCollector

        self._metrics = MetricsCollector.get()
        self._metrics.bind_cache(self._cache)
        await self._metrics.start_background_tasks()
        logger.info("[runtime] MetricsCollector bound to shared cache")

        # 3. pipeline observability hook — 包一次就够，所有 client 共用
        self._install_pipeline_observability()
        logger.info("[runtime] SharedRuntime ready")

    def _install_pipeline_observability(self) -> None:
        """与 HyMemoryClient._install_pipeline_observability_hooks 等价，
        但只装一次。重复装会导致每个 store_pipeline_log 调用被多层包装。"""
        if self._observability_installed:
            return
        self._observability_installed = True

        from .utils.pipeline_observability import (
            is_pipeline_trace_enabled,
            resolve_pipeline_log_dir,
        )
        from .utils.pipeline_log_writer import PipelineLogWriter

        log_dir = resolve_pipeline_log_dir()
        log_writer = PipelineLogWriter(log_dir)
        trace_enabled = is_pipeline_trace_enabled()
        logger.info(
            f"[runtime] Pipeline log (JSONL) enabled: {log_dir}/<subdir>/<date>.log "
            f"(subdir from user_id prefix before '__', else default) ; "
            f"pipeline trace (DB)={'on' if trace_enabled else 'off'}"
        )

        original_store = self._cache.store_pipeline_log

        from collections import OrderedDict as _OrderedDict
        _last_event: "_OrderedDict[str, float]" = _OrderedDict()
        _LAST_EVENT_MAX = 4096
        _TERMINAL_STEPS = {"READ_SUMMARY", "READ_CODING_RESULT", "WRITE_TIMELINE"}
        _STEP_TIMING_KEY = {
            "READ_EMBED_QUERY": "embed_ms", "READ_CODING_EMBED": "embed_ms",
            "READ_RECALL_VEC": "vdb_search_ms", "READ_RECALL_PROFILE": "vdb_search_ms",
            "READ_RECALL_TAG": "vdb_search_ms", "READ_BM25": "vdb_search_ms",
            "READ_TAG_MATCH": "vdb_search_ms", "READ_CODING_VDB": "vdb_search_ms",
            "READ_KEYWORD_EMBED": "embed_ms",
            "READ_SCENE_JUDGE": "llm_ms",
            "EXTRACT": "llm_ms", "RECONCILE": "llm_ms",
            "SEARCH_QUERY": "llm_ms", "SUMMARY": "llm_ms",
        }

        def _inject_op_ms(parsed_str: str, step: str, op_ms: float) -> str:
            key = _STEP_TIMING_KEY.get(step)
            if not key or not op_ms:
                return parsed_str
            try:
                obj = json.loads(parsed_str) if parsed_str else {}
                if isinstance(obj, dict):
                    obj[key] = round(float(op_ms), 2)
                    return json.dumps(obj, ensure_ascii=False, default=str)
            except Exception:
                pass
            return parsed_str

        async def _store_pipeline_observability(
            *,
            request_id: str = "",
            user_id: str = "",
            agent_id: str = "",
            step: str = "",
            prompt: str = "",
            response: str = "",
            parsed: str = "",
            memory_ids=None,
            elapsed_ms: float = 0.0,
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
            total_tokens: int = 0,
            session_id: str = "",
            **kwargs,
        ):
            import time as _time
            _now = _time.time()
            event_at_ms = int(_now * 1000)

            parsed = _inject_op_ms(parsed, step, elapsed_ms)

            _prev = _last_event.get(request_id)
            step_elapsed_ms = round((_now - _prev) * 1000, 2) if _prev is not None else 0.0
            if step in _TERMINAL_STEPS:
                _last_event.pop(request_id, None)
            else:
                _last_event[request_id] = _now
                _last_event.move_to_end(request_id)
                while len(_last_event) > _LAST_EVENT_MAX:
                    _last_event.popitem(last=False)

            try:
                log_writer.write_step(
                    request_id=request_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    step=step,
                    prompt=prompt,
                    response=response,
                    parsed=parsed,
                    memory_ids=memory_ids,
                    elapsed_ms=step_elapsed_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    event_at_ms=event_at_ms,
                )
            except Exception:
                pass

            if not trace_enabled:
                return True

            return await original_store(
                request_id=request_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                step=step,
                prompt=prompt,
                response=response,
                parsed=parsed,
                memory_ids=memory_ids,
                elapsed_ms=step_elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                event_at_ms=event_at_ms,
                **kwargs,
            )

        self._cache.store_pipeline_log = _store_pipeline_observability

    # ------------------------------------------------------------
    # 暴露给 HyMemoryClient 的属性（read-only）
    # ------------------------------------------------------------

    @property
    def loop_thread(self):
        """共享的 _LoopThread 实例。"""
        return self._loop_thread

    @property
    def cache(self):
        """共享的 cache（已 initialize 完毕）。"""
        return self._cache

    @property
    def metrics(self):
        """MetricsCollector 单例（已 bind_cache 到 self.cache）。"""
        return self._metrics

    @property
    def config(self) -> "MemoryConfig":
        """构造 runtime 时用的 base config（client 不必跟它一致，仅 cache 配置须一致）。"""
        return self._config

    # ------------------------------------------------------------
    # 关闭
    # ------------------------------------------------------------

    async def aclose(self) -> None:
        """异步关闭 shared 资源（cache + loop）。"""
        if self._closed:
            return
        self._closed = True

        if self._loop_thread is not None:
            try:
                # metrics 与 cache 都绑定在 _LoopThread 上，必须先停 metrics 再关 cache。
                await self._loop_thread.run_async(self._aclose_inner())
            except Exception as e:
                logger.warning(f"[runtime] async close inner failed: {e}")
        self._cache = None

        if self._loop_thread is not None:
            try:
                self._loop_thread.stop()
            except Exception as e:
                logger.warning(f"[runtime] loop_thread stop failed: {e}")
            self._loop_thread = None

        logger.info("[runtime] SharedRuntime closed")

    def close(self) -> None:
        """同步关闭，方便从非 async 场景调用。"""
        if self._closed:
            return
        if self._loop_thread is not None:
            try:
                # cache.close 必须在它绑定的 loop 上跑
                self._loop_thread.run(self._aclose_inner())
            except Exception as e:
                logger.warning(f"[runtime] sync close inner failed: {e}")
            try:
                self._loop_thread.stop()
            except Exception as e:
                logger.warning(f"[runtime] loop_thread stop failed: {e}")
        self._loop_thread = None
        self._cache = None
        self._closed = True
        logger.info("[runtime] SharedRuntime closed")

    async def _aclose_inner(self) -> None:
        """在 loop 内执行的关闭流程（仅 close cache，不停 loop —— stop 由外面来做）。"""
        if self._metrics is not None:
            try:
                await self._metrics.stop_background_tasks()
            except Exception as e:
                logger.warning(f"[runtime] metrics close failed: {e}")
        if self._cache is not None:
            try:
                await self._cache.close()
            except Exception as e:
                logger.warning(f"[runtime] cache close failed: {e}")
