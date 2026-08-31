# -*- coding: utf-8 -*-
"""
CodingCuratorWriter — 渐进式披露的 coding memory 写入 agent。

write() 接口签名与 legacy CodingWriter 完全一致：
    async def write(messages, *, user_id, agent_id, workspace_id, branch,
                    session_id, request_id) -> Dict[str, Any]

返回结构：
    {"success": True, "scene": "productivity", "request_id": ...,
     "ops": [...], "memory_ids": [...],
     "iterations": N, "truncated": bool, "elapsed_ms": ...}

详见 /root/.claude-internal/plans/squishy-leaping-orbit.md
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...pipelines.base import ChatMessage
from ..preproc import truncate_messages, extract_files
from ..store import CodingMemoryStore
from .prompts import (
    SYSTEM_PROMPT,
    render_initial_user_prompt,
    render_messages_compact,
)
from .tools import (
    TOOL_SCHEMAS,
    CuratorContext,
    ToolResult,
    dispatch_tool,
)

logger = logging.getLogger(__name__)


DEFAULT_MAX_ITERATIONS = 15
DEFAULT_MAX_TOKENS = 2000
DEFAULT_LAST_K_USER_QUERIES = 5
DEFAULT_EXISTING_INDEX_LIMIT = 30


@dataclass
class AgentResult:
    ops: List[Any]               # ReconcileOp（避免循环 import）
    memory_ids: List[str]
    iterations: int
    truncated: bool
    error: Optional[str] = None
    # ── metrics ──
    tool_call_counts: Dict[str, int] = None  # tool_name → 调用次数
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # 每轮发出的 prompt 内部分桶字符数（跨轮累加）
    # buckets: system / initial_user / tool_schemas / agent_text /
    #          agent_tool_calls / tool_result
    prompt_chars_by_bucket: Dict[str, int] = None

    def __post_init__(self):
        if self.tool_call_counts is None:
            self.tool_call_counts = {}
        if self.prompt_chars_by_bucket is None:
            self.prompt_chars_by_bucket = {}


class CodingCuratorWriter:
    """
    与 legacy CodingWriter 接口一致的 agent-based writer。
    通过 OpenAI function-calling 让 LLM 自主决定何时取详情、何时写动作。
    """

    def __init__(
        self,
        store: CodingMemoryStore,
        llm_provider,
        embed_service,
        *,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        last_k_user_queries: int = DEFAULT_LAST_K_USER_QUERIES,
        existing_index_limit: int = DEFAULT_EXISTING_INDEX_LIMIT,
    ):
        self.store = store
        self.llm = llm_provider
        self.embed_service = embed_service
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        # None → 复用 LLMConfig.temperature（避免硬编码触发 kimi/deepseek 的温度硬约束）
        if temperature is None:
            temperature = getattr(getattr(llm_provider, "_llm_config", None), "temperature", 0.1)
        self.temperature = temperature
        self.last_k_user_queries = last_k_user_queries
        self.existing_index_limit = existing_index_limit

    async def write(
        self,
        messages: List[ChatMessage],
        *,
        user_id: str,
        agent_id: str = "default_agent",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        rid = request_id or str(uuid.uuid4())

        # ── 1. preprocess: 截断单条 tool_result（防御）──
        try:
            truncated = truncate_messages(messages or [])
        except Exception as e:
            logger.warning(f"[curator-write] truncate failed: {e}; using raw messages")
            truncated = messages or []

        if not truncated:
            return self._empty_response(rid, t0, error="empty messages")

        # ── 2. 构建 CuratorContext ──
        try:
            ctx = await self._build_context(
                truncated,
                user_id=user_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
                branch=branch,
                session_id=session_id,
                request_id=rid,
            )
        except Exception as e:
            logger.exception(f"[curator-write] build context failed: {e}")
            return self._empty_response(rid, t0, error=f"build context failed: {e}")

        # ── 3. 渲染 initial prompt ──
        try:
            existing_index = await self.store.list_user_memories_metadata(
                user_id=user_id, limit=self.existing_index_limit,
            )
        except Exception as e:
            logger.warning(f"[curator-write] list existing memories failed: {e}")
            existing_index = []

        last_user_queries = self._extract_last_user_queries(truncated)
        new_messages_compact = render_messages_compact(truncated)
        initial_user_prompt = render_initial_user_prompt(
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            branch=branch,
            session_id=session_id,
            existing_index=existing_index,
            last_user_queries=last_user_queries,
            new_messages_compact=new_messages_compact,
        )

        # ── 4. 跑 agent loop ──
        try:
            result = await self._run_agent_loop(initial_user_prompt, ctx)
        except Exception as e:
            logger.exception(f"[curator-write] agent loop failed: {e}")
            return self._empty_response(rid, t0, error=f"agent loop failed: {e}")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"[curator-write] done request_id={rid} "
            f"ops={len(result.ops)} (ADD={sum(1 for o in result.ops if o.action=='ADD')}, "
            f"UPDATE={sum(1 for o in result.ops if o.action=='UPDATE')}, "
            f"DELETE={sum(1 for o in result.ops if o.action=='DELETE')}) "
            f"iterations={result.iterations} truncated={result.truncated} "
            f"tokens={result.total_tokens} (p={result.prompt_tokens}/c={result.completion_tokens}) "
            f"tool_calls={dict(sorted(result.tool_call_counts.items()))} "
            f"elapsed={elapsed_ms}ms"
        )
        return {
            "success": True,
            "scene": "productivity",
            "request_id": rid,
            "ops": [
                {
                    "action": o.action,
                    "target_memory_id": o.target_memory_id,
                    "reason": o.reason,
                }
                for o in result.ops
            ],
            "memory_ids": result.memory_ids,
            "iterations": result.iterations,
            "truncated": result.truncated,
            "elapsed_ms": elapsed_ms,
            "writer": "agent",
            # ── metrics ──
            "tool_call_counts": dict(result.tool_call_counts),
            "tokens": {
                "prompt": result.prompt_tokens,
                "completion": result.completion_tokens,
                "total": result.total_tokens,
            },
            "prompt_chars_by_bucket": dict(result.prompt_chars_by_bucket),
        }

    # ----------------------------------------------------------------
    # Agent loop
    # ----------------------------------------------------------------

    async def _run_agent_loop(
        self,
        initial_user_prompt: str,
        ctx: CuratorContext,
    ) -> AgentResult:
        messages: List[Dict[str, Any]] = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": initial_user_prompt},
        ]
        ops: List[Any] = []
        memory_ids: List[str] = []
        iteration = 0
        last_error: Optional[str] = None
        # ── metrics ──
        tool_call_counts: Dict[str, int] = {}
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        # tool_schemas 每次都附带（fixed），先算一次
        try:
            tool_schemas_chars = len(json.dumps(TOOL_SCHEMAS, ensure_ascii=False))
        except Exception:
            tool_schemas_chars = 0
        prompt_chars_by_bucket: Dict[str, int] = {
            "system": 0,
            "initial_user": 0,
            "tool_schemas": 0,
            "agent_text": 0,
            "agent_tool_calls": 0,
            "tool_result": 0,
        }

        while iteration < self.max_iterations:
            iteration += 1

            # 在 LLM 调用前对当前 messages + tool_schemas 做一次分桶累加
            try:
                this_round = _count_prompt_buckets(messages, tool_schemas_chars)
                for k, v in this_round.items():
                    prompt_chars_by_bucket[k] = prompt_chars_by_bucket.get(k, 0) + v
            except Exception as e:
                logger.debug(f"[curator-loop] bucket count failed: {e}")

            try:
                resp = await self.llm.complete_messages(
                    messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except Exception as e:
                last_error = f"LLM call failed at iteration {iteration}: {e}"
                logger.warning(f"[curator-loop] {last_error}")
                break

            # 累积 token 消耗（LLMResponse 字段）
            prompt_tokens += int(getattr(resp, "prompt_tokens", 0) or 0)
            completion_tokens += int(getattr(resp, "completion_tokens", 0) or 0)
            total_tokens += int(getattr(resp, "tokens_used", 0) or 0)

            tool_calls = resp.tool_calls or []

            # 没 tool_calls 视为终止（容错：LLM 直接给文本结束）
            if not tool_calls:
                break

            # 把 assistant tool_calls 加回 history
            messages.append({
                "role": "assistant",
                "content": resp.content or "",
                "tool_calls": tool_calls,
            })

            terminated = False
            for tc in tool_calls:
                # 累积 tool 调用计数
                tool_name = ((tc.get("function") or {}).get("name")) or "<unknown>"
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

                tool_result: ToolResult = await dispatch_tool(
                    tc, ctx, self.store, self.embed_service,
                )
                # tool 消息回填（即使有 error 也要回填，给 LLM 改正机会）
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", "") or "",
                    "content": tool_result.payload_str,
                })
                if tool_result.op is not None:
                    ops.append(tool_result.op)
                    if tool_result.memory_id:
                        memory_ids.append(tool_result.memory_id)
                if tool_result.is_done:
                    terminated = True

            if terminated:
                break

        return AgentResult(
            ops=ops,
            memory_ids=memory_ids,
            iterations=iteration,
            truncated=(iteration >= self.max_iterations and not _last_was_done(messages)),
            error=last_error,
            tool_call_counts=tool_call_counts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_chars_by_bucket=prompt_chars_by_bucket,
        )

    # ----------------------------------------------------------------
    # Context builder
    # ----------------------------------------------------------------

    async def _build_context(
        self,
        messages: List[ChatMessage],
        *,
        user_id: str,
        agent_id: str,
        workspace_id: Optional[str],
        branch: Optional[str],
        session_id: Optional[str],
        request_id: str,
    ) -> CuratorContext:
        # full user messages: turn_idx → 完整 user 文本
        full_user_messages: Dict[int, str] = {}
        for i, m in enumerate(messages):
            if m.role == "user" and not m.is_tool_message():
                full_user_messages[i] = m.content or ""

        # tool_call index: tool_use_id → {name, arguments, result}
        tool_call_index: Dict[str, Dict[str, Any]] = {}
        # 第一遍：assistant.tool_calls 入索引
        for m in messages:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.id:
                        tool_call_index[tc.id] = {
                            "tool_use_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": None,
                        }
        # 第二遍：role=tool 消息回填 result
        for m in messages:
            if m.is_tool_message() and m.tool_call_id:
                info = tool_call_index.get(m.tool_call_id)
                if info is None:
                    info = {
                        "tool_use_id": m.tool_call_id,
                        "name": m.tool_name or "",
                        "arguments": None,
                        "result": None,
                    }
                    tool_call_index[m.tool_call_id] = info
                info["result"] = m.content or ""
                if not info.get("name"):
                    info["name"] = m.tool_name or ""

        # observed_files：从 tool_calls.arguments 抽 path / file_path 等
        observed_files = set(extract_files(messages))

        return CuratorContext(
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            branch=branch,
            session_id=session_id,
            request_id=request_id,
            full_user_messages=full_user_messages,
            tool_call_index=tool_call_index,
            observed_files=observed_files,
        )

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _extract_last_user_queries(
        self, messages: List[ChatMessage]
    ) -> List[Dict[str, Any]]:
        """
        提取最后 K 个真实 user 提问轮（去除 tool_result-only），
        返回 [{turn_idx, query}]，turn_idx 是该消息在原 messages 中的全局 idx。
        """
        all_queries: List[Dict[str, Any]] = []
        for i, m in enumerate(messages):
            if m.role == "user" and not m.is_tool_message():
                all_queries.append({"turn_idx": i, "query": m.content or ""})
        return all_queries[-self.last_k_user_queries:]

    def _empty_response(self, rid: str, t0: float, *, error: Optional[str] = None) -> Dict[str, Any]:
        return {
            "success": True,
            "scene": "productivity",
            "request_id": rid,
            "ops": [],
            "memory_ids": [],
            "iterations": 0,
            "truncated": False,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "writer": "agent",
            "tool_call_counts": {},
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "prompt_chars_by_bucket": {},
            **({"error": error} if error else {}),
        }


def _last_was_done(messages: List[Dict[str, Any]]) -> bool:
    """检查最后一轮是否调用了 done()。用于 truncated 标记。"""
    for m in reversed(messages):
        if m.get("role") == "tool":
            content = m.get("content") or ""
            return '"ok": true' in content and '"summary"' in content
        if m.get("role") == "assistant":
            return False
    return False


def _count_prompt_buckets(
    messages: List[Dict[str, Any]],
    tool_schemas_chars: int,
) -> Dict[str, int]:
    """
    对当前 messages 列表 + tool_schemas 做分桶字符计数。
    每轮 LLM 调用都会把整个 messages + tool_schemas 重发，所以累加跨轮即得到
    本次写入的"总 prompt 字符消耗"分布。

    Buckets:
      - system          : messages[0]（SYSTEM_PROMPT）
      - initial_user    : messages[1]（render_initial_user_prompt 输出）
      - tool_schemas    : 7 个 tool 的 OpenAI schema JSON
      - agent_text      : assistant 消息的 content（agent 的 reasoning 文本）
      - agent_tool_calls: assistant 消息的 tool_calls JSON
      - tool_result     : role=tool 消息的 content（dispatch 返回）
    """
    out: Dict[str, int] = {
        "system": 0,
        "initial_user": 0,
        "tool_schemas": tool_schemas_chars,
        "agent_text": 0,
        "agent_tool_calls": 0,
        "tool_result": 0,
    }
    for i, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content") or ""
        if i == 0 and role == "system":
            out["system"] += len(content)
        elif i == 1 and role == "user":
            out["initial_user"] += len(content)
        elif role == "assistant":
            out["agent_text"] += len(content)
            tcs = m.get("tool_calls") or []
            if tcs:
                try:
                    out["agent_tool_calls"] += len(json.dumps(tcs, ensure_ascii=False))
                except Exception:
                    out["agent_tool_calls"] += sum(
                        len(str(tc.get("function", {}).get("arguments", "")))
                        for tc in tcs
                    )
        elif role == "tool":
            out["tool_result"] += len(content)
    return out
