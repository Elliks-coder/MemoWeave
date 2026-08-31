# -*- coding: utf-8 -*-
"""
CodingCurator — Initial system + user prompt 渲染（渐进式披露：metadata-first）。

设计原则（详见 /root/.claude-internal/plans/squishy-leaping-orbit.md）:
- system prompt：agent 角色 + 决策原则 + tool catalog 简介 + 价值门槛 + boundary scope 规则 + 凭据脱敏要求
- initial user message：
  * ctx 元数据（user_id / agent_id / workspace_id / branch / session_id）
  * existing memories index（仅 metadata：memory_id + task + scope + confidence；不含 solution）
  * last 5 user queries（仅 user 文本，去除 tool_result-only，按 turn_idx 编号）
  * new messages compact 视图（user 完整 / assistant 头200尾200 / tool_use name+keys / tool_result 头256尾256）

详细 task 内容由 agent 主动调 read_existing_memory / read_full_user_message /
read_tool_call_detail / search_memory 按需取。
"""

from typing import Any, Dict, List, Optional

from ...pipelines.base import ChatMessage


# ================================================================
# SYSTEM PROMPT
# ================================================================

SYSTEM_PROMPT = """\
You are CodingCurator — a careful agent that decides what engineering knowledge
to record from a programming session, so future you / your teammates can find
and reuse it.

You have access to a small set of tools. Use them in two phases:

PHASE A — Investigate (read-only tools)
  - read_full_user_message(turn_idx)   ← see the full text of a user query
  - read_existing_memory(memory_id)    ← see full task / search_keys / solution / boundary_envs / files of an existing memory
  - read_tool_call_detail(tool_use_id) ← see the full arguments + full result of a single tool call
  - search_memory(query, top_k)        ← semantic search across this user's existing memories

PHASE B — Decide (write tools, side-effects)
  - create_memory(...)                 ← record a brand-new piece of knowledge
  - update_memory(memory_id, ...)      ← refine / expand an existing memory
  - delete_memory(memory_id, reason)   ← remove an obsolete memory (rare; only when explicitly superseded)

When you are completely done:
  - done(summary)                      ← terminate the loop

# Decision principles

1. **High value bar**. Record knowledge only if a future engineer would search for it later.
   - Reusable how-tos: deployment / build / config / debugging procedures
   - Architectural decisions and their rationale
   - Project conventions, coding standards, tooling choices
   - Subtle gotchas that cost real time to figure out
   Reject:
   - One-off operations (e.g. "I ran `ls` to check this directory")
   - Obvious / common knowledge ("Python lists are zero-indexed")
   - Fleeting state ("There are 3 files in /tmp right now")

2. **Self-contained solutions**. Each memory's `solution` must be readable on its own.
   Include relevant file paths, env vars, command snippets, decision rationale.

3. **Search keys are real searches**. Each search_key should be a phrase a colleague
   would actually type into a search bar — questions, error symptoms, "how do I X".
   2~5 keys per memory; avoid redundancy.

4. **Boundary scope**:
   - "strict"  — tied to a specific (workspace_id, branch). Examples: branch-specific WIP.
   - "project" — tied to a workspace_id. Most engineering memories. Requires workspace_id present.
   - "user"    — user preferences across projects (e.g. "always use pnpm not npm").
   - "global"  — universally true (rarely used; reserve for genuinely universal facts).
   If workspace_id is absent, do NOT use "strict" or "project".

5. **Credential redaction (CRITICAL)**.
   Solution must NOT contain plaintext secrets. The dispatcher will auto-redact:
     password / api[_-]?key / secret / token / NEO4J_PASSWORD / etc.
   But you should also avoid putting raw credentials there in the first place.
   Reference the env var name, not the value.

6. **Files honesty**. The `files` list must be paths actually accessed during this
   session (read / written / referenced via tool calls). The dispatcher filters
   out unobserved paths automatically — don't try to enrich.

7. **Be efficient**. Use read tools sparingly — only when you really need details
   that the compact initial view doesn't show. Don't read the same thing twice.

# Workflow

A typical loop:
  1. Skim the initial compact view (user queries, message digest, existing index).
  2. If something looks promising but unclear → call ONE read tool to see details.
   3. If a candidate memory matches existing tasks → search_memory or read_existing_memory
      to decide between create vs update vs skip.
  4. Call create_memory / update_memory (one per knowledge unit).
  5. Call done(summary) to terminate.

You don't have to extract anything. If the conversation has no reusable knowledge,
just call done("no value") immediately.

Output format: ALWAYS use tool calls, never reply in plain text.\
"""


# ================================================================
# INITIAL USER PROMPT
# ================================================================

def render_initial_user_prompt(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: Optional[str],
    branch: Optional[str],
    session_id: Optional[str],
    existing_index: List[Dict[str, Any]],
    last_user_queries: List[Dict[str, Any]],
    new_messages_compact: str,
) -> str:
    """
    渲染首条 user message（agent 看到的工作空间）。

    Args:
        existing_index: List of {memory_id, task, boundary_scope, confidence}
        last_user_queries: List of {turn_idx, query} (already capped to last K)
        new_messages_compact: 当前 batch 的压缩视图（caller 已用 _format_compact_messages 渲染好）
    """
    parts: List[str] = []

    # ── 1. ctx 元数据 ──
    parts.append("# Session context")
    parts.append(f"- user_id     : {user_id}")
    parts.append(f"- agent_id    : {agent_id}")
    parts.append(f"- workspace_id: {workspace_id or '(absent)'}")
    parts.append(f"- branch      : {branch or '(absent)'}")
    parts.append(f"- session_id  : {session_id or '(absent)'}")

    # ── 2. existing memories index ──
    parts.append("\n# Existing memories index (this user)")
    if not existing_index:
        parts.append("(none — first time we are recording for this user)")
    else:
        parts.append("Use `read_existing_memory(memory_id)` for full content of any entry below.")
        for e in existing_index:
            parts.append(
                f"- memory_id={e['memory_id']}  "
                f"scope={e.get('boundary_scope', '?')}  "
                f"conf={e.get('confidence', '?')}  "
                f"task={e.get('task', '')!r}"
            )

    # ── 3. last K user queries ──
    parts.append(f"\n# Last {len(last_user_queries)} user queries (most recent last)")
    if not last_user_queries:
        parts.append("(none)")
    else:
        parts.append("Use `read_full_user_message(turn_idx)` to see full text if needed.")
        for q in last_user_queries:
            text = (q.get("query") or "").strip()
            # initial 视图给截断版
            if len(text) > 400:
                text = text[:200] + " [...] " + text[-200:]
            parts.append(f"- turn_idx={q['turn_idx']}: {text}")

    # ── 4. compact view of new messages ──
    parts.append("\n# New messages (compact view)")
    parts.append(
        "Each tool_use has a tool_use_id you can pass to "
        "`read_tool_call_detail(tool_use_id)` for full args + full result."
    )
    parts.append(new_messages_compact)

    parts.append(
        "\n# Your task"
        "\nReview the above. Decide which (if any) reusable engineering knowledge"
        "\nis worth recording, then call write tools (create / update / delete)"
        "\nfollowed by `done(summary)`. Use read tools to inspect details when needed."
    )

    return "\n".join(parts)


# ================================================================
# Compact view of messages（initial prompt 中的 new_messages_compact 块）
# ================================================================

DEFAULT_ASSISTANT_HEAD = 200
DEFAULT_ASSISTANT_TAIL = 200
DEFAULT_TOOL_RESULT_HEAD = 256
DEFAULT_TOOL_RESULT_TAIL = 256


def _truncate_head_tail(text: str, head: int, tail: int) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return text[:head] + f" [...{omitted} chars omitted...] " + text[-tail:]


def _compact_args_keys(args: Dict[str, Any]) -> str:
    """渲染 tool_use 的 arg key 列表（不含 value）"""
    if not isinstance(args, dict) or not args:
        return ""
    return ", ".join(sorted(args.keys()))


def render_messages_compact(
    messages: List[ChatMessage],
    *,
    assistant_head: int = DEFAULT_ASSISTANT_HEAD,
    assistant_tail: int = DEFAULT_ASSISTANT_TAIL,
    tool_result_head: int = DEFAULT_TOOL_RESULT_HEAD,
    tool_result_tail: int = DEFAULT_TOOL_RESULT_TAIL,
) -> str:
    """
    渲染当前 batch 消息的「压缩视图」给 agent。
    详细 tool 调用 / 完整 user 文本由 agent 主动调 read tools 取。
    """
    rows: List[str] = []
    for i, m in enumerate(messages):
        if m.role == "user" and not m.is_tool_message():
            text = (m.content or "").strip()
            rows.append(f"[turn_idx={i}] user: {text}")
        elif m.role == "assistant":
            text = _truncate_head_tail(m.content or "", assistant_head, assistant_tail)
            line = f"[idx={i}] assistant: {text}"
            if m.tool_calls:
                tc_parts = []
                for tc in m.tool_calls:
                    keys = _compact_args_keys(tc.arguments)
                    tc_parts.append(
                        f"{tc.name}(id={tc.id} keys=[{keys}])"
                    )
                line += "  [tool_use: " + "; ".join(tc_parts) + "]"
            rows.append(line)
        elif m.is_tool_message():
            tn = m.tool_name or "?"
            text = _truncate_head_tail(m.content or "", tool_result_head, tool_result_tail)
            rows.append(f"[idx={i}] tool({tn}, id={m.tool_call_id}): {text}")
        else:
            rows.append(f"[idx={i}] {m.role}: {m.content}")
    return "\n".join(rows)
