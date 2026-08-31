# -*- coding: utf-8 -*-
"""
CodingCurator — Tool catalog (OpenAI function-calling schemas) + dispatcher.

7 个工具：4 read + 3 write + 1 done。

Dispatcher 兜底防御（响应独立审计员发现的 P0 问题）:
- 凭据脱敏：solution / boundary_envs 中常见 password / api_key / secret / token
  自动 redact 成占位符
- files 白名单：只保留确实在 trajectory 中通过 tool_calls.arguments 出现过的路径
- boundary guard：缺 workspace_id 时拒绝 strict / project；缺 branch 时拒绝 strict
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..types import CodingMemory, ReconcileOp, BoundaryScope

logger = logging.getLogger(__name__)


# ================================================================
# Tool schemas (OpenAI function-calling 风格)
# ================================================================

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    # ── Read tools ──
    {
        "type": "function",
        "function": {
            "name": "read_full_user_message",
            "description": (
                "Read the full original text of a user query at a given turn_idx, "
                "bypassing any compact-view truncation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "turn_idx": {
                        "type": "integer",
                        "description": "The turn_idx as shown in the compact view.",
                    },
                },
                "required": ["turn_idx"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_existing_memory",
            "description": (
                "Read the full content (task / search_keys / solution / boundary_envs / files / "
                "boundary_scope / confidence) of an existing memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_tool_call_detail",
            "description": (
                "Read the full arguments + full result of a single tool call by its tool_use_id, "
                "bypassing the head/tail truncation in the compact view."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_use_id": {"type": "string"},
                },
                "required": ["tool_use_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Semantic search this user's existing memories. Returns metadata only "
                "(memory_id / task / score). Use read_existing_memory for full content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    # ── Write tools ──
    {
        "type": "function",
        "function": {
            "name": "create_memory",
            "description": (
                "Create a brand-new coding memory. Use only when no existing memory covers "
                "this knowledge. Cred values are auto-redacted; unobserved file paths are "
                "auto-filtered."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Short description of the task / question. Self-contained."},
                    "search_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2~5 phrases someone would actually search for to find this memory.",
                    },
                    "solution": {"type": "string", "description": "Self-contained solution / answer / decision rationale."},
                    "boundary_envs": {"type": "string", "description": "Optional structured env hints (file paths, env var names, versions). Multi-line OK."},
                    "boundary_scope": {
                        "type": "string",
                        "enum": ["strict", "project", "user", "global"],
                        "description": "strict=branch-specific / project=workspace-tied / user=cross-project preference / global=universally true",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths actually accessed during this session. Will be filtered against the observed-paths whitelist.",
                    },
                    "confidence": {"type": "number", "description": "0.0~1.0; how sure you are this is correct.", "default": 0.8},
                },
                "required": ["task", "solution", "boundary_scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": (
                "Update an existing memory's fields. Only the fields you specify are changed; "
                "other fields are preserved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "task": {"type": "string"},
                    "search_keys": {"type": "array", "items": {"type": "string"}},
                    "solution": {"type": "string"},
                    "boundary_envs": {"type": "string"},
                    "boundary_scope": {"type": "string", "enum": ["strict", "project", "user", "global"]},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string", "description": "Why this update is needed."},
                },
                "required": ["memory_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory",
            "description": (
                "Delete an existing memory. Use ONLY when the user explicitly deprecates / "
                "supersedes the prior knowledge. Not for 'I think this is stale'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "reason": {"type": "string", "description": "Required. Cite the explicit deprecation evidence."},
                },
                "required": ["memory_id", "reason"],
            },
        },
    },
    # ── Terminate ──
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Terminate the agent loop. Always call this when finished.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "One-line summary of what you did (or why you decided not to record anything)."},
                },
                "required": ["summary"],
            },
        },
    },
]


# ================================================================
# CuratorContext — agent loop 期间需要查询的「事实库」
# ================================================================

@dataclass
class CuratorContext:
    """
    封装 agent 在 loop 期间所有需要的查询数据，避免每次 dispatch 都重新算。
    由 CodingCuratorWriter 在 write() 入口处构建一次。
    """
    user_id: str
    agent_id: str
    workspace_id: Optional[str]
    branch: Optional[str]
    session_id: Optional[str]
    request_id: str

    # 用于 read_full_user_message：turn_idx → 完整 user 文本
    full_user_messages: Dict[int, str] = field(default_factory=dict)

    # 用于 read_tool_call_detail：tool_use_id → {"name", "arguments", "result"}
    tool_call_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 用于 files 白名单兜底（防止 LLM 编造路径）
    observed_files: set = field(default_factory=set)

    # 用于 read_existing_memory / search_memory（不放在 ctx 里，dispatcher 直接用 store）


# ================================================================
# ToolResult
# ================================================================

@dataclass
class ToolResult:
    """单次 tool dispatch 的产出"""
    payload_str: str               # 序列化好喂给 LLM 的字符串
    op: Optional[ReconcileOp] = None       # 写动作产出的 ops（read tools 为 None）
    memory_id: Optional[str] = None        # 写动作涉及的 memory_id
    is_done: bool = False                  # done() tool 触发
    error: Optional[str] = None            # 出错时填，仍走 fail-safe 不抛


# ================================================================
# 凭据脱敏 + files 白名单
# ================================================================

# 常见凭据键名 + 紧跟着的 value
_CREDENTIAL_PATTERNS: List[re.Pattern] = [
    # KEY=value / KEY: value / KEY = "value"
    re.compile(
        r"((?:password|passwd|pwd|api[_-]?key|apikey|secret|token|access[_-]?key|"
        r"private[_-]?key|auth[_-]?token|bearer|"
        r"NEO4J_PASSWORD|MYSQL_PASSWORD|REDIS_PASSWORD|"
        r"MEMORY_EMBEDDER_API_KEY|MEMORY_LLM_API_KEY|MEMORY_LLM_EVAL_APIKEY)"
        r"\s*[=:]\s*)"
        r"['\"]?"
        r"([^\s'\"<>]+)"
        r"['\"]?",
        re.IGNORECASE,
    ),
    # twine upload --password XXX 风格
    re.compile(
        r"(--(?:password|api[_-]?key|token)\s+)([^\s]+)",
        re.IGNORECASE,
    ),
]


def redact_secrets(text: str) -> str:
    """对已知凭据键的 value 做替换。保守版本：只处理常见键名，不误伤 IP / 普通字面量。"""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pat in _CREDENTIAL_PATTERNS:
        out = pat.sub(lambda m: m.group(1) + "<REDACTED>", out)
    return out


def filter_files_by_whitelist(
    proposed: List[str],
    observed_files: set,
) -> List[str]:
    """只保留 observed_files 中实际出现过的路径。"""
    if not proposed:
        return []
    if not observed_files:
        # 没有任何观察到的路径 → LLM 不应该说有，全部丢弃
        return []
    return [p for p in proposed if isinstance(p, str) and p in observed_files]


# ================================================================
# Dispatcher
# ================================================================

async def dispatch_tool(
    tc: Dict[str, Any],
    ctx: CuratorContext,
    store,            # CodingMemoryStore
    embed_service,    # EmbedService
) -> ToolResult:
    """
    单条 tool_call 调度。任何异常都包装成 ToolResult(error=...)，不向上抛。
    """
    fn = tc.get("function") or {}
    name = fn.get("name", "")
    raw_args = fn.get("arguments")
    try:
        if isinstance(raw_args, str):
            args = json.loads(raw_args) if raw_args else {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
    except Exception as e:
        return ToolResult(
            payload_str=json.dumps({"error": f"invalid arguments JSON: {e}"}, ensure_ascii=False),
            error=str(e),
        )

    try:
        if name == "done":
            return ToolResult(
                payload_str=json.dumps({"ok": True, "summary": args.get("summary", "")}, ensure_ascii=False),
                is_done=True,
            )
        if name == "read_full_user_message":
            return _tool_read_full_user_message(args, ctx)
        if name == "read_existing_memory":
            return await _tool_read_existing_memory(args, ctx, store)
        if name == "read_tool_call_detail":
            return _tool_read_tool_call_detail(args, ctx)
        if name == "search_memory":
            return await _tool_search_memory(args, ctx, store, embed_service)
        if name == "create_memory":
            return await _tool_create_memory(args, ctx, store)
        if name == "update_memory":
            return await _tool_update_memory(args, ctx, store)
        if name == "delete_memory":
            return await _tool_delete_memory(args, ctx, store)
    except Exception as e:
        logger.warning(f"[curator-tool] dispatch {name} failed: {e}")
        return ToolResult(
            payload_str=json.dumps({"error": f"{name} failed: {e}"}, ensure_ascii=False),
            error=str(e),
        )

    return ToolResult(
        payload_str=json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False),
        error=f"unknown tool: {name}",
    )


# ── 各 tool 实现 ──

def _tool_read_full_user_message(args: Dict[str, Any], ctx: CuratorContext) -> ToolResult:
    turn_idx = args.get("turn_idx")
    if not isinstance(turn_idx, int):
        return ToolResult(payload_str=json.dumps({"error": "turn_idx must be int"}, ensure_ascii=False))
    text = ctx.full_user_messages.get(turn_idx)
    if text is None:
        return ToolResult(payload_str=json.dumps({"error": f"no user message at turn_idx={turn_idx}"}, ensure_ascii=False))
    return ToolResult(payload_str=json.dumps({"turn_idx": turn_idx, "text": text}, ensure_ascii=False))


async def _tool_read_existing_memory(args: Dict[str, Any], ctx: CuratorContext, store) -> ToolResult:
    memory_id = args.get("memory_id")
    if not memory_id:
        return ToolResult(payload_str=json.dumps({"error": "memory_id required"}, ensure_ascii=False))
    m = await store.get_by_id(memory_id, user_id=ctx.user_id)
    if m is None:
        return ToolResult(payload_str=json.dumps({"error": f"memory_id={memory_id} not found"}, ensure_ascii=False))
    return ToolResult(payload_str=json.dumps({
        "memory_id": m.memory_id,
        "task": m.task,
        "search_keys": list(m.search_keys),
        "solution": m.solution,
        "boundary_envs": m.boundary_envs,
        "boundary_scope": m.boundary_scope,
        "workspace_id": m.workspace_id,
        "branch": m.branch,
        "files": list(m.files),
        "confidence": m.confidence,
    }, ensure_ascii=False))


def _tool_read_tool_call_detail(args: Dict[str, Any], ctx: CuratorContext) -> ToolResult:
    tool_use_id = args.get("tool_use_id")
    if not tool_use_id:
        return ToolResult(payload_str=json.dumps({"error": "tool_use_id required"}, ensure_ascii=False))
    info = ctx.tool_call_index.get(tool_use_id)
    if info is None:
        return ToolResult(payload_str=json.dumps({"error": f"tool_use_id={tool_use_id} not found"}, ensure_ascii=False))
    return ToolResult(payload_str=json.dumps(info, ensure_ascii=False))


async def _tool_search_memory(args: Dict[str, Any], ctx: CuratorContext, store, embed_service) -> ToolResult:
    query = (args.get("query") or "").strip()
    top_k = int(args.get("top_k") or 5)
    if not query:
        return ToolResult(payload_str=json.dumps({"error": "query required"}, ensure_ascii=False))
    try:
        emb = await embed_service.embed(query)
    except Exception as e:
        return ToolResult(payload_str=json.dumps({"error": f"embed failed: {e}"}, ensure_ascii=False))
    try:
        hits = await store.search_by_query_embedding(
            emb,
            user_id=ctx.user_id,
            workspace_id=ctx.workspace_id,
            branch=ctx.branch,
            top=top_k,
        )
    except Exception as e:
        return ToolResult(payload_str=json.dumps({"error": f"search failed: {e}"}, ensure_ascii=False))
    # 拉详情仅取 task name
    out = []
    for h in (hits or [])[:top_k]:
        out.append({
            "memory_id": h.get("memory_id"),
            "score": round(float(h.get("score", 0.0)), 4),
            "matched_key": h.get("matched_key"),
            "matched_key_kind": h.get("matched_key_kind"),
        })
    return ToolResult(payload_str=json.dumps({"query": query, "hits": out}, ensure_ascii=False))


# ── Write tools ──

def _validate_boundary_scope(
    scope: str,
    workspace_id: Optional[str],
    branch: Optional[str],
) -> Optional[str]:
    """返回错误描述，None=OK"""
    if scope not in ("strict", "project", "user", "global"):
        return f"invalid boundary_scope={scope!r}"
    if scope == "strict" and (not workspace_id or not branch):
        return "boundary_scope='strict' requires both workspace_id and branch"
    if scope == "project" and not workspace_id:
        return "boundary_scope='project' requires workspace_id"
    return None


async def _tool_create_memory(args: Dict[str, Any], ctx: CuratorContext, store) -> ToolResult:
    task = (args.get("task") or "").strip()
    solution = args.get("solution") or ""
    scope = (args.get("boundary_scope") or "").strip()
    if not task:
        return ToolResult(payload_str=json.dumps({"error": "task required"}, ensure_ascii=False))
    if not solution:
        return ToolResult(payload_str=json.dumps({"error": "solution required"}, ensure_ascii=False))
    err = _validate_boundary_scope(scope, ctx.workspace_id, ctx.branch)
    if err:
        return ToolResult(payload_str=json.dumps({"error": err}, ensure_ascii=False))

    # 凭据脱敏（双保险：solution + boundary_envs）
    solution = redact_secrets(solution)
    boundary_envs = redact_secrets(args.get("boundary_envs") or "")

    # files 白名单
    files = filter_files_by_whitelist(args.get("files") or [], ctx.observed_files)

    # search_keys：去重保序，截断 5 条
    raw_keys = args.get("search_keys") or []
    seen = set()
    keys: List[str] = []
    for k in raw_keys:
        if isinstance(k, str) and k.strip() and k not in seen:
            seen.add(k)
            keys.append(k.strip())
    keys = keys[:5]

    try:
        confidence = float(args.get("confidence", 0.8))
    except Exception:
        confidence = 0.8

    memory_id = str(uuid.uuid4())
    mem = CodingMemory(
        memory_id=memory_id,
        user_id=ctx.user_id,
        agent_id=ctx.agent_id,
        task=task,
        search_keys=keys,
        solution=solution,
        boundary_envs=boundary_envs,
        boundary_scope=scope,
        workspace_id=ctx.workspace_id,
        branch=ctx.branch,
        session_id=ctx.session_id,
        files=files,
        confidence=confidence,
        source="curator_agent",
    )
    await store.insert(mem)
    op = ReconcileOp(action="ADD", target_memory_id=memory_id)
    return ToolResult(
        payload_str=json.dumps({
            "ok": True,
            "memory_id": memory_id,
            "redacted_credentials": (solution != args.get("solution")) or (boundary_envs != (args.get("boundary_envs") or "")),
            "files_kept": files,
            "files_dropped": sorted(set(args.get("files") or []) - set(files)),
        }, ensure_ascii=False),
        op=op,
        memory_id=memory_id,
    )


async def _tool_update_memory(args: Dict[str, Any], ctx: CuratorContext, store) -> ToolResult:
    memory_id = args.get("memory_id")
    reason = args.get("reason")
    if not memory_id:
        return ToolResult(payload_str=json.dumps({"error": "memory_id required"}, ensure_ascii=False))
    if not reason:
        return ToolResult(payload_str=json.dumps({"error": "reason required"}, ensure_ascii=False))

    existing = await store.get_by_id(memory_id, user_id=ctx.user_id)
    if existing is None:
        return ToolResult(payload_str=json.dumps({"error": f"memory_id={memory_id} not found"}, ensure_ascii=False))

    # 仅覆盖 LLM 指定的字段
    if "task" in args and args["task"]:
        existing.task = args["task"].strip()
    if "search_keys" in args and isinstance(args["search_keys"], list):
        seen = set(); keys = []
        for k in args["search_keys"]:
            if isinstance(k, str) and k.strip() and k not in seen:
                seen.add(k); keys.append(k.strip())
        existing.search_keys = keys[:5]
    if "solution" in args and args["solution"]:
        existing.solution = redact_secrets(args["solution"])
    if "boundary_envs" in args:
        existing.boundary_envs = redact_secrets(args["boundary_envs"] or "")
    if "boundary_scope" in args and args["boundary_scope"]:
        new_scope = args["boundary_scope"].strip()
        err = _validate_boundary_scope(new_scope, existing.workspace_id, existing.branch)
        if err:
            return ToolResult(payload_str=json.dumps({"error": err}, ensure_ascii=False))
        existing.boundary_scope = new_scope
    if "files" in args and isinstance(args["files"], list):
        existing.files = filter_files_by_whitelist(args["files"], ctx.observed_files)
    if "confidence" in args:
        try:
            existing.confidence = float(args["confidence"])
        except Exception:
            pass

    await store.update(existing)
    op = ReconcileOp(action="UPDATE", target_memory_id=memory_id, reason=reason)
    return ToolResult(
        payload_str=json.dumps({"ok": True, "memory_id": memory_id, "reason": reason}, ensure_ascii=False),
        op=op,
        memory_id=memory_id,
    )


async def _tool_delete_memory(args: Dict[str, Any], ctx: CuratorContext, store) -> ToolResult:
    memory_id = args.get("memory_id")
    reason = args.get("reason")
    if not memory_id:
        return ToolResult(payload_str=json.dumps({"error": "memory_id required"}, ensure_ascii=False))
    if not reason:
        return ToolResult(payload_str=json.dumps({"error": "reason required (deletion needs explicit deprecation evidence)"}, ensure_ascii=False))
    existing = await store.get_by_id(memory_id, user_id=ctx.user_id)
    if existing is None:
        return ToolResult(payload_str=json.dumps({"error": f"memory_id={memory_id} not found"}, ensure_ascii=False))
    await store.delete(memory_id, user_id=ctx.user_id)
    op = ReconcileOp(action="DELETE", target_memory_id=memory_id, reason=reason)
    return ToolResult(
        payload_str=json.dumps({"ok": True, "memory_id": memory_id, "reason": reason}, ensure_ascii=False),
        op=op,
        memory_id=memory_id,
    )
