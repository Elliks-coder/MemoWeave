# -*- coding: utf-8 -*-
"""
Coding Memory - Extractor

LLM 提取 task / search_keys / solution / boundary_envs / boundary_scope / confidence。
价值门槛严格：rename a file 这种琐碎操作不该成 memory；只抽取未来可能复发的有价值问题。

复用 LLMConfig.model（与 chat extractor 共用同一 LLM 配置）。

详见 docs/coding_memory_mvp_design.md §6.5 / §6.5.2。
"""

import json
import logging
from typing import List, Optional, Dict, Any

from ..agent.llm_provider import LLMProvider
from ..pipelines.base import ChatMessage
from .preproc import truncate_messages, keep_last_k_user_turns, extract_files
from .types import CodingMemoryDraft, BOUNDARY_SCOPES

logger = logging.getLogger(__name__)


# ================================================================
# Prompt
# ================================================================

EXTRACT_PROMPT = """\
You extract durable engineering memories from a coding session segment.
Each memory is a (task, search_keys, solution, boundary_envs, boundary_scope) bundle.

═══════════════════════════════════════════════════════════════════
VALUE BAR — DO NOT cross this lightly.
═══════════════════════════════════════════════════════════════════
A memory MUST satisfy ALL of:
  (a) Solves a problem likely to recur for the same user
  (b) Solution is non-trivial: combines multiple pieces of info, OR carries
      a decision rationale, OR captures a hard-won workaround
  (c) A future agent / engineer would prefer looking it up over re-discovering it

DO NOT extract:
  - Trivial single-step ops ("rename file", "git status", "run ls")
  - One-off bugs already fixed with no transferable lesson
  - Information that is already in the repo (don't duplicate package.json)
  - Per-PR temporary tasks unless they encode a reusable pattern

Heuristic: if a senior engineer joining the project would care about this,
keep it. If they'd discover it via `ls / cat / --help` in 30 seconds, drop it.

═══════════════════════════════════════════════════════════════════
MEMORY FIELDS
═══════════════════════════════════════════════════════════════════

task          — Query-shaped name; phrase as how a user would later ASK.
                "打包发布到 xx 平台" not "publish_release_internal"
                Self-contained (no "this", "the file", "we").

search_keys   — 0 to 5 alternative phrasings capturing important aspects of
                solution that `task` alone wouldn't match. Phrase as questions
                or topics future users would ASK ABOUT. Distinct from task.
                Omit if task already covers everything.

solution      — Complete, self-contained content the user actually needs:
                commands, credential locations (NOT credential VALUES),
                file paths, ordered steps, gotchas, decision rationale.
                Aggregate multi-turn end-to-end work into ONE solution if it
                forms a coherent task; split into multiple memories only when
                the user did unrelated tasks.

boundary_envs — Concrete pinning that solution depends on, multi-line "key: value":
                  runtime / SDK / library + version
                  API endpoint or method signature
                  external service / middleware version
                  env var NAME (not value)
                  config file path
                  host / cluster / region identifier
                If solution is environment-agnostic, leave as "" (empty).
                Purpose: when the underlying SDK/middleware upgrades, future
                consumers can inspect this to decide if solution still applies.

boundary_scope — Reuse range of the memory. Pick narrower over wider:
  strict   — branch-specific or PR-specific (rare)
  project  — project-scoped (specific files / credentials / services / commands in solution)
  user     — cross-project user preference (no project tokens; e.g. tool/lang habit)
  global   — context-free knowledge (almost never; LLMs already know this)

confidence  — 0.0 to 1.0
  1.0   user explicitly stated this as a rule / instruction
  0.7   inferred with good evidence from tool actions
  0.5   weakly inferred; one observation; uncertain

═══════════════════════════════════════════════════════════════════
BOUNDARY GUARDS (host context, given by SDK)
═══════════════════════════════════════════════════════════════════
- workspace_id present: {workspace_present}
- branch present: {branch_present}

If workspace_id is NOT present, you MUST NOT pick boundary_scope of
"strict" or "project" — fall back to "user" or "global". Memories with a
project-bound solution but no workspace_id will be REJECTED downstream.

If branch is NOT present, you MUST NOT pick "strict" — fall back to
"project" / "user" / "global".

═══════════════════════════════════════════════════════════════════
EXISTING TASKS FOR THIS USER (avoid duplication; reconciler will UPDATE)
═══════════════════════════════════════════════════════════════════
{existing_tasks_block}

═══════════════════════════════════════════════════════════════════
CONVERSATION
═══════════════════════════════════════════════════════════════════
{conversation_block}

═══════════════════════════════════════════════════════════════════
OUTPUT (strict JSON only, no markdown)
═══════════════════════════════════════════════════════════════════
{{
  "memories": [
    {{
      "task": "...",
      "search_keys": ["...", "..."],
      "solution": "...",
      "boundary_envs": "...",
      "boundary_scope": "project",
      "confidence": 0.9
    }}
  ]
}}

If nothing meets the value bar, output:
{{"memories": []}}
"""


# ================================================================
# CodingMemoryExtractor
# ================================================================

class CodingMemoryExtractor:
    """
    LLM 驱动的 coding memory 抽取器。

    输入：已规范化、已截断的整段 messages（is_coding=True 的那段）+ 用户已有 task list
    输出：List[CodingMemoryDraft]
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_tokens: int = 2000,
        temperature: Optional[float] = None,
    ):
        self.llm_provider = llm_provider
        self.max_tokens = max_tokens
        # None → 使用 LLMConfig.temperature（避免硬编码触发 kimi/deepseek 的温度硬约束）
        if temperature is None:
            temperature = getattr(getattr(llm_provider, "_llm_config", None), "temperature", 0.1)
        self.temperature = temperature

    async def extract(
        self,
        messages: List[ChatMessage],
        *,
        user_id: str,
        agent_id: str = "default_agent",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        session_id: Optional[str] = None,
        existing_tasks: Optional[List[str]] = None,
    ) -> List[CodingMemoryDraft]:
        """
        Args:
            messages: 已经过 _parse_input 规范化和 truncate 的整段消息
            user_id / agent_id / workspace_id / branch / session_id: SDK 注入
            existing_tasks: 该用户已有的 task 列表（用于去重提示），可空

        Returns:
            草稿列表（可能为空，价值门槛未达）
        """
        if not messages:
            return []

        # 1) 截断单条 tool_result（防御：writer 已截过一次，再保险一次）
        truncated = truncate_messages(messages)
        # 2) 压缩到最近 K 个 user-turn 窗口；assistant 文本头尾截断
        #    避免长 trajectory 导致 LLM 注意力稀释 / under-extract
        windowed = keep_last_k_user_turns(truncated)

        prompt = EXTRACT_PROMPT.format(
            workspace_present=str(bool(workspace_id)).lower(),
            branch_present=str(bool(branch)).lower(),
            existing_tasks_block=self._format_existing_tasks(existing_tasks),
            conversation_block=self._format_conversation(windowed),
        )

        try:
            resp = await self.llm_provider.complete(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.warning(f"[coding-extract] LLM call failed: {e}")
            return []

        parsed = self._parse_output(resp.content)
        if not parsed:
            return []

        # 自动抽取 files（规则）
        files = extract_files(messages)

        drafts: List[CodingMemoryDraft] = []
        for raw in parsed:
            d = self._normalize_draft(
                raw,
                user_id=user_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
                branch=branch,
                session_id=session_id,
                files=files,
            )
            if d is not None:
                drafts.append(d)

        logger.info(
            f"[coding-extract] produced {len(drafts)} draft(s) "
            f"(workspace={workspace_id!r}, branch={branch!r})"
        )
        return drafts

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    @staticmethod
    def _format_existing_tasks(tasks: Optional[List[str]]) -> str:
        if not tasks:
            return "(none)"
        # 限制最多 30 条避免 prompt 膨胀
        rows = [f"- {t}" for t in tasks[:30]]
        if len(tasks) > 30:
            rows.append(f"... and {len(tasks) - 30} more")
        return "\n".join(rows)

    @staticmethod
    def _format_conversation(messages: List[ChatMessage]) -> str:
        """渲染整段对话给 LLM 读。"""
        rows: List[str] = []
        for i, m in enumerate(messages):
            if m.role == "user" and not m.is_tool_message():
                rows.append(f"[{i}] user: {m.content}")
            elif m.role == "assistant":
                line = f"[{i}] assistant: {m.content}"
                if m.tool_calls:
                    tc_summary = "; ".join(
                        f"{tc.name}({_compact_args(tc.arguments)})"
                        for tc in m.tool_calls
                    )
                    line += f"  [tools: {tc_summary}]"
                rows.append(line)
            elif m.is_tool_message():
                tn = m.tool_name or "?"
                rows.append(f"[{i}] tool({tn}): {m.content}")
            elif m.role == "system":
                rows.append(f"[{i}] system: {m.content}")
            else:
                rows.append(f"[{i}] {m.role}: {m.content}")
        return "\n".join(rows)

    @staticmethod
    def _parse_output(text: str) -> List[Dict[str, Any]]:
        """宽松解析 LLM 输出 JSON，返回 memories 列表。"""
        if not text:
            return []
        s = text.strip()
        # 去 markdown fence
        if s.startswith("```"):
            lines = s.split("\n")
            body, in_fence = [], False
            for line in lines:
                if line.startswith("```"):
                    in_fence = not in_fence
                    continue
                body.append(line)
            s = "\n".join(body).strip() or s
        # 尝试纯 JSON
        try:
            obj = json.loads(s)
            mems = obj.get("memories")
            if isinstance(mems, list):
                return mems
        except Exception:
            pass
        # 找第一个 {...} 块
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(s[start: end + 1])
                mems = obj.get("memories")
                if isinstance(mems, list):
                    return mems
            except Exception:
                pass
        logger.warning(f"[coding-extract] failed to parse output: {text!r}")
        return []

    @staticmethod
    def _normalize_draft(
        raw: Dict[str, Any],
        *,
        user_id: str,
        agent_id: str,
        workspace_id: Optional[str],
        branch: Optional[str],
        session_id: Optional[str],
        files: List[str],
    ) -> Optional[CodingMemoryDraft]:
        """
        把 LLM 输出的单条 memory 字典规范化为 CodingMemoryDraft。

        返回 None 表示该条无效（缺关键字段或 boundary 违规）。
        """
        task = (raw.get("task") or "").strip()
        solution = (raw.get("solution") or "").strip()
        if not task or not solution:
            logger.debug(f"[coding-extract] drop draft missing task/solution: {raw}")
            return None

        scope = (raw.get("boundary_scope") or "project").strip().lower()
        if scope not in BOUNDARY_SCOPES:
            logger.debug(f"[coding-extract] unknown boundary_scope {scope!r}, fallback to 'project'")
            scope = "project"

        # boundary 守卫：缺 workspace_id 时拒绝 strict/project；缺 branch 时拒绝 strict
        if scope in ("strict", "project") and not workspace_id:
            logger.info(
                f"[coding-extract] reject draft scope={scope} due to missing workspace_id; "
                f"task={task!r}"
            )
            return None
        if scope == "strict" and not branch:
            logger.info(
                f"[coding-extract] reject draft scope=strict due to missing branch; task={task!r}"
            )
            return None

        # search_keys 规范化
        sk_raw = raw.get("search_keys") or []
        if not isinstance(sk_raw, list):
            sk_raw = []
        search_keys = [str(s).strip() for s in sk_raw if str(s).strip()]
        # 去重 + 与 task 同名的 key 去掉
        seen = set()
        deduped = []
        for k in search_keys:
            if k == task:
                continue
            if k in seen:
                continue
            seen.add(k)
            deduped.append(k)
        search_keys = deduped[:5]  # 软上限 5

        # confidence 规范化
        try:
            confidence = float(raw.get("confidence", 0.7))
        except Exception:
            confidence = 0.7
        confidence = max(0.0, min(1.0, confidence))

        boundary_envs = (raw.get("boundary_envs") or "").strip()

        return CodingMemoryDraft(
            task=task,
            search_keys=search_keys,
            solution=solution,
            boundary_envs=boundary_envs,
            boundary_scope=scope,  # type: ignore[arg-type]
            confidence=confidence,
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            branch=branch,
            session_id=session_id,
            files=list(files),
            source="auto_extract",
        )


# ================================================================
# Helpers
# ================================================================

def _compact_args(args: Dict[str, Any], max_chars: int = 80) -> str:
    """单行紧凑展示 tool args。"""
    if not args:
        return ""
    try:
        s = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(args)
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s
