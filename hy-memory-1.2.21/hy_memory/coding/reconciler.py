# -*- coding: utf-8 -*-
"""
Coding Memory Reconciler

输入：List[CodingMemoryDraft]（来自 extractor）
输出：List[ReconcileOp] —— ADD / UPDATE / DELETE / SKIP，**ops 数可超过 drafts 数**
        （DELETE 由 LLM 在 draft 与 candidate 关系中观察到"明确否定/作废"时附带产出）

执行端：reconcile() 调用 store 完成实际 ADD / UPDATE / DELETE 持久化操作。

复用 LLMConfig.model（与 chat reconciler 共用）。

详见 docs/coding_memory_mvp_design.md §6.6。
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..agent.llm_provider import LLMProvider
from ..core.embed_service import EmbedService
from .store import CodingMemoryStore
from .types import CodingMemory, CodingMemoryDraft, ReconcileOp

logger = logging.getLogger(__name__)


# ================================================================
# Prompt
# ================================================================

RECONCILE_PROMPT = """\
You decide how to merge new coding memory drafts with the user's existing memories.

For each new draft, decide one of:
  ADD     — no existing memory describes the same task/aspect
  UPDATE  — an existing memory describes the same task; the new draft replaces it
            (solution may have changed, env may have upgraded, etc.)
  SKIP    — duplicate of existing; nothing new

Additionally, you may emit DELETE ops on existing candidates when the new
content makes them obsolete or invalid:
  DELETE  — the new conversation EXPLICITLY supersedes/cancels/invalidates an
            existing memory that cannot be repaired via UPDATE. Examples:
              * user says "don't use X anymore" and X is only present as a
                separate memory (not the one being added/updated)
              * candidate's boundary_envs (service / endpoint / SDK) is gone
                and the new conversation confirms the deprecation

Rules:
- "Same task" means the existing memory and the new draft would both be valid
  answers to the same future user query.
- If solution differs, prefer UPDATE (we don't keep history).
- If both task and solution match closely, SKIP.
- DELETE only when there is EXPLICIT evidence the candidate is invalid/obsolete.
  Never DELETE just because a candidate looks "stale".
- Output ops can outnumber drafts when DELETEs are produced.

═══════════════════════════════════════════════════════════════════
NEW DRAFTS
═══════════════════════════════════════════════════════════════════
{drafts_block}

═══════════════════════════════════════════════════════════════════
EXISTING CANDIDATES (per draft, retrieved by task similarity)
═══════════════════════════════════════════════════════════════════
{candidates_block}

═══════════════════════════════════════════════════════════════════
OUTPUT (strict JSON only, no markdown)
═══════════════════════════════════════════════════════════════════
{{
  "ops": [
    {{"draft_idx": 0, "action": "ADD"}},
    {{"draft_idx": 1, "action": "UPDATE", "target_memory_id": "mem_xxx", "reason": "..."}},
    {{"draft_idx": 1, "action": "DELETE", "target_memory_id": "mem_yyy", "reason": "user explicitly deprecated Y"}},
    {{"draft_idx": 2, "action": "SKIP",   "reason": "duplicate"}}
  ]
}}

If there are no drafts, output {{"ops": []}}.
"""


# ================================================================
# CodingMemoryReconciler
# ================================================================

class CodingMemoryReconciler:
    """
    Reconcile + 持久化执行器。

    一次调用：
      1. 为每个 draft 用 task embedding 检索 top-3 candidates
      2. 单次 LLM 决策产出 ReconcileOp[]
      3. 调 store 执行 ADD / UPDATE / DELETE
    """

    def __init__(
        self,
        store: CodingMemoryStore,
        embed_service: EmbedService,
        llm_provider: LLMProvider,
        *,
        max_tokens: int = 1500,
        temperature: Optional[float] = None,
        candidates_per_draft: int = 3,
    ):
        self.store = store
        self.embed_service = embed_service
        self.llm_provider = llm_provider
        self.max_tokens = max_tokens
        # None → 复用 LLMConfig.temperature（避免硬编码触发 kimi/deepseek 的温度硬约束）
        if temperature is None:
            temperature = getattr(getattr(llm_provider, "_llm_config", None), "temperature", 0.1)
        self.temperature = temperature
        self.candidates_per_draft = candidates_per_draft

    async def reconcile(
        self,
        drafts: List[CodingMemoryDraft],
    ) -> List[ReconcileOp]:
        """
        执行 reconcile + 持久化。返回执行的 ReconcileOp[]（含成功执行的 ADD/UPDATE/DELETE/SKIP）。

        执行错误（如 LLM 失败）会 fail-safe 回退到对每个 draft 直接 ADD。
        """
        if not drafts:
            return []

        # 1. 检索 candidates
        candidates_per_draft: List[List[CodingMemory]] = await self._retrieve_candidates(drafts)

        # 2. LLM 决策
        try:
            ops = await self._llm_decide(drafts, candidates_per_draft)
        except Exception as e:
            logger.warning(f"[coding-reconcile] LLM decide failed, fallback to ADD-all: {e}")
            ops = [ReconcileOp(action="ADD", draft_idx=i) for i in range(len(drafts))]

        if not ops:
            # LLM 输出无 ops，但有 drafts → fallback ADD-all
            logger.info("[coding-reconcile] LLM returned no ops, fallback to ADD-all")
            ops = [ReconcileOp(action="ADD", draft_idx=i) for i in range(len(drafts))]

        # 3. 校验 + 执行
        executed = await self._execute(ops, drafts)
        logger.info(
            f"[coding-reconcile] drafts={len(drafts)} ops={len(executed)} "
            + ", ".join(f"{a}={sum(1 for o in executed if o.action == a)}"
                        for a in ("ADD", "UPDATE", "DELETE", "SKIP"))
        )
        return executed

    # ------------------------------------------------------------
    # 检索 candidates
    # ------------------------------------------------------------

    async def _retrieve_candidates(
        self, drafts: List[CodingMemoryDraft]
    ) -> List[List[CodingMemory]]:
        """每个 draft 用 task embedding 检索同 user 的 top-N candidates。"""
        result: List[List[CodingMemory]] = []
        for d in drafts:
            try:
                emb = await self.embed_service.embed_batch([d.task])
                hits = await self.store.search_by_query_embedding(
                    emb[0],
                    user_id=d.user_id,
                    workspace_id=d.workspace_id,
                    branch=d.branch,
                    top=self.candidates_per_draft * 4,
                )
                cand_ids = [h["memory_id"] for h in hits[: self.candidates_per_draft]]
                cands = await self.store.get_many(cand_ids, user_id=d.user_id)
                result.append(cands)
            except Exception as e:
                logger.warning(f"[coding-reconcile] retrieve candidates failed: {e}")
                result.append([])
        return result

    # ------------------------------------------------------------
    # LLM 决策
    # ------------------------------------------------------------

    async def _llm_decide(
        self,
        drafts: List[CodingMemoryDraft],
        candidates_per_draft: List[List[CodingMemory]],
    ) -> List[ReconcileOp]:
        prompt = RECONCILE_PROMPT.format(
            drafts_block=self._format_drafts(drafts),
            candidates_block=self._format_candidates(drafts, candidates_per_draft),
        )

        resp = await self.llm_provider.complete(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return self._parse_ops(resp.content)

    @staticmethod
    def _format_drafts(drafts: List[CodingMemoryDraft]) -> str:
        rows = []
        for i, d in enumerate(drafts):
            rows.append(
                f"draft[{i}]:\n"
                f"  task: {d.task!r}\n"
                f"  search_keys: {d.search_keys}\n"
                f"  solution: {_truncate(d.solution, 800)!r}\n"
                f"  boundary_envs: {_truncate(d.boundary_envs, 400)!r}\n"
                f"  boundary_scope: {d.boundary_scope}"
            )
        return "\n\n".join(rows) if rows else "(none)"

    @staticmethod
    def _format_candidates(
        drafts: List[CodingMemoryDraft],
        candidates_per_draft: List[List[CodingMemory]],
    ) -> str:
        rows = []
        for i, d in enumerate(drafts):
            cands = candidates_per_draft[i] if i < len(candidates_per_draft) else []
            if not cands:
                rows.append(f"for draft[{i}]: (no candidates)")
                continue
            cand_rows = []
            for c in cands:
                cand_rows.append(
                    f"  - memory_id={c.memory_id}\n"
                    f"    task: {c.task!r}\n"
                    f"    solution: {_truncate(c.solution, 600)!r}\n"
                    f"    boundary_envs: {_truncate(c.boundary_envs, 300)!r}\n"
                    f"    boundary_scope: {c.boundary_scope}\n"
                    f"    updated_at: {c.updated_at.isoformat() if c.updated_at else 'n/a'}"
                )
            rows.append(f"for draft[{i}]:\n" + "\n".join(cand_rows))
        return "\n\n".join(rows) if rows else "(none)"

    @staticmethod
    def _parse_ops(text: str) -> List[ReconcileOp]:
        """宽松解析 ops 列表。"""
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

        obj: Optional[Dict[str, Any]] = None
        try:
            obj = json.loads(s)
        except Exception:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end > start:
                try:
                    obj = json.loads(s[start: end + 1])
                except Exception:
                    obj = None

        if not obj or not isinstance(obj, dict):
            logger.warning(f"[coding-reconcile] failed to parse ops output: {text!r}")
            return []
        ops_raw = obj.get("ops") or []
        if not isinstance(ops_raw, list):
            return []

        ops: List[ReconcileOp] = []
        for raw in ops_raw:
            if not isinstance(raw, dict):
                continue
            action = (raw.get("action") or "").strip().upper()
            if action not in ("ADD", "UPDATE", "DELETE", "SKIP"):
                continue
            di = raw.get("draft_idx")
            if di is not None:
                try:
                    di = int(di)
                except Exception:
                    di = None
            ops.append(ReconcileOp(
                action=action,  # type: ignore[arg-type]
                draft_idx=di,
                target_memory_id=raw.get("target_memory_id"),
                reason=raw.get("reason"),
            ))
        return ops

    # ------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------

    async def _execute(
        self,
        ops: List[ReconcileOp],
        drafts: List[CodingMemoryDraft],
    ) -> List[ReconcileOp]:
        """
        校验 + 执行 ops。返回成功执行的 ops（filter 掉无效 op）。

        校验规则：
        - ADD     需 draft_idx 合法
        - UPDATE  需 draft_idx 合法 + target_memory_id 非空
        - DELETE  需 target_memory_id 非空
        - SKIP    不执行任何 store 操作
        """
        executed: List[ReconcileOp] = []
        # 已被 UPDATE 占用的 draft_idx → 防止同 draft 被重复 ADD/UPDATE
        consumed: set = set()

        for op in ops:
            if op.action == "ADD":
                if op.draft_idx is None or not (0 <= op.draft_idx < len(drafts)):
                    logger.warning(f"[coding-reconcile] drop invalid ADD: {op.to_dict()}")
                    continue
                if op.draft_idx in consumed:
                    logger.warning(f"[coding-reconcile] drop ADD on consumed draft: {op.to_dict()}")
                    continue
                d = drafts[op.draft_idx]
                memory = self._draft_to_new_memory(d)
                try:
                    await self.store.insert(memory)
                    op.target_memory_id = memory.memory_id  # 回写 id 供 caller 读
                    executed.append(op)
                    consumed.add(op.draft_idx)
                except Exception as e:
                    logger.warning(f"[coding-reconcile] ADD failed: {e}")

            elif op.action == "UPDATE":
                if op.draft_idx is None or not (0 <= op.draft_idx < len(drafts)):
                    logger.warning(f"[coding-reconcile] drop invalid UPDATE: {op.to_dict()}")
                    continue
                if not op.target_memory_id:
                    logger.warning(f"[coding-reconcile] drop UPDATE missing target: {op.to_dict()}")
                    continue
                if op.draft_idx in consumed:
                    logger.warning(f"[coding-reconcile] drop UPDATE on consumed draft: {op.to_dict()}")
                    continue
                d = drafts[op.draft_idx]
                memory = self._draft_to_new_memory(d, memory_id=op.target_memory_id)
                try:
                    await self.store.update(memory)
                    executed.append(op)
                    consumed.add(op.draft_idx)
                except Exception as e:
                    logger.warning(f"[coding-reconcile] UPDATE failed: {e}")

            elif op.action == "DELETE":
                if not op.target_memory_id:
                    logger.warning(f"[coding-reconcile] drop DELETE missing target: {op.to_dict()}")
                    continue
                # DELETE 需要知道 user_id；通过 draft_idx（如有）或第一个 draft 推断
                user_id = self._infer_user_id(op, drafts)
                if not user_id:
                    logger.warning(f"[coding-reconcile] drop DELETE no user_id context: {op.to_dict()}")
                    continue
                try:
                    ok = await self.store.delete(op.target_memory_id, user_id=user_id)
                    if ok:
                        executed.append(op)
                except Exception as e:
                    logger.warning(f"[coding-reconcile] DELETE failed: {e}")

            elif op.action == "SKIP":
                if op.draft_idx is not None and 0 <= op.draft_idx < len(drafts):
                    consumed.add(op.draft_idx)
                executed.append(op)

        # 兜底：每个 draft 至少要被处理一次
        for i in range(len(drafts)):
            if i in consumed:
                continue
            logger.info(f"[coding-reconcile] draft[{i}] not handled, fallback ADD")
            d = drafts[i]
            memory = self._draft_to_new_memory(d)
            try:
                await self.store.insert(memory)
                executed.append(ReconcileOp(
                    action="ADD", draft_idx=i,
                    target_memory_id=memory.memory_id,
                    reason="fallback ADD (no LLM op)",
                ))
            except Exception as e:
                logger.warning(f"[coding-reconcile] fallback ADD failed: {e}")

        return executed

    @staticmethod
    def _infer_user_id(
        op: ReconcileOp, drafts: List[CodingMemoryDraft]
    ) -> Optional[str]:
        if op.draft_idx is not None and 0 <= op.draft_idx < len(drafts):
            return drafts[op.draft_idx].user_id
        # 后备：用 drafts 里第一个非空 user_id（同一 reconcile 调用通常同 user）
        for d in drafts:
            if d.user_id:
                return d.user_id
        return None

    @staticmethod
    def _draft_to_new_memory(
        d: CodingMemoryDraft,
        *,
        memory_id: Optional[str] = None,
    ) -> CodingMemory:
        return CodingMemory(
            memory_id=memory_id or str(uuid.uuid4()),
            user_id=d.user_id,
            agent_id=d.agent_id,
            task=d.task,
            search_keys=list(d.search_keys),
            solution=d.solution,
            boundary_envs=d.boundary_envs,
            boundary_scope=d.boundary_scope,
            workspace_id=d.workspace_id,
            branch=d.branch,
            session_id=d.session_id,
            files=list(d.files),
            confidence=d.confidence,
            source=d.source,
            type=None,
        )


# ================================================================
# Helpers
# ================================================================

def _truncate(s: str, n: int) -> str:
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."
