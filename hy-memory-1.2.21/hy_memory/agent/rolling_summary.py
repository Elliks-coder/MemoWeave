"""
HY Memory - Rolling Summary（topic-aware 滚动摘要）

与 summarizer.py 的「单次 add 即 summary」不同，本模块面向 buffer 攒够（按 user 轮次）
后的批量摘要：

  - 输入 prev_summary（同 buffer_key 当天最近一条 L3 文本，可空）+ 当前 pending turns。
  - LLM 把 turns 按 topic 分组，对「有价值且足够」的组生成 summary（可延续 prev_summary 的
    topic），对「有价值但还不够」的 turn 放 keep（留 buffer 等下次），对「无价值」的 turn 丢弃。
  - 每条 summary 落一条独立 L3 节点（无 chain）；溯源用 raw_id list（粗粒度）。

turn 标识：每个 turn 在 prompt 里用一个临时展示行号 `line`（0..N-1），代码持 line→turn 映射，
LLM 只引用 line，避免拼复合键出错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import logging

from .llm_provider import LLMProvider
from ..config import LLMConfig as GlobalLLMConfig

logger = logging.getLogger(__name__)


# ================================================================
# 数据结构
# ================================================================

@dataclass
class RollingSummaryItem:
    """LLM 产出的单条 summary。"""
    content: str
    topic: str = ""
    line_ids: List[int] = field(default_factory=list)   # 本条覆盖的展示行号
    is_continuation: bool = False                         # 是否延续 prev_summary 的 topic


@dataclass
class RollingSummaryResult:
    success: bool
    summaries: List[RollingSummaryItem] = field(default_factory=list)
    keep_line_ids: List[int] = field(default_factory=list)  # 留 buffer 等下次的行号
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    _actual_prompt: Optional[str] = None


# ================================================================
# Prompt
# ================================================================

ROLLING_SUMMARY_PROMPT = """You maintain a user's daily conversation summaries, organized by topic.

You are given:
1. PREVIOUS SUMMARY — the most recent summary for this user today (may be empty).
2. NEW TURNS — recent conversation turns, each prefixed with a line number [n].

Your job: group the NEW TURNS by topic and decide, for each group:
- If it is valuable AND substantial enough → write a summary for it. If the group continues
  the same topic as PREVIOUS SUMMARY, REWRITE an updated summary that merges the previous
  summary with the new turns (set "is_continuation": true).
- If it is valuable but NOT yet substantial enough to summarize → put those line numbers in
  "keep" (they stay buffered for next time).
- If it is trivial / not worth remembering → simply omit those lines (discard them).

PREVIOUS SUMMARY:
---
{prev_summary}
---

NEW TURNS:
---
{turns}
---

Memory date: {memory_date}
Current date: {current_date}

Summary writing rules (same as standard summaries):
1. Third-person: "The user ...". 2. 1-3 sentences each, max 200 words. 3. Preserve
preferences/decisions/changes. 4. Self-contained. 5. No fabrication. 6. Output language
MUST match the conversation language. 7. Resolve relative time against Memory date.

## Output contract — STRICT JSON, no markdown/code fence:
{{
  "summary_list": [
    {{"content": "<summary text>", "topic": "<short topic label>", "line_ids": [<int>, ...], "is_continuation": <true|false>}}
  ],
  "keep_line_ids": [<int>, ...]
}}

Rules for the output:
- Every line number appears in AT MOST one place (one summary's line_ids, or keep_line_ids,
  or omitted entirely if discarded). Never repeat a line in two groups.
- line_ids / keep_line_ids must reference only line numbers present in NEW TURNS.
- If nothing is worth summarizing yet, return {{"summary_list": [], "keep_line_ids": [...]}}.
Output JSON only."""


# ================================================================
# RollingSummarizer
# ================================================================

class RollingSummarizer:
    """topic-aware 滚动摘要生成器（buffer 触发时调用）。"""

    def __init__(
        self,
        llm_provider: LLMProvider,
        llm_config: Optional[GlobalLLMConfig] = None,
    ):
        self.llm = llm_provider
        self._llm_config = llm_config or GlobalLLMConfig()
        self._call_count = 0
        self._total_tokens = 0

    async def summarize(
        self,
        *,
        prev_summary: str,
        turns: List[Dict[str, Any]],
        current_time: str = "",
    ) -> RollingSummaryResult:
        """
        Args:
            prev_summary: 同 buffer_key 当天最近一条 L3 摘要文本（可空）。
            turns: 待摘要的 turn 列表，每个 {"raw_id","turn_idx","role","content"}。
                   本方法用列表下标作为展示行号 line（0..N-1）。
            current_time: 记忆发生时间（ISO 字符串）。

        Returns:
            RollingSummaryResult；summaries[*].line_ids / keep_line_ids 均为 turns 的下标。
        """
        if not turns:
            return RollingSummaryResult(success=True)

        try:
            from datetime import datetime as _dt, date as _date
            _current_date = _date.today().isoformat()
            if current_time:
                try:
                    _memory_date = _dt.fromisoformat(current_time).strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    _memory_date = _current_date
            else:
                _memory_date = _current_date

            # 渲染带行号的 turns
            lines = []
            for i, t in enumerate(turns):
                role = t.get("role", "user")
                content = (t.get("content") or "").strip()
                lines.append(f"[{i}][{role}]: {content}")
            turns_text = "\n".join(lines)

            # 语言选择
            from ..utils.lang_detect import is_chinese
            joined = "\n".join((t.get("content") or "") for t in turns)
            if is_chinese(joined):
                from .prompts_zh import ROLLING_SUMMARY_PROMPT_ZH
                prompt = ROLLING_SUMMARY_PROMPT_ZH.format(
                    prev_summary=prev_summary or "(none)",
                    turns=turns_text,
                    memory_date=_memory_date,
                    current_date=_current_date,
                )
            else:
                prompt = ROLLING_SUMMARY_PROMPT.format(
                    prev_summary=prev_summary or "(none)",
                    turns=turns_text,
                    memory_date=_memory_date,
                    current_date=_current_date,
                )

            response = await self.llm.complete(
                prompt=prompt,
                max_tokens=self._llm_config.agent_max_tokens,
                temperature=self._llm_config.temperature,
            )
            self._call_count += 1
            self._total_tokens += response.tokens_used

            parsed = self._parse(response.content, n_turns=len(turns))
            parsed.tokens_used = response.tokens_used
            parsed.prompt_tokens = response.prompt_tokens
            parsed.completion_tokens = response.completion_tokens
            parsed._actual_prompt = prompt
            return parsed

        except Exception as e:
            logger.error(f"RollingSummarizer.summarize failed: {e}")
            return RollingSummaryResult(success=False, error=str(e))

    def _parse(self, raw: str, *, n_turns: int) -> RollingSummaryResult:
        """解析 LLM JSON 输出，并做行号合法性校验 + 去重。

        校验失败（重叠 / 越界 / 非法 JSON）→ 保守降级为「全部 keep」，不丢数据、不误 summary。
        """
        def _degrade(reason: str) -> RollingSummaryResult:
            logger.warning(f"[rolling-summary] output invalid ({reason}); keep all turns")
            return RollingSummaryResult(
                success=True,
                summaries=[],
                keep_line_ids=list(range(n_turns)),
            )

        text = (raw or "").strip()
        if "```" in text:
            # 去 code fence
            try:
                text = text.split("```json")[1].split("```")[0].strip()
            except Exception:
                try:
                    text = text.split("```")[1].split("```")[0].strip()
                except Exception:
                    pass
        try:
            data = json.loads(text)
        except Exception:
            return _degrade("json parse")

        if not isinstance(data, dict):
            return _degrade("not an object")

        seen: set = set()
        summaries: List[RollingSummaryItem] = []
        for item in (data.get("summary_list") or []):
            if not isinstance(item, dict):
                return _degrade("summary item not object")
            content = (item.get("content") or "").strip()
            ids_raw = item.get("line_ids") or []
            ids: List[int] = []
            for x in ids_raw:
                try:
                    xi = int(x)
                except (ValueError, TypeError):
                    return _degrade("line_id not int")
                if xi < 0 or xi >= n_turns:
                    return _degrade("line_id out of range")
                if xi in seen:
                    return _degrade("line_id overlap")
                seen.add(xi)
                ids.append(xi)
            if not content or not ids:
                # 空摘要或无覆盖行：跳过该条（不致命）
                continue
            summaries.append(RollingSummaryItem(
                content=content,
                topic=(item.get("topic") or "").strip(),
                line_ids=ids,
                is_continuation=bool(item.get("is_continuation", False)),
            ))

        keep: List[int] = []
        for x in (data.get("keep_line_ids") or []):
            try:
                xi = int(x)
            except (ValueError, TypeError):
                return _degrade("keep id not int")
            if xi < 0 or xi >= n_turns:
                return _degrade("keep id out of range")
            if xi in seen:
                return _degrade("keep id overlaps summary")
            seen.add(xi)
            keep.append(xi)

        # seen 之外的行 = LLM 主动丢弃的无价值 turn（合法，不报错）
        return RollingSummaryResult(success=True, summaries=summaries, keep_line_ids=keep)

    def get_stats(self) -> Dict[str, Any]:
        return {"call_count": self._call_count, "total_tokens": self._total_tokens}
