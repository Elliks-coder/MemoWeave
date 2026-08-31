# -*- coding: utf-8 -*-
"""
Coding Memory - Scene Judge

两个职责（同模块、共用 LLMConfig.model）：

1. classify_messages_is_coding(messages, llm)
   写入端：对整段 messages 做单一二分判定（is_coding ∈ {True, False}）
   不切 segment、不分 turn —— 整段一条链路。

2. classify_queries_is_coding(queries, llm)
   搜索端：以 queries 中末尾为目标 query，前置 queries 作为上下文，判定 is_coding。

详见 docs/coding_memory_mvp_design.md §6.3 / §8.3。
"""

import json
import logging
from typing import List, Optional, Dict, Any

from ..agent.llm_provider import LLMProvider
from ..pipelines.base import ChatMessage
from .preproc import extract_tool_summary

logger = logging.getLogger(__name__)


def _resolve_llm_temperature(llm_provider: LLMProvider, default: float = 0.1) -> float:
    """
    从 MemoryConfig.llm.temperature 解析温度（与 MemAgent 路径一致）。

    LLMProvider._llm_config 是 agent 内部 LLMConfig，历史上不含 temperature；
    优先读 config.llm.temperature，避免 fallback 到 0.1 触发 kimi 等平台硬约束。
    """
    config = getattr(llm_provider, "config", None)
    if config is not None:
        llm = getattr(config, "llm", None)
        temp = getattr(llm, "temperature", None) if llm is not None else None
        if temp is not None:
            return float(temp)
    inner = getattr(llm_provider, "_llm_config", None)
    temp = getattr(inner, "temperature", None) if inner is not None else None
    if temp is not None:
        return float(temp)
    return float(default)


# ================================================================
# 写入端：整段二分类
# ================================================================

CLASSIFY_MESSAGES_PROMPT = """\
You are a single-label scene classifier. Decide the DOMINANT scene of the entire
conversation chunk passed to you.

Output exactly one of:
  "coding"  ← the chunk is CLEARLY about a concrete software/code task: writing /
              reading / debugging / refactoring code, working with source files,
              a specific error or stack trace, API / library / framework usage,
              or building / testing / deploying code. Tool calls are doing real
              code work, and code instructions / conventions / decisions emerge.
  "chat"    ← everything else: casual conversation, personal info, preferences,
              general or factual Q&A that is not about code, life / work topics.
              Tool calls (if any) are incidental and not code work.

Rules:
- Look at the WHOLE chunk, not individual turns. A few stray off-topic turns
  inside a clearly code-focused chunk should still be "coding".
- A few stray tool calls inside a clearly chat chunk should still be "chat".
- Non-code technical/ops chit-chat, or merely mentioning a tool/product without
  a concrete code task, is "chat".
- When in doubt, prefer "chat" (we'd rather miss a coding extraction than
  pollute coding memory with chat content).

Conversation summary (turns with user query + tool names used):
{turns_block}

Output strict JSON only, no markdown:
{{"is_coding": true/false, "reason": "..."}}
"""


async def classify_messages_is_coding(
    messages: List[ChatMessage],
    llm_provider: LLMProvider,
    *,
    max_tokens: int = 200,
    temperature: Optional[float] = None,
) -> bool:
    """
    对整段 messages 做单一 is_coding 判定。

    复用 LLMConfig.model（不引入独立 classifier 配置）。
    temperature=None 时使用 config.llm.temperature（与 MemAgent 路径一致），
    避免硬编码 0.0 触发 kimi/deepseek 等模型的温度硬约束。
    fail-safe：LLM 调用失败或解析失败，返回 False（fallback 走 chat 链）。
    """
    if not messages:
        return False

    summary = extract_tool_summary(messages)
    turns_block = _format_turns(summary)
    prompt = CLASSIFY_MESSAGES_PROMPT.format(turns_block=turns_block)

    if temperature is None:
        temperature = _resolve_llm_temperature(llm_provider)

    try:
        resp = await llm_provider.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result = _parse_is_coding_json(resp.content)
        if result is None:
            logger.warning(
                f"[coding-judge] failed to parse classify_messages output: {resp.content!r}; defaulting to chat"
            )
            return False
        is_coding = bool(result.get("is_coding", False))
        logger.info(
            f"[coding-judge] is_coding={is_coding} reason={result.get('reason', '')!r}"
        )
        return is_coding
    except Exception as e:
        logger.warning(f"[coding-judge] classify_messages LLM failed: {e}; defaulting to chat")
        return False


# ================================================================
# 搜索端：用 queries 上下文判类
# ================================================================

CLASSIFY_QUERIES_PROMPT = """\
Classify the LATEST query as either "coding" or "chat".
Use the earlier queries (if any) as context for disambiguation.

Be STRICT: only mark "coding" when the target is CLEARLY about a concrete
software/code task, e.g. writing / reading / debugging / refactoring code,
a specific error, stack trace or exception, an API / library / framework usage,
build / test / deployment of code, or a decision about the codebase itself.

Everything else is "chat", including: casual conversation, personal info,
preferences, general factual Q&A, life / work topics that are not code, and
vague references ("that thing we discussed") with no clear code signal.
When in doubt, choose "chat".

Recent queries (oldest → newest, last is target):
{queries_block}

Output strict JSON only, no markdown:
{{"is_coding": true/false}}
"""


async def classify_queries_is_coding(
    queries: List[str],
    llm_provider: LLMProvider,
    *,
    max_tokens: int = 100,
    temperature: Optional[float] = None,
) -> bool:
    """
    对一组 queries（末尾为目标）判类。

    单 query 输入也支持（前置上下文为空）。
    temperature=None 时使用 config.llm.temperature。
    fail-safe：LLM 调用失败或解析失败，返回 False（走现有 chat 召回）。
    """
    if not queries:
        return False

    queries_block = "\n".join(
        f"{i + 1}. {repr(q)}" + ("   ← target" if i == len(queries) - 1 else "")
        for i, q in enumerate(queries)
    )
    prompt = CLASSIFY_QUERIES_PROMPT.format(queries_block=queries_block)

    if temperature is None:
        temperature = _resolve_llm_temperature(llm_provider)

    try:
        resp = await llm_provider.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result = _parse_is_coding_json(resp.content)
        if result is None:
            logger.warning(
                f"[coding-judge] failed to parse classify_queries output: {resp.content!r}; defaulting to chat"
            )
            return False
        is_coding = bool(result.get("is_coding", False))
        logger.info(f"[coding-judge] queries is_coding={is_coding}")
        return is_coding
    except Exception as e:
        logger.warning(f"[coding-judge] classify_queries LLM failed: {e}; defaulting to chat")
        return False


# ================================================================
# 搜索端：极简判类 + 改写（单次 LLM，搜索路径必须低时延）
# ================================================================

CLASSIFY_AND_REWRITE_PROMPT = """\
Task: Given queries (oldest -> newest, last is target), do TWO things:
1) Classify the target as 0 (chat) or 1 (coding). Be STRICT: use 1 ONLY when the
   target is CLEARLY about a concrete software/code task — writing / reading /
   debugging / refactoring code, a specific error or stack trace, API / library /
   framework usage, build / test / deploy of code, or a decision about the
   codebase itself. Everything else (casual talk, personal info, preferences,
   non-code factual Q&A, vague references with no code signal) is 0.
   When in doubt, choose 0.
2) Rewrite the target into ONE concise standalone query, expanding pronouns /
   restoring omitted context using the earlier queries. Keep the same language.

Output STRICTLY in this format, nothing else:
<0 or 1>
<rewritten query>

Queries:
{queries_block}
"""


async def classify_and_rewrite_queries(
    queries: List[str],
    llm_provider: LLMProvider,
    *,
    max_tokens: int = 200,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    单次 LLM 调用同时做：
    1) 判定 target query (queries[-1]) 是 coding(1) / chat(0)
    2) 用前置 queries 上下文做改写（消除指代 / 补全省略），返回 rewrite_query

    输出格式刻意保持极简（避免 JSON 解析开销 + 让 LLM 输出短）：
        第一行: 0 或 1
        第二行: rewritten query

    Returns:
        {"is_coding": bool, "rewrite_query": str, "ok": bool}
        失败时 ok=False，is_coding 默认 False，rewrite_query 等于 queries[-1]

    fail-safe：LLM 失败 / 解析失败一律 ok=False（caller 应回退默认 chat 路径）。
    """
    target = queries[-1] if queries else ""
    fallback = {"is_coding": False, "rewrite_query": target, "ok": False}
    if not queries:
        return fallback

    queries_block = "\n".join(
        f"- {q}" + ("   <-- target" if i == len(queries) - 1 else "")
        for i, q in enumerate(queries)
    )
    prompt = CLASSIFY_AND_REWRITE_PROMPT.format(queries_block=queries_block)

    if temperature is None:
        temperature = _resolve_llm_temperature(llm_provider)

    try:
        resp = await llm_provider.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        logger.warning(f"[coding-judge] classify_and_rewrite LLM failed: {e}; falling back to chat")
        return fallback

    text = (resp.content or "").strip()
    if not text:
        logger.warning("[coding-judge] classify_and_rewrite empty content; falling back to chat")
        return fallback

    # 去 markdown fence（保险）
    if text.startswith("```"):
        lines = text.split("\n")
        body = []
        in_fence = False
        for line in lines:
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            body.append(line)
        text = "\n".join(body).strip() or text

    # 拆解：第一行 = 0/1，剩下的 = rewrite
    lines = text.split("\n", 1)
    head = lines[0].strip().rstrip(",.").rstrip("。，")
    rewrite = (lines[1].strip() if len(lines) > 1 else "").strip()
    if not rewrite:
        rewrite = target  # 没改写就用原 target

    # 容错：LLM 可能输出 "0/1" 周围带空格、引号、句号
    if head in ("1", "true", "True", "TRUE", "yes", "Yes", "YES", "y", "Y"):
        is_coding = True
    elif head in ("0", "false", "False", "FALSE", "no", "No", "NO", "n", "N"):
        is_coding = False
    else:
        c = head[:1] if head else ""
        if c == "1":
            is_coding = True
        elif c == "0":
            is_coding = False
        else:
            logger.warning(
                f"[coding-judge] classify_and_rewrite unrecognized head={head!r}; falling back to chat"
            )
            return {"is_coding": False, "rewrite_query": rewrite, "ok": False}

    logger.info(
        f"[coding-judge] classify_and_rewrite is_coding={is_coding} "
        f"rewrite={rewrite!r}"
    )
    return {"is_coding": is_coding, "rewrite_query": rewrite, "ok": True}


# ================================================================
# 搜索端：合并分析（判类 + 改写 + 时间跨度），单次 LLM
#
# 支持三种组合（由 caller 通过 want_classify / want_rewrite 控制）：
#   - want_classify + want_rewrite : working memory + query rewrite 同开，一次 LLM
#   - want_rewrite only            : 只 query rewrite（恒 chat / 不判类）
#   - want_classify only 走旧的 classify_and_rewrite_queries / classify_queries_is_coding
#
# 时间：LLM 只输出相对锚点（today / N days ago / an ISO date），由本函数结合
# current_time 换算成 created_after（unix 秒，下界）。无上界。
# ================================================================

ANALYZE_QUERIES_PROMPT = """\
You rewrite a user's memory-search query. Given queries (oldest -> newest, last is the target){classify_clause}.

Current time: {current_time}

Do:
{task_lines}
R) Rewrite the target into ONE short, clear, standalone query. Resolve pronouns and
   restore omitted context from earlier queries. Keep the same language. No time words
   in the rewrite (time goes in SINCE).
S) SINCE: if the target refers to a time range (e.g. "yesterday", "last week",
   "刚才", "上个月"), output the START date as YYYY-MM-DD. If none, output NONE.

Output STRICTLY these lines, nothing else:
{output_lines}

Queries:
{queries_block}
"""


async def analyze_queries(
    queries: List[str],
    llm_provider: LLMProvider,
    *,
    want_classify: bool,
    want_rewrite: bool,
    current_time: str = "",
    max_tokens: int = 220,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    单次 LLM：按需 (判类) + 改写 + 时间跨度。

    Args:
        want_classify: 是否要 is_coding（working memory 开启时）
        want_rewrite:  是否要 rewrite + timespan（query rewrite 开启时）
        current_time:  系统当前时间（ISO），用于把相对时间换算成 created_after

    Returns:
        {
          "is_coding": bool,          # want_classify=False 时恒 False
          "rewrite_query": str,       # 失败/未改写时 = queries[-1]
          "created_after": float|None,# SINCE 换算的 unix 秒下界，无则 None
          "since_raw": str,           # LLM 原始 SINCE（调试用）
          "ok": bool,
        }
    fail-safe：LLM / 解析失败 → ok=False，is_coding=False，rewrite=target，created_after=None
    """
    target = queries[-1] if queries else ""
    fallback = {
        "is_coding": False, "rewrite_query": target,
        "created_after": None, "since_raw": "", "ok": False,
    }
    if not queries or not want_rewrite:
        # 不需要 rewrite 就不该调这个函数；防御性返回 fallback
        return fallback

    queries_block = "\n".join(
        f"- {q}" + ("   <-- target" if i == len(queries) - 1 else "")
        for i, q in enumerate(queries)
    )

    if want_classify:
        classify_clause = ""
        task_lines = (
            "C) Classify the target as 0 (chat) or 1 (coding). Be STRICT: use 1 ONLY\n"
            "   when the target is CLEARLY about a concrete software/code task —\n"
            "   writing / reading / debugging / refactoring code, a specific error or\n"
            "   stack trace, API / library / framework usage, build / test / deploy of\n"
            "   code, or a decision about the codebase itself. Everything else is 0\n"
            "   (casual talk, personal info, preferences, non-code Q&A, vague refs).\n"
            "   When in doubt, choose 0.\n"
        )
        output_lines = "<0 or 1>\n<rewritten query>\nSINCE: <YYYY-MM-DD or NONE>"
    else:
        classify_clause = ""
        task_lines = ""
        output_lines = "<rewritten query>\nSINCE: <YYYY-MM-DD or NONE>"

    prompt = ANALYZE_QUERIES_PROMPT.format(
        classify_clause=classify_clause,
        current_time=current_time or "(unknown)",
        task_lines=task_lines,
        output_lines=output_lines,
        queries_block=queries_block,
    )

    if temperature is None:
        temperature = _resolve_llm_temperature(llm_provider)

    try:
        resp = await llm_provider.complete(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
        )
    except Exception as e:
        logger.warning(f"[coding-judge] analyze_queries LLM failed: {e}; fallback")
        return fallback

    text = (resp.content or "").strip()
    if not text:
        return fallback

    # 去 markdown fence
    if text.startswith("```"):
        body = [ln for ln in text.split("\n") if not ln.startswith("```")]
        text = "\n".join(body).strip() or text

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return fallback

    is_coding = False
    idx = 0
    if want_classify:
        head = lines[0].strip().rstrip(",.").rstrip("。，")
        c = head[:1]
        if c == "1":
            is_coding = True
        elif c == "0":
            is_coding = False
        else:
            logger.warning(
                f"[coding-judge] analyze_queries unrecognized head={head!r}; ok=False"
            )
            return {**fallback, "since_raw": ""}
        idx = 1

    # 找 SINCE 行；其余行拼成 rewrite
    since_raw = ""
    rewrite_lines: List[str] = []
    for ln in lines[idx:]:
        s = ln.strip()
        if s.upper().startswith("SINCE:"):
            since_raw = s[len("SINCE:"):].strip()
        else:
            rewrite_lines.append(s)
    rewrite = " ".join(rewrite_lines).strip() or target

    created_after = _since_to_created_after(since_raw, current_time)

    logger.info(
        f"[coding-judge] analyze is_coding={is_coding} rewrite={rewrite!r} "
        f"since={since_raw!r} created_after={created_after}"
    )
    return {
        "is_coding": is_coding, "rewrite_query": rewrite,
        "created_after": created_after, "since_raw": since_raw, "ok": True,
    }


def _since_to_created_after(since_raw: str, current_time: str) -> Optional[float]:
    """把 LLM 输出的 SINCE（YYYY-MM-DD 或 NONE）换算成 unix 秒下界。"""
    if not since_raw:
        return None
    s = since_raw.strip().strip('"').strip("'").rstrip(".,。，")
    if not s or s.upper() in ("NONE", "N/A", "NULL", "-"):
        return None
    from datetime import datetime
    # 只取前 10 个字符（YYYY-MM-DD），容忍 LLM 带时间
    candidate = s[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt.timestamp()
        except (ValueError, TypeError):
            continue
    logger.warning(f"[coding-judge] analyze_queries unparsable SINCE={since_raw!r}")
    return None


# ================================================================
# Helpers
# ================================================================

def _format_turns(summary: List[Dict[str, Any]]) -> str:
    """渲染 turn 简化视图给 LLM 看。"""
    if not summary:
        return "(empty)"
    rows = []
    for t in summary:
        user = t.get("user", "") or ""
        # 单 turn 用户文本截到 200 字符
        if len(user) > 200:
            user = user[:200] + "..."
        tools = t.get("tools", []) or []
        rows.append(
            f"- turn {t.get('turn', '?')}: user={user!r}, tools={tools}"
            + (", has_tool_result=True" if t.get("has_tool_result") else "")
        )
    return "\n".join(rows)


def _parse_is_coding_json(text: str) -> Optional[Dict[str, Any]]:
    """
    宽松解析 LLM 输出。先试纯 JSON，再试在文本里找第一个 {...} 块。
    返回 None 表示无法解析。
    """
    if not text:
        return None
    text = text.strip()
    # 去 markdown fence
    if text.startswith("```"):
        # ```json ... ```
        lines = text.split("\n")
        # 去掉首尾 fence
        body_lines = []
        in_fence = False
        for line in lines:
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or not body_lines and line.startswith("```"):
                continue
            body_lines.append(line)
        text = "\n".join(body_lines).strip() or text

    # 直接 JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "is_coding" in obj:
            return obj
    except Exception:
        pass

    # 找第一个 { ... } 块
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        snippet = text[start: end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict) and "is_coding" in obj:
                return obj
        except Exception:
            return None
    return None
