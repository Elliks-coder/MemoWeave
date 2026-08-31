# -*- coding: utf-8 -*-
"""
Pipeline Log Writer — 环节级 JSONL 日志（始终开启，与 SQLite Trace 分离）。

目录: {log_dir}/{subdir}/{date}.log
每行: request_id, user_id, agent_id, step, prompt, content, parsed, tokens, elapsed_ms, timestamp
  （prompt = 真实请求 prompt，含 EXTRACT 的 [SYSTEM]/[USER]、RECONCILE 的 existing_memories，便于 debug）

子目录（subdir）来源（SDK 不感知"租户/app_id"等业务概念，只做纯字符串切分）：
  1. 从该条记录的 user_id 取第一个 "__" 左侧作为子目录；
     多租户场景下，上层（App）会把 user_id 拼成 "{prefix}__{user_id}"，
     于是不同租户的日志天然分到不同子目录（也避免单文件过大、便于采集）；
  2. user_id 无 "__" 前缀（如单人直接使用 SDK）→ 回落到构造时传入的
     default_subdir（默认 "default"）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# 兼容旧 import
TraceFileWriter = None  # set after class definition


class PipelineLogWriter:
    """按 {subdir}/{date}.log 追加 JSONL；best-effort，失败静默。

    subdir 解析（SDK 层不感知 app_id/租户业务概念，仅做字符串前缀切分）：
      1. 从该条记录的 user_id 取第一个 "__" 左侧作为 subdir
         （多租户下上层把 user_id 拼成 "{prefix}__{user_id}"，日志天然分开，
          同时避免单个日志文件过大影响采集）；
      2. user_id 无 "__" 前缀 → 回落到构造时传入的 default_subdir；
      3. 再回落 "default"。
    """

    def __init__(self, log_dir: str, default_subdir: str = "default"):
        self._log_dir = log_dir
        self._default_subdir = default_subdir or "default"
        try:
            (Path(self._log_dir) / self._default_subdir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _resolve_subdir(self, user_id: str) -> str:
        """从 user_id 的 "{prefix}__{user_id}" 前缀取 prefix 作为子目录；无前缀则回落。"""
        if user_id and "__" in user_id:
            prefix = user_id.split("__", 1)[0].strip()
            if prefix:
                return prefix
        return self._default_subdir

    def write_step(
        self,
        *,
        request_id: str = "",
        user_id: str = "",
        agent_id: str = "",
        session_id: str = "",
        step: str = "",
        prompt: str = "",
        response: str = "",
        parsed: str = "",
        memory_ids: Optional[List[str]] = None,
        elapsed_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        event_at_ms: int = 0,
        **kwargs: Any,
    ) -> None:
        try:
            from .pipeline_observability import split_app_user

            tokens = (prompt_tokens or 0) + (completion_tokens or 0) + (total_tokens or 0)
            # user_id 按 "__" 拆出 app_id（SDK 不感知 app_id，上层多租户拼接约定）
            app_id, uid = split_app_user(user_id or "")
            # timestamp = step 发生时刻（可读 ISO ms）。优先用上层统一捕获的 event_at_ms
            # （与 DB 的 event_at_ms 同源，保证跨 JSONL/DB 对账），缺省回落当前时刻。
            if event_at_ms:
                _ts = datetime.fromtimestamp(event_at_ms / 1000.0).isoformat(timespec="milliseconds")
            else:
                _ts = datetime.now().isoformat(timespec="milliseconds")
            record = {
                "request_id": request_id or "",
                "app_id": app_id,
                "user_id": uid,
                "agent_id": agent_id or "",
                "session_id": session_id or "",
                "step": step or "",
                # 完整写入，不截断（prompt / response / parsed 均保留全文，便于排障）
                "prompt": prompt or "",
                "content": response or "",
                "parsed": parsed or "",
                "tokens": tokens,
                "elapsed_ms": round(float(elapsed_ms or 0), 2),
                "timestamp": _ts,
            }
            if memory_ids:
                record["memory_ids"] = list(memory_ids)

            line = json.dumps(record, ensure_ascii=False, default=str)
            date_str = datetime.now().strftime("%Y-%m-%d")
            subdir = self._resolve_subdir(user_id)
            log_dir = Path(self._log_dir) / subdir
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{date_str}.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.debug(f"[pipeline-log] write failed: {e}")


# 向后兼容
TraceFileWriter = PipelineLogWriter
