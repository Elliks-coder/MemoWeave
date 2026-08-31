# -*- coding: utf-8 -*-
"""
CodingCurator — 渐进式披露的 coding memory 写入 agent。

与 legacy CodingWriter (extractor + reconciler) 通过客户端参数 coding_writer="legacy"|"agent"
共存做 AB；接口签名 100% 一致，调用方无侵入。

详见 /root/.claude-internal/plans/squishy-leaping-orbit.md
"""

from .curator import CodingCuratorWriter

__all__ = ["CodingCuratorWriter"]
