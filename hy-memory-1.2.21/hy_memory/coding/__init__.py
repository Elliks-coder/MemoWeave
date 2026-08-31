# -*- coding: utf-8 -*-
"""
HY Memory - Coding Memory 模块

为生产力 / 编码场景设计的独立记忆链路：
- 独立 schema (CodingMemory)
- 独立物理存储（SQLite meta + VDB keys）
- 独立 extractor / reconciler / store
- 与现有 chat 链路彻底分轨，chat 路径零回归

详见 docs/coding_memory_mvp_design.md
"""

from .types import (
    CodingMemory,
    CodingMemoryDraft,
    ReconcileOp,
    BoundaryScope,
    BOUNDARY_SCOPES,
)

__all__ = [
    "CodingMemory",
    "CodingMemoryDraft",
    "ReconcileOp",
    "BoundaryScope",
    "BOUNDARY_SCOPES",
]
