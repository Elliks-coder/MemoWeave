# -*- coding: utf-8 -*-
"""
Coding Memory 数据结构

设计原则：
- 与 chat 链路完全分轨，独立 schema
- task + search_keys 多 key 召回
- boundary_scope 为一等字段，决定 memory 复用范围
- 覆盖式 update（不留历史），可携带 DELETE op

详见 docs/coding_memory_mvp_design.md §4 / §15
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal


# boundary_scope 取值（typing.Literal 校验 + 运行时常量）
BOUNDARY_SCOPES = ("strict", "project", "user", "global")
BoundaryScope = Literal["strict", "project", "user", "global"]


# ================================================================
# 草稿（LLM extract 阶段输出）
# ================================================================

@dataclass
class CodingMemoryDraft:
    """
    LLM extractor 产出的草稿，未经 reconcile / 持久化。

    字段语义见 docs/coding_memory_mvp_design.md §6.5。
    """
    # 检索主键
    task: str
    search_keys: List[str] = field(default_factory=list)
    solution: str = ""
    boundary_envs: str = ""

    # 边界（LLM 给出）
    boundary_scope: BoundaryScope = "project"
    confidence: float = 0.7

    # 上下文（SDK 注入）
    user_id: str = ""
    agent_id: str = "default_agent"
    workspace_id: Optional[str] = None
    branch: Optional[str] = None
    session_id: Optional[str] = None
    files: List[str] = field(default_factory=list)
    source: str = "auto_extract"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "search_keys": list(self.search_keys),
            "solution": self.solution,
            "boundary_envs": self.boundary_envs,
            "boundary_scope": self.boundary_scope,
            "confidence": self.confidence,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "workspace_id": self.workspace_id,
            "branch": self.branch,
            "session_id": self.session_id,
            "files": list(self.files),
            "source": self.source,
        }


# ================================================================
# 已持久化记忆
# ================================================================

@dataclass
class CodingMemory:
    """已持久化的 coding memory（SQLite + VDB 双层共同表示一条）"""
    # 主键
    memory_id: str
    user_id: str
    agent_id: str = "default_agent"

    # 检索主键
    task: str = ""
    search_keys: List[str] = field(default_factory=list)
    solution: str = ""
    boundary_envs: str = ""

    # 边界
    boundary_scope: BoundaryScope = "project"
    workspace_id: Optional[str] = None
    branch: Optional[str] = None

    # 上下文
    session_id: Optional[str] = None
    files: List[str] = field(default_factory=list)

    # 元数据
    confidence: float = 0.7
    source: str = "auto_extract"
    type: Optional[str] = None  # 保留字段，MVP 不启用

    # 时间
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "task": self.task,
            "search_keys": list(self.search_keys),
            "solution": self.solution,
            "boundary_envs": self.boundary_envs,
            "boundary_scope": self.boundary_scope,
            "workspace_id": self.workspace_id,
            "branch": self.branch,
            "session_id": self.session_id,
            "files": list(self.files),
            "confidence": self.confidence,
            "source": self.source,
            "type": self.type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ================================================================
# Reconcile op
# ================================================================

ReconcileAction = Literal["ADD", "UPDATE", "DELETE", "SKIP"]


@dataclass
class ReconcileOp:
    """
    Reconciler 决策出的单个操作。

    - ADD     ← 由某 draft 触发新增；draft_idx 必填，target_memory_id 不填
    - UPDATE  ← 由某 draft 触发更新；draft_idx + target_memory_id 都必填
    - DELETE  ← 新内容明确否定/作废了某 candidate；draft_idx 记录"由哪条 draft 触发"
                可空（少数情况），target_memory_id 必填
    - SKIP    ← duplicate；draft_idx 必填，target_memory_id 不填

    注：一次 reconcile 调用产出的 ops 数 *可以超过* drafts 数（一个 draft 可能伴随多条 DELETE）。
    """
    action: ReconcileAction
    draft_idx: Optional[int] = None
    target_memory_id: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "draft_idx": self.draft_idx,
            "target_memory_id": self.target_memory_id,
            "reason": self.reason,
        }
