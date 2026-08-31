# MemoWeave

面向长期陪伴与个人智能助理场景的个人长期记忆系统实验项目。系统将跨会话对话加工为可检索、可演化、可追溯的用户记忆，并建立了覆盖写入、检索、回答和证据溯源的 LoCoMo 评测链路。

## 当前能力

- 分层记忆：原始会话、基础画像、原子事实、未来意图和高阶行为 Schema；
- 快慢双路径：System 1 在线抽取与 reconcile，System 2 跨会话归纳；
- 混合存储：Chroma 保存语义记忆，Kuzu 保存 L6 Schema、事实依赖与可追溯实体关系；
- 记忆演化：时间字段、`ADD / UPDATE / SUPERSEDE` 和双向版本链；
- 可解释检索：L2/L7 保留原始 session / turn provenance；
- 系统评测：Accuracy、Exact Hit@K、Recall@K、provenance 覆盖率和延迟统计。

## 评测基线

在 LoCoMo 9 段长对话、253 个 session、500 道固定分层抽样题上：

| 指标 | 结果 |
|---|---:|
| Accuracy | 65.40% |
| turn-level Exact Hit@5 | 74.01% |
| turn-level Recall@5 | 64.57% |
| provenance 覆盖率 | 98.84% |
| Single-hop Accuracy | 75.00% |
| Adversarial Accuracy | 90.62% |

这是固定 seed 的项目评测，不是 LoCoMo 官方全量榜单结果。详细口径与分类结果见 [500 题评测报告](benchmarks/locomo/RESULTS_HELDOUT_500_20260816.md)。

## 目录

```text
hy-memory-1.2.21/hy_memory/  核心记忆系统源码
scripts/                     运行与评测脚本
tests/                       回归测试
benchmarks/locomo/           评测清单、说明和汇总报告
RUN_ULTRA.md                 本地运行指南
GRAPH_MEMORY_SIMPLE_DESIGN.md GraphRAG 简版设计稿
```

本仓库不会提交 `.env`、API Key、虚拟环境、运行数据库、原始评测结果 JSON、wheel 安装包和 LoCoMo 原始数据。

## 本地运行

环境准备、模型配置和 Ultra 端到端运行方式见 [RUN_ULTRA.md](RUN_ULTRA.md)。

运行回归测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Entity-Fact GraphRAG

`feat/graphrag-memory` 分支实现了实体事实图：由 L1 原始对话抽取显式候选关系，经 L2 reconcile 审核后发布为可信 `Assertion`，再以向量事实/query 实体为种子执行受限两跳扩展，并根据 `FactRef` 回到 Chroma 获取 L2 原文。

默认关闭；建议先启用 Shadow 模式观察路径，不改变回答结果：

```env
MEMORY_GRAPHRAG_ENABLED=true
MEMORY_GRAPHRAG_SHADOW_MODE=true
```

详细数据模型、过滤规则、A/B 指标和启用方式见 [Graph 记忆链路简版设计](GRAPH_MEMORY_SIMPLE_DESIGN.md)。

## 开源来源与许可

本项目以 MIT 许可发布的 `hy-memory 1.2.21` Python wheel 为基础进行学习、修复和扩展。仓库保留原项目版权与许可证声明；新增评测、可观测性、时间语义、provenance、演化链和后续 GraphRAG 代码属于本项目的二次开发内容。

详见 [LICENSE](LICENSE)。LoCoMo 数据遵循其上游许可，原始数据不包含在本仓库中。
