# MemoWeave GraphRAG 配对 A/B 试验

评测日期：2026-08-31（Asia/Shanghai）

运行 ID：`20260831-133027`

原始结果：`results/locomo-graphrag-ab-20260831-133027.json`（本地文件，不提交 Git）

## 1. 试验设计

- 数据：LoCoMo `conv-26` 的完整 19 个 session；
- 题目：25 题分层开发集，每类 5 题；
- Baseline：Profile + Proactive/L7 + Normal/L2 三路召回；
- Graph：完全相同的三路召回，再加入 Entity-Fact GraphRAG 候选；
- 两个方案共用同一份写入后的记忆、同一模型、同一问题和阈值；
- Normal 最终仍固定为 Top-10，Graph 需要参与同一预算内的融合，不能靠扩大上下文取得优势；
- 模型：`deepseek-v4-pro`；Embedding：`qwen3.7-text-embedding`。

该试验用于发现工程问题，不是正式 held-out 成绩，也不替代 500 题基线。

## 2. A/B 结果

| 指标 | 三路召回 | 三路召回 + Graph | 差值 |
|---|---:|---:|---:|
| Accuracy | 72.00% | 72.00% | 0.00 pp |
| Exact Hit@1 | 40.00% | 40.00% | 0.00 pp |
| Exact Hit@5 | 80.00% | 80.00% | 0.00 pp |
| Exact Hit@10 | 85.00% | 85.00% | 0.00 pp |
| Recall@1 | 31.25% | 31.25% | 0.00 pp |
| Recall@5 | 62.50% | 62.50% | 0.00 pp |
| Recall@10 | 72.50% | 72.50% | 0.00 pp |

25 道题中，Graph 使 0 题由错变对、0 题由对变错。7 道题生成了 9 条图候选，
但最终只有 1 道题的 1 条 Graph 事实进入回答上下文，候选进入率仅 11.11%。
5 道 Multi-hop 题中有 2 道生成图候选，但进入最终上下文的数量为 0，因此
Multi-hop Accuracy 和 Recall@K 均没有变化。

本轮固定先运行 baseline、再运行 graph，第二次搜索会受 embedding/vector cache
预热影响，因此 `1500.58 ms` 与 `242.87 ms` 的搜索耗时不能用于证明 Graph 更快。
脚本已改为后续按题目交替执行顺序。

## 3. 图结构与关系质量

- 19/19 个 session 写入成功，System 2 digest 成功；
- L2 操作：119 ADD、22 UPDATE、0 SUPERSEDE；
- 写入阶段发布关系 26 次，graph failure 为 0；
- 最终 Kuzu 中有 24 个 Entity、25 个 active Assertion、28 个 EpisodeRef；
- Assertion 的 Subject/Object/FactRef 三类必需边完整率 100%；
- 原始 turn provenance 可解析率 100%；
- 独立关系裁判检查 25 条 active assertion，显式支持率为 84.00%；
- 有 2 组相同三元组由不同 L2 事实重复支持，没有发现同时 active 的冲突型功能关系。

84% 不是可直接扩大启用范围的关系精度。主要低质量关系包括：

- 把 `The user's friends`、`The user's family` 当作独立实体，形成信息量很低的
  `用户 --FRIEND_OF/FAMILY_OF--> 泛化群体`；
- 根据亲密对话推断 `用户 --FRIEND_OF--> Melanie`，但原话没有明确声明朋友关系；
- 同一语义关系按 Fact ID 生成多个 Assertion，能够保留多证据，但也会把近义 L2
  重新召回，降低上下文多样性。

图片 caption 已纳入复审。`用户 --CREATED--> painting of a tree with a bright sun`
来自 D9:14 的图片描述，不属于关系幻觉。

## 4. 当前实现的核心问题

### 4.1 用户中心星型拓扑与遍历规则冲突

25 条 active assertion 中有 24 条以 `用户` 为 subject；同时检索实现为了避免
用户超级节点造成无关扩散，不允许 `__user__` 作为跨跳桥梁。结果是绝大多数关系
成为互不连通的星型叶子。当前产生的图候选主要来自“同一外部实体被重复提及”，
而不是跨实体的真正多跳推理。

LoCoMo 问题通常使用人名 Caroline，而图中该角色被归一为 `__user__`，query 实体
种子又主动忽略 `__user__`。因此 query 人名、用户节点和事实图之间也没有形成有效入口。

### 4.2 Graph 分数很难进入固定 Top-K

当前分数为：

```text
graph_score = anchor_score × 0.9^hop × relation_confidence
```

Graph 分数天然低于锚点；向量候选已经通过 0.4 阈值且存在大量 over-fetch 结果，
Graph 候选与其直接按原始分数竞争时大多会在 Top-10 截断阶段被淘汰。本次 9 条候选
只有 1 条进入上下文。题 47 已找到 2 条候选、题 65 已找到 1 条候选，但全部被截掉。

### 4.3 进入上下文的唯一 Graph 事实相关性不足

题 81 问“Caroline 是否会很快搬回祖国”，Graph 补入的是“朋友、家人和导师支持她
转型”的事实。它与问题只有宽泛人物相关性，没有补到参考答案所需的“正在收养孩子”，
最终 baseline 和 graph 都回答错误。这说明即使为 Graph 预留名额，也必须先增加
query/path 相关性校验，不能直接强插。

### 4.4 当前关系 Schema 对多跳问题覆盖不足

关系白名单偏人物、地点和兴趣；LoCoMo 多跳答案还需要 `SUPPORTED_BY`、
`INSPIRED_BY`、状态变化、事件参与及事件时间等关系。只抽取当前 25 种实体关系，
无法覆盖大量“经历 → 原因 → 结果”型问题。

## 5. 结论与下一步

本轮结论是：**当前三路 + Graph 没有优于三路召回，不应立即进行 500 题正式评测。**
Kuzu 的真实写入、边完整性、租户范围和 provenance 没有发现结构性假成功；问题位于
关系语义和召回融合层。

建议按以下顺序修复，并仍先做 25 题配对试验：

1. 建立 `Caroline ↔ __user__` 的受控别名入口，并设计有限制的 user-hub 遍历；
2. 拒绝或规范化 `The user's friends/family` 等泛化伪实体，增加 relation-level
   grounding 校验；
3. 改用 RRF 或为通过 query/path 相关性校验的 Graph 事实预留 1–2 个 normal 名额，
   同时保持总 Top-K 不变；
4. 增加事件节点和 `SUPPORTED_BY / INSPIRED_BY / CAUSED_BY` 等通用关系；
5. 以“Graph 候选进入率、Graph 事实 precision、Multi-hop Recall@5、paired accuracy
   regression”为门槛，通过小样本后再跑 held-out 500。
