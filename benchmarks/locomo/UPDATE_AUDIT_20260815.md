# LoCoMo 25 次 UPDATE 审计（2026-08-15）

审计输入：`.runtime/locomo-ultra/20260813-150401/logs/pipeline/default/2026-08-13.log`
中的 25 条 `RECONCILE_APPLY / UPDATE` 记录，并结合会话顺序与最终 memory inventory
复核。

## 结论

- 确认属于兼容补充/合并：25
- 确认属于互斥状态变化、应改为 SUPERSEDE：0
- 无法判定：0

本次 `SUPERSEDE=0` 本身不是缺陷证据。25 次 UPDATE 都是在同一主题上增加细节，
没有出现城市、雇主、关系状态、固定日程或明确偏好反转等“新旧不能同时为真”的变化。
因此不对这批历史数据强行补链；否则会制造伪演化。

## 分组复核

| 目标记忆 | 次数 | 会话 | 内容变化 | 判定 |
|---|---:|---|---|---|
| `0832365e…` | 5 | 2、4、5、6、7 | 咨询/心理健康职业兴趣逐步补充为明确投入、服务跨性别人群及动机 | UPDATE 正确 |
| `c317927b…` | 4 | 7、10、13、14 | LGBTQ 权益、真实生活、发声与包容价值逐步合并 | UPDATE 正确 |
| `c9fa2d9e…` | 3 | 13、14、16 | 绘画疗愈、自我表达、转变经历等兼容细节合并 | UPDATE 正确 |
| `49cdbba2…` | 2 | 6、19 | 朋友、家人、导师支持网络增加细节 | UPDATE 正确 |
| `026d8dd7…` | 2 | 17、19 | 建立家庭、收养及单亲准备的兼容目标补充 | UPDATE 正确 |
| `df743ea5…` | 1 | 8 | 同一次 Pride parade 经历增加“最珍贵记忆”等感受 | UPDATE 正确 |
| `bae1d073…` | 1 | 11 | 艺术创作主题增加 LGBTQ/跨性别表达细节 | UPDATE 正确 |
| `f70f23ed…` | 1 | 12 | 与 Melanie 的友谊增加“支持圈”含义 | UPDATE 正确 |
| `1b0be48c…` | 1 | 14 | 支持性社交关系增加应对歧视的作用 | UPDATE 正确 |
| `1c0ef655…` | 1 | 18 | 重视亲友相处增加“家人给予力量”的解释 | UPDATE 正确 |
| `f0ba2c7e…` | 1 | 18 | 喜爱画花与亲近自然的兼容内容合并 | UPDATE 正确 |
| `c5535c4c…` | 1 | 19 | 自我接纳经历增加转变过程不易的细节 | UPDATE 正确 |
| `30d31be5…` | 1 | 19 | 用个人经历帮助 LGBTQ 社区的目标增加动机 | UPDATE 正确 |
| `47147caf…` | 1 | 19 | 真实表达自我的经历增加社区支持与自由感 | UPDATE 正确 |

## 本次代码修复的演化保障

1. Reconciler 的每个操作必须声明 `source_indices`，时间字段由对应的新记忆继承，
   避免合并时丢失事件时间。
2. 提示词明确禁止把不同时间发生的独立事件仅因主题相似而 UPDATE。
3. 地点、雇主、关系、日程、当前计划、偏好反转等时间变化状态必须 SUPERSEDE。
4. 如果模型输出 `UPDATE + state_transition=true`，解析器强制提升为 SUPERSEDE。
5. SUPERSEDE 先创建并回读新链头，再一次性闭合旧节点：
   `valid_until=transition_at`，新链头 `valid_from=transition_at`，并同时维护
   `supersedes/superseded_by/is_latest`。旧节点更新或回读核验失败时，
   回滚旧节点并删除未提交的新链头，同时把失败传播给上层。
6. 兼容 UPDATE 保留首次 `observed_at` 与原始 `memory_at`，最新观测写入
   `custom.observation_times/last_observed_at`，避免用一次补充覆盖整个时间来源。

## 后续回归用例

- “用户住在北京”→“用户已搬到上海”：必须 SUPERSEDE。
- “用户每周二、四学习”→“改为周一、三学习”：必须 SUPERSEDE。
- “用户喜欢绘画”→“绘画能帮助用户放松”：应 UPDATE。
- “用户上周参加 Pride”→“用户本周又参加 Pride”：应 ADD 两个事件，不能 UPDATE。
- “用户计划周五面试”→“面试已完成”：应形成可追溯状态变化，旧计划保留时间区间。

## 自动化核验

`tests/test_memory_correctness_fixes.py` 已覆盖 UPDATE→SUPERSEDE 强制提升、
时间 schema 序列化、SUPERSEDE 成功闭链和失败回滚。当前结果：
`10 passed`，其中包含 Chroma 时间字段落库回读、Sweeper 行为抽象失败
不再被吞掉，以及真实临时 Kuzu 数据库的双向 `RELATED_TO` 写入、
幂等性与 L7 时间字段回读核验。
