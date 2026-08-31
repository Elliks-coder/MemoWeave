"""
中文版 Prompt 定义。

当检测到用户输入为中文时，使用此文件中的 prompt 替代英文版，
确保 LLM 输出为中文。
"""

from typing import Dict, List, Optional

# ================================================================
# EXTRACT PROMPT (中文版)
# ================================================================
# System prompt（固定，不参与 format）
EXTRACT_SYSTEM_PROMPT_ZH = """你是一个记忆提取系统（Memory Extractor）。

你的任务是从对话中提取：

* 持久的
* 高价值的
* 对未来对话有帮助的

用户记忆，并将其转换为结构化、可检索、自包含的记忆。

---

# 输入内容

你将收到：

* 最近 k 条消息（Last k Messages）
* 新消息（New Messages）
* Memory Date
* Current Date

用途：

* Last k Messages 仅用于：

  * 指代消解
  * 上下文理解
* Memory Date 是唯一允许用于解析时间表达的时间锚点
* Current Date 仅作为系统元数据，不用于时间推理
* 当 New Messages 含 `turn_id` 时，每条 memory/intention 必须在
  `source_turn_ids` 中引用支撑该条目的原始 ID。只引用直接证据，既不能遗漏
  支撑回合，也不能编造、改写 ID；不得引用 Last k Messages 的 ID。

---

# 提取原则

只提取 NEW user messages 中首次出现的新信息。

优先提取：

* 长期偏好
* 兴趣爱好
* 长期项目
* 未来计划
* 决策
* 约束条件
* 行为习惯
* 价值观
* 稳定态度
* 重复性活动
* 重要经历
* 用户接受的建议

不要提取：

* 寒暄
* 客套话
* 一次性情绪
* 临时上下文
* assistant 对旧 memory 的复述
* 已经明显存在于历史上下文的信息
* 模糊的人格评价

---

# 用户主体规则

每条 memory 必须：

* 自包含
* 可脱离上下文理解
* 明确以用户为主体
* 使用第三人称

中文使用：

* “用户…”

英文使用：

* “The user …”

---

# 语言规则

* 输出语言必须与输入语言一致
* 保留原始术语、品牌名、公式、代码
* 不要转写或翻译专有名词

---

# 两类记忆

你提取两类记忆：**memory（记忆）** 和 **intentions（意图）**。

## memory（记忆）

一切值得长期记住的用户信息都放这里：

* 偏好、观点、态度、价值观
* 事件、行动、经历
* 习惯、行为模式
* 关系、稳定特质
* 客观事实、决策、约束条件
* 正在进行的活动

如果用户隐式表达偏好，也应提取：

* 主动恢复某个活动 → 正向兴趣
* 长期避免某事 → 负向偏好
* 重复主动投入 → 持续兴趣

不要把"用户是怎样的人"和"发生了什么"分开。两者都放进 `memory`。

## intentions（意图）

用户**未来打算/计划去做**的、可被执行的、一旦完成或过期就失去意义的前瞻事项。

例如：

* "用户下周二要面试。"
* "用户打算月底搬家。"
* "用户计划周五前完成报告。"

不属于意图（这些放进 `memory`）：

* 稳定偏好/特质（"用户喜欢篮球"）
* 已完成的过去事件（"用户去年搬到了北京"）
* 没有明确未来行动的泛泛目标

### valid_until（有效截止日）

每条 intention 给出 `valid_until`：该意图失效的日期（截止日/目标日），
基于 **Memory Date** 解析相对时间（如"下周二"→对应的绝对日期）。
如果无法确定截止日，则设为 `null`。

---

# OWNER（归属）

每条 memory 和 intention 都必须带 `owner`，标明它属于谁：

* `"user"` —— 关于用户的事实/偏好/特质，用户陈述的或用户自己要做的事。
* `"agent"` —— assistant 提供的信息（推荐、确认、查到的资料），
  或用户希望 **assistant** 未来去做的事（如"8 小时后提醒我" → 一条 agent 意图）。

逐条判定 `owner`。无法确定时默认 `"user"`。

---

# 时间规则

使用 Memory Date 解析：

* 明天
* 下周
* 上个月
* 最近

等相对时间。

每条 memory 和 intention 都必须返回以下时间字段：

* `observed_at`：ISO 格式的 Memory Date，表示这句话何时被观测到
* `temporal_relation`：`past`、`present`、`future`、`recurring`、`atemporal` 或 `unknown`
* `event_time_text`：原始时间表达；未提到时间时为 `null`
* `event_start` / `event_end`：标准化后的 ISO 日期/时间；无法确定时为 `null`
* `normalization_confidence`：0 到 1；无法解析时为 0
* `valid_from` / `valid_until`：该状态或计划的有效区间；无法确定时为 `null`

如果新消息是对上一句问题的直接回答，必须继承上一句的时间范围。例如：
“今年暑假有什么计划？”→“研究收养机构”必须保留 `event_time_text: "今年暑假"`
和 `temporal_relation: "future"`。

如果无法安全解析，保留 `event_time_text` 原文，把标准化字段设为 `null` 并降低
`normalization_confidence`。不得仅因无法标准化就删除时间证据。

不要编造时间顺序。

---

# 质量要求

每条 memory 必须：

* 信息完整
* 自包含
* 适合 retrieval
* 高信息密度

保留重要细节：

* 人名
* 产品名
* 地点
* 数字
* 公式
* 约束条件

避免：

* 缺少上下文的碎片化记忆
* 纯情绪噪音
* 无长期价值的信息

---

# TAGS

每条 memory 必须包含 1–3 个 tags。

要求：

* 小写
* 简洁
* 主题明确

例如：

* ai
* work
* travel
* gaming
* research
"""

EXTRACT_PROMPT_ZH = """## 最近 k 条消息

{last_messages}

---

## 新消息

{content}

---

## Memory Date

{memory_at}

---

## Current Date

{current_date}

---


# 输出格式

仅返回一个 JSON 对象。

格式：

```json
{{
  "memory": [
    {{
      "content": "...",
      "tags": ["..."],
      "owner": "user",
      "source_turn_ids": ["原始turn-id"],
      "observed_at": "YYYY-MM-DDTHH:MM:SS",
      "temporal_relation": "past|present|future|recurring|atemporal|unknown",
      "event_time_text": null,
      "event_start": null,
      "event_end": null,
      "normalization_confidence": 0.0,
      "valid_from": null,
      "valid_until": null
    }}
  ],
  "intentions": [
    {{
      "content": "...",
      "tags": ["..."],
      "valid_until": "YYYY-MM-DD",
      "owner": "user",
      "source_turn_ids": ["原始turn-id"],
      "observed_at": "YYYY-MM-DDTHH:MM:SS",
      "temporal_relation": "future",
      "event_time_text": "...",
      "event_start": "YYYY-MM-DD",
      "event_end": "YYYY-MM-DD",
      "normalization_confidence": 0.0,
      "valid_from": "YYYY-MM-DD"
    }}
  ],
  "basic_info": {{}}
}}
```

规则：

* memory：
  一切值得长期记忆的信息——偏好、观点、态度、特质、事件、计划、经历、习惯、决策

* intentions：
  用户未来打算/计划去做的前瞻事项，附 `valid_until`（截止日/目标日，
  基于 Memory Date 解析）；无法确定截止日时设为 `null`

* owner（每条 memory 和 intention 都必填）：`"user"` 或 `"agent"`
  （见上方"# OWNER"；无法确定时默认 `"user"`）

* source_turn_ids（每条 memory 和 intention 都必填）：填写 New Messages 中
  直接支撑该条目的原始 `turn_id`。必须原样复制、覆盖全部直接证据且不夹带无关
  回合；绝不编造。仅当输入完全没有 ID 时才使用 `[]`。

* basic_info：
  仅当 NEW user messages 中**明确**陈述稳定画像字段（如 姓名、年龄、所在地、职业、雇主）时才填写。
  绝不推断、绝不编造。没有任何字段被明确陈述时输出 `{{}}`。

* 不要跨 section 重复信息

* 每条 memory/intention 都必须包含全部时间字段；未知值使用 `null`，不能省略。
  `observed_at` 默认使用 Memory Date。

* tags 必须为 1–3 个小写关键词

* 没有内容时使用 [] 或 {{}}

* 仅输出合法 JSON

现在请输出JSON内容"""


GRAPH_RELATION_SYSTEM_PROMPT_ZH = """

---

# 用于图记忆的实体关系

启用 GraphRAG 关系抽取时，每个 `memory` 条目都要增加 `relations` 数组。
关系必须由 NEW messages 明确陈述；仅在同一段出现、主题相近、猜测出的社交
关系，以及只存在于 Last k Messages 的事实，都不能作为关系。

每条关系必须包含：

* `subject`、`object`：可独立理解的规范实体名称，不能使用未消解的代词；
* `subject_type`、`object_type`：person、place、organization、event、
  product、activity、animal、concept 或 other；
* `predicate`：只能使用 KNOWS、FRIEND_OF、PARTNER_OF、FAMILY_OF、
  PARENT_OF、CHILD_OF、SIBLING_OF、COLLEAGUE_OF、WORKS_AT、STUDIES_AT、
  MEMBER_OF、LIVES_IN、FROM、LOCATED_IN、LIKES、DISLIKES、OWNS、USES、
  CREATED、PARTICIPATED_IN、ATTENDED、VISITED、HAS_PET、CARES_FOR、
  INTERESTED_IN、LEARNING 或 RELATED_TO；
* `confidence`：0 到 1；
* `evidence_type`：固定为 `explicit`，不要输出推断关系。

关系涉及用户本人时统一使用“用户”。没有明确且有价值的实体连接时输出空数组。
"""


GRAPH_RELATION_OUTPUT_PROMPT_ZH = """

GraphRAG 扩展：`memory` 中的每个对象都必须包含 `relations`。
示例：`{"relations":[{"subject":"用户","subject_type":"person",
"predicate":"COLLEAGUE_OF","object":"小林","object_type":"person",
"confidence":0.95,"evidence_type":"explicit"}]}`。
没有明确实体关系时使用 `"relations": []`。
"""


# ================================================================
# EXTRACT FEW-SHOT 示例（中文版）
# ================================================================
# 仅在 few_shot_enabled 时附加到 system prompt 末尾。
# 每个示例都贴合真实输入结构（最近 k 条消息 / 新消息 / Memory Date）
# 与输出 schema（memory / intentions / basic_info，含 owner 与 tags），
# 并在结尾给出一句要点点评。
EXTRACT_FEW_SHOT_ZH = """
---

# 示例

## 示例 1 —— 边界：旧消息只用于指代消解，绝不重复提取

最近 k 条消息：
[{"role": "user", "content": "我最近在市中心的收容所看流浪狗。"},
 {"role": "assistant", "content": "真好——有看中的吗？"}]
新消息：
[{"role": "user", "content": "嗯，昨天把她接回来了——一只三岁的边牧，叫露娜。"}]
Memory Date：2025-04-10

输出：
```json
{
  "memory": [
    {"content": "用户在2025-04-09从市中心的收容所领养了一只三岁的边境牧羊犬，名叫露娜。", "tags": ["pets", "family"], "owner": "user"}
  ],
  "intentions": [],
  "basic_info": {}
}
```
旧消息只用于消解"她"并补充上下文（"市中心收容所"）。不要再输出"用户在考虑领养狗"这种旧信息，只提取真正的新事实（已完成的领养），并用 Memory Date 把"昨天"解析成绝对日期。

## 示例 2 —— 浓缩 agent 的长篇回复

新消息：
[{"role": "user", "content": "帮我写一篇短文，论述为什么远程办公能提升效率。"},
 {"role": "assistant", "content": "好的——开头：'远程办公正在重塑团队的协作方式……'【中间约2000字正文省略】……结尾：'总之，自主性和更少的打断带来了更高的产出。'"}]
Memory Date：2025-03-22

输出：
```json
{
  "memory": [
    {"content": "assistant 为用户写了一篇短文，论点是远程办公因自主性更强、打断更少而提升效率。", "tags": ["writing", "remote-work"], "owner": "agent"}
  ],
  "intentions": [],
  "basic_info": {}
}
```
agent 的回复很长。绝不原样存储整篇短文，而要提炼成"主题+论点"。owner 为 "agent"，因为内容由 assistant 产出。臃肿的原文会让 memory 不可用。

## 示例 3 —— 复合提取：basic_info + intention + 概括后的 agent 计划

新消息：
[{"role": "user", "content": "你好，我叫健司。我下个月打算去日本旅行，能帮我做一个7天的行程吗？"},
 {"role": "assistant", "content": "# 日本7天经典黄金路线（东京 → 京都 → 奈良 → 大阪）\\n飞抵东京，乘新干线南下京都，奈良一日游，最后从大阪关西机场离境。\\n\\n## 第1天：抵达东京\\n- 降落羽田/成田，在东京站附近酒店放行李。\\n- 傍晚：浅草寺，逛仲见世商店街，尝鲷鱼烧。\\n- 晚上：涩谷十字路口 + 涩谷天空看日落；居酒屋晚餐（烤串、清酒）。\\n\\n## 第2天：东京市场与地标 —— 【略】\\n## 第3天：新干线到京都，伏见稻荷，祇园 —— 【略】\\n## 第4天：京都寺庙（清水寺、金阁寺、银阁寺）—— 【略】\\n## 第5天：奈良一日游（奈良公园喂鹿、东大寺）—— 【略】\\n## 第6天：大阪（大阪城、道顿堀街头美食）—— 【略】\\n## 第7天：大阪观光后从关西机场离境 —— 【略】\\n\\n小贴士：建议出发前买7天JR Pass。需要我调整吗？"}]
Memory Date：2025-09-15

输出：
```json
{
  "memory": [
    {"content": "assistant 为用户制定了一份7天日本经典黄金路线行程：东京（浅草、涩谷）、新干线到京都（伏见稻荷、清水寺、金阁寺、祇园）、奈良一日游（喂鹿公园、东大寺），最后大阪（大阪城、道顿堀），从关西机场离境。", "tags": ["travel", "japan", "itinerary"], "owner": "agent"}
  ],
  "intentions": [
    {"content": "用户计划在2025年10月去日本旅行7天。", "tags": ["travel", "japan"], "valid_until": "2025-10-31", "owner": "user"}
  ],
  "basic_info": {"name": "健司"}
}
```
一个轮次产出三种不同 bucket：姓名进 basic_info；旅行是带截止日的前瞻 intention；assistant 那份很长的多段行程被概括成**一条**高密度 agent memory——只保留路线骨架和各城市的关键景点，而**不是**逐日的细节、贴士或美食清单。原样存整篇回复会让 memory 臃肿，存骨架才可检索。

## 示例 4 —— 后续消息中用户采纳/确认了先前建议，本身就是事实

新消息：
[{"role": "user", "content": "我家6岁的孩子一点蔬菜都不肯吃，有什么办法吗？"},
 {"role": "assistant", "content": "可以把菠菜或胡萝卜打碎拌进意面酱里——颜色和味道基本尝不出来，很多孩子就这么接受了。"},
 {"role": "user", "content": "好主意，今晚就试试。对了，这个年纪几点睡比较好？"},
 {"role": "assistant", "content": "6岁的孩子建议晚上7到9点睡，保证大约10到11小时睡眠。"}]
Memory Date：2025-06-20

输出：
```json
{
  "memory": [
    {"content": "用户有一个6岁的孩子，不肯吃蔬菜，用户采纳了 assistant 的建议——把菠菜或胡萝卜打碎拌进意面酱里以掩盖蔬菜。", "tags": ["parenting", "food"], "owner": "user"},
    {"content": "assistant 建议用户：6岁孩子应在晚上7到9点入睡，保证约10到11小时睡眠。", "tags": ["parenting", "sleep"], "owner": "agent"}
  ],
  "intentions": [],
  "basic_info": {}
}
```
用户在第三条消息说"今晚就试试"——明确采纳了先前的建议。这条采纳信息很有价值，必须记录（"用户采纳了……"），不能当寒暄丢掉。睡眠建议刚给出、尚未被采纳，因此记为 assistant 给出的建议（owner "agent"）。

## 示例 5 —— RAG/注入上下文：绝不存储被注入的文本

新消息：
[{"role": "user", "content": "【检索到的资料：埃菲尔铁塔高330米，1889年建成。】基于以上，我打算今年夏天在埃菲尔铁塔顶上向女友求婚。"}]
Memory Date：2025-05-02

输出：
```json
{
  "memory": [],
  "intentions": [
    {"content": "用户计划在2025年夏天于埃菲尔铁塔顶上向女友求婚。", "tags": ["relationship", "travel"], "valid_until": "2025-08-31", "owner": "user"}
  ],
  "basic_info": {}
}
```
方括号内检索到的资料（铁塔高度、建成年份）是 RAG 注入的参考材料，并不是用户讲述的关于自己的信息——绝不存储。只提取用户自己的前瞻计划。系统/RAG 注入的文本只是给模型的上下文，永远不是 memory。

## 示例 6 —— 用户分享资料：提取资料内容，而非分享动作

新消息：
[{"role": "user", "content": "帮我记住我奶奶的饺子配方，别弄丢了。馅料：猪肉糜500克，大白菜300克，生抽2勺，香油1勺，姜末一小块。白菜先用盐腌15分钟再挤干水。下锅煮，点两次凉水，饺子浮起来就好了。"},
 {"role": "assistant", "content": "好的，记下了！"}]
Memory Date：2025-07-01

输出：
```json
{
  "memory": [
    {"content": "用户奶奶的饺子配方：馅料为猪肉糜500克、大白菜300克（先用盐腌15分钟再挤干水）、生抽2勺、香油1勺、姜末一小块；下锅煮、点两次凉水，饺子浮起即熟。", "tags": ["food", "recipe", "family"], "owner": "user"}
  ],
  "intentions": [],
  "basic_info": {}
}
```
提取真正的事实内容（食材、用量、腌挤步骤、煮法），而不是"用户分享了配方"或"用户要求记住配方"。分享动作没有价值，内容才是可检索的。

## 示例 7 —— 结构化数值与属性完整保留

新消息：
[{"role": "user", "content": "记一下我这个周期的家庭健身房 PR：深蹲 140kg x 3，卧推 100kg x 5，硬拉 180kg x 2，推举 60kg x 6。体重 78kg，每个动作 3 组，组间休息 3 分钟。"},
 {"role": "assistant", "content": "记好了——数据很强！"}]
Memory Date：2026-01-20

输出：
```json
{
  "memory": [
    {"content": "用户当前的力量 PR 为深蹲 140kg x 3、卧推 100kg x 5、硬拉 180kg x 2、推举 60kg x 6，体重 78kg。", "tags": ["fitness", "strength"], "owner": "user"},
    {"content": "用户的训练安排为每个动作 3 组、组间休息 3 分钟。", "tags": ["fitness", "training"], "owner": "user"}
  ],
  "intentions": [],
  "basic_info": {}
}
```
每个数值和单位（kg、次数、组数、休息时间）都完整保留——丢掉它们会毁掉最可检索的信息。拆成两个清晰主题（PR 与训练方案），但绝不取整或省略具体数值。

## 示例 8 —— 多主题长对话：不要只提取第一个话题

新消息：
[{"role": "user", "content": "这周很重要——我周二终于通过了博士论文答辩！"},
 {"role": "assistant", "content": "太恭喜了！感觉怎么样？"},
 {"role": "user", "content": "如释重负！我还在柏林签了一套公寓的租约，下个月搬过去。另外我开始在多邻国学德语了——已经连续打卡30天。我的伴侣索菲娅暂时还留在马德里，这点挺难受的。"}]
Memory Date：2025-11-08

输出：
```json
{
  "memory": [
    {"content": "用户在2025-11-04通过了博士论文答辩。", "tags": ["education", "milestone"], "owner": "user"},
    {"content": "用户在柏林签了一套公寓的租约。", "tags": ["housing"], "owner": "user"},
    {"content": "用户在多邻国学德语，已连续打卡30天。", "tags": ["language", "learning"], "owner": "user"},
    {"content": "用户的伴侣索菲娅暂时留在马德里，用户觉得这点挺难受。", "tags": ["relationship"], "owner": "user"}
  ],
  "intentions": [
    {"content": "用户计划下个月（2025年12月左右）搬到柏林。", "tags": ["housing", "relocation"], "valid_until": "2025-12-31", "owner": "user"}
  ],
  "basic_info": {}
}
```
四个以上话题——答辩、柏林公寓、学德语、伴侣在马德里。不要只提取第一个话题。注意一个轮次拆出了已完成的事实（已签租约）和一条 intention（即将搬家），关于索菲娅的情绪虽带情绪但含长期关系信息，因此保留。

---
"""


# ================================================================
# SEARCH_QUERY PROMPT (中文版)
# ================================================================

SEARCH_QUERY_PROMPT_ZH = """你是一个搜索查询生成器。给定一组新提取的记忆，生成一组简短的搜索查询，用于在向量数据库中找到相关的已有记忆。

目标是最大化召回率——即使措辞差异很大，也要找到与新记忆语义相关的已有记忆。

## 新记忆:
{new_memories}

## 指引

生成的搜索查询应覆盖:
- 记忆中提到的关键主题、实体和主题
- 核心概念的改写或抽象版本
- 用户记忆库中可能存在的相关概念

输出一个 JSON 字符串数组（5-15条查询，简短聚焦）:

["查询1", "查询2", "查询3", ...]

只输出 JSON 数组，不要有其他文字。"""


# ================================================================
# RECONCILE PROMPT (中文版)
# ================================================================


RECONCILE_PROMPT_ZH = """你是一个记忆整合系统（Memory Reconciler）。

你的任务是将新的 memories 整合进已有 memory base，同时保持：

* 信息无损
* 无矛盾
* 高可检索性
* 结构紧凑
* 避免碎片化

---

# 输入内容

你将收到：

* Existing memories
* New memories
* Current date

---

# Existing memories

每条 existing memory 包含：

* memory_id
* content
* owner
* memory_at
* tags

---

# New memories

每条 new memory 包含：

* source_index
* content
* owner
* tags

new memories 没有 ID。

---

# 操作类型

---

## ADD

当信息：

* 完全新
* 与已有 memory 不冲突
* 不属于已有主题

时使用。

格式：

```json
{{
  "op": "ADD",
  "content": "...",
  "source_indices": [0],
  "owner": "user",
  "tags": ["..."]
}}
```

---

## UPDATE

当：

* 新旧 memory 描述同一主题
* 且没有冲突
* 合并后能提高完整性或可检索性

时使用。

要求：

* 无损合并
* 保留所有有效信息
* 不要产生重复 memory

格式：

```json
{{
  "op": "UPDATE",
  "memory_id": "...",
  "content": "...",
  "source_indices": [0],
  "owner": "user",
  "tags": ["..."]
}}
```

---

## SUPERSEDE

仅在以下情况使用：

* 新 memory 与旧 memory 在同一维度冲突
* 两者不能同时为真
* 旧 memory 已不再代表当前状态

例如：

* 用户换城市
* 用户换工作
* 用户关系状态变化
* 用户长期偏好发生明确反转

不要用于：

* 增加细节
* wording 差异
* 同义表达
* 可共存偏好

要求：

* content 只能写“新的状态”
* 不要复制旧信息
* 系统会自动保留旧节点并接到演化链上，你无需关心历史版本

格式：

```json
{{
  "op": "SUPERSEDE",
  "memory_id": "...",
  "content": "...",
  "source_indices": [0],
  "supersede_reason": "...",
  "owner": "user",
  "tags": ["..."]
}}
```

### 把多条旧记忆折叠进同一条链

如果**两条或更多**已有记忆描述的是**同一维度**，而新记忆是该维度上的最新变化，
你可以把它们折叠进同一条演化链。此时用 `memory_ids`（有序列表，**旧→新**）
代替单个 `memory_id`：

```json
{{
  "op": "SUPERSEDE",
  "memory_ids": ["<最旧的id>", "<较新的id>"],
  "content": "<只写最新的状态>",
  "supersede_reason": "...",
  "owner": "user",
  "tags": ["..."]
}}
```

列出的旧记忆会按旧→新连成一条链，新记忆成为链头。`content` 仍然只描述**最新状态**——
**绝不**把旧记忆的内容抄进来；旧内容保留在链的历史里，依然可被检索。

---

# 关键规则

---

## 信息无损

所有有效信息必须仍然可检索。

不要丢失：

* 新 memory 中的信息
* 被 UPDATE/SUPERSEDE 的旧信息

注意：

SUPERSEDE 不需要复制旧内容，因为旧节点仍存在。

---

## 重复检测

如果 new memory 已经被 existing memory 完整覆盖：

不要输出任何操作。

---

## 避免碎片化

如果多个 memory：

* 讨论同一主题
* 内容兼容

优先 UPDATE 合并。

---

## Granularity

每条 memory 应只表达：

* 一个完整主题
* 一个清晰状态

避免：

* 过碎
* 过长
* 多主题混杂

---

## owner 规则

`owner`（`user`/`agent`）：从被写入的 memory 原样拷贝。
**绝不跨 owner 合并或 supersede** —— 用户的事实和 assistant 的事实即使主题相同也是两条不同记忆。

---

## 时间规则

`memory_at: null`

表示时间未知。

不要编造时间顺序。

每个操作都必须输出 `source_indices`，引用 new memories 中的 `source_index`。
必须保留这些新记忆的结构化时间字段；不能仅因主题相似就把不同时间发生的独立事件合并。

UPDATE 仅用于同一状态或同一事件的兼容补充。地点、雇主、关系、固定日程、正在执行的
计划或偏好反转等随时间变化的状态一旦改变，必须使用 SUPERSEDE，使旧状态的有效区间
仍然可以查询。如果不能确认新旧状态可以同时成立，不得用 UPDATE 覆盖。

不要仅因为“看起来更新”就 supersede。

---

## NO FORK RULE

* SUPERSEDE 要么指向一条 existing memory（`memory_id`），要么把**同一维度**的多条
  memory 折叠进同一条链（`memory_ids`，旧→新）；UPDATE 只能指向一条 existing memory。
* 同一条 existing memory 不得被多个操作 touch。
* 不要创建分叉 supersede 链。

---

{few_shot_section}
###################
下面是输入信息!
###################


当前日期：{current_date}

## 已有记忆（Existing memories）

{existing_memories}

## 新记忆（New Memories）

{new_memories}

# 输出格式

输出一个 JSON array。

例如：

```json
[
  {{
    "op": "ADD",
    "content": "...",
    "owner": "user",
    "tags": ["..."]
  }},
  {{
    "op": "UPDATE",
    "memory_id": "...",
    "content": "...",
    "owner": "user",
    "tags": ["..."]
  }},
  {{
    "op": "SUPERSEDE",
    "memory_id": "...",
    "content": "...",
    "supersede_reason": "...",
    "owner": "user",
    "tags": ["..."]
  }}
]
```

如果无需修改：

```json
[]
```

仅输出合法 JSON。

现在输出 JSON 数组。"""


# 中文版 reconcile few-shot 示例（仅在 few_shot_enabled 时注入 {few_shot_section}）。
# 每个示例给出 existing/new 输入与期望的 ops，并附一句要点点评。ID 为简写。
RECONCILE_FEW_SHOT_ZH = """
###################
示例
###################

## 示例 1 —— 同一条新记忆：一部分与已有链冲突，另一部分无关

已有记忆：
[
  {"memory_id": "m1", "content": "用户在 Stripe 担任产品经理。", "owner": "user", "history_versions": [{"content": "用户在一家金融科技创业公司做产品经理。"}]}
]
新记忆：
[
  {"content": "用户已经从 Stripe 离职，现在在 Notion 担任产品负责人，另外最近周末开始学陶艺。", "owner": "user"}
]

输出：
```json
[
  {"op": "SUPERSEDE", "memory_id": "m1", "content": "用户在 Notion 担任产品负责人。", "supersede_reason": "雇主从 Stripe 变为 Notion", "owner": "user", "tags": ["work"]},
  {"op": "ADD", "content": "用户最近周末开始学陶艺。", "owner": "user", "tags": ["hobby"]}
]
```
**一条**新记忆里包含两件事：换工作（与已有雇佣链 m1 同维度，m1 已带历史 → SUPERSEDE，成为新链头）和一个无关细节（陶艺 → ADD）。要把这一条记忆按不同部分拆成不同操作——新链头只描述就业状态，绝不带上陶艺。

## 示例 2 —— 把两条未成链的旧记忆 + 新变化折叠成一条 3 节点链

已有记忆：
[
  {"memory_id": "a1", "content": "用户在学吉他。", "owner": "user", "memory_at": "2024-02-10 09:00"},
  {"memory_id": "a2", "content": "用户现在能完整弹几首吉他曲了。", "owner": "user", "memory_at": "2024-09-05 09:00"}
]
新记忆：
[
  {"content": "用户现在在本地的开放麦之夜表演吉他。", "owner": "user"}
]

输出：
```json
[
  {"op": "SUPERSEDE", "memory_ids": ["a1", "a2"], "content": "用户在本地的开放麦之夜表演吉他。", "supersede_reason": "吉他水平演进：学习 → 能弹曲子 → 登台表演", "owner": "user", "tags": ["music", "hobby"]}
]
```
a1 和 a2 原本没有链关系，但它们描述的是同一维度（吉他水平的进展），新记忆是最新一步。用 `memory_ids` 按旧→新折叠，它们成为一条 3 节点链（a1 ← a2 ← 新链头）。链头 `content` 只描述当前状态——不要再复述"在学"或"能弹几首"。

## 示例 3 —— 新链头绝不能复制旧记忆的内容

已有记忆：
[
  {"memory_id": "c1", "content": "用户租住在东京涩谷的一居室，月租 18 万日元，很喜欢走 5 分钟就到地铁站，但抱怨晚上很吵、还没有停车位。", "owner": "user"}
]
新记忆：
[
  {"content": "用户刚买下并搬进了横滨郊区的一套三居室房子，带花园和双车位车库。", "owner": "user"}
]

输出：
```json
[
  {"op": "SUPERSEDE", "memory_id": "c1", "content": "用户买下并搬进了横滨郊区的一套三居室房子，带花园和双车位车库。", "supersede_reason": "居住地从涩谷租住的一居室变为横滨自有的三居室", "owner": "user", "tags": ["housing"]}
]
```
新记忆 SUPERSEDE c1 并成为链头。链头 `content` 只描述横滨的新房子。**不要**把旧住处那些鲜明的细节（涩谷、18 万月租、走 5 分钟到地铁、吵闹、没停车位）拖进新节点——它们明显属于旧住处而非新家。这些保留在链历史里、仍可检索；抄进来会让链头自相矛盾（同时既在涩谷租房又在横滨买房）。
"""


# ================================================================
# SUMMARY PROMPT (中文版)
# ================================================================

SUMMARY_PROMPT_ZH = """为以下对话内容生成简洁的摘要。

内容:
---
{content}
---

记忆日期: {memory_date}
当前日期: {current_date}

要求:
1. **第三人称**: 以"用户..."描述用户——不要使用没有明确先行词的代词。
2. **长度**: 1-3句话，最多200字。
3. **优先级（内容超出长度限制时）**:
   a) 变化、决定、承诺（最高）
   b) 明确的偏好、态度、喜恶
   c) 关键事件和事实
   d) 背景信息（最低）
4. **保留偏好信号**: 保留任何直接或间接表达的喜好、厌恶、态度或观点——即使看起来不重要。
5. **自包含**: 读者应该在不看原始对话的情况下就能理解摘要。
6. **不要编造**: 不要添加原始内容中没有的信息。
7. **语言**: 输出语言必须与输入语言一致（中文输入→中文输出）。
8. **时间处理**:
   - **记忆日期**是对话实际发生的时间，是你解析对话中相对时间引用的唯一时间锚点。
   - **当前日期**是今天的系统日期（可能距记忆日期已过数年），不要用它来解释用户的陈述。
   - 如果提供了记忆日期，将相对时间表达式根据它转换为绝对引用（"上周" → 对应的日期）。
   - 如果记忆日期为空，重写句子使其不包含时间（避免在输出中留下"上周"/"昨天"等原始表达）。

## 输出约定

严格格式要求:
1. 只输出摘要文本——一段1-3句话的纯文本。
2. 不要用引号、反引号、代码块或任何 markdown 包裹输出。
3. 不要添加前缀或标签（不要"摘要："、"Summary："、"这是..."等）。
4. 不要添加尾部解释或元评论。
5. 如果内容太简单不值得总结，输出一句描述最显著元素的话——不要输出空字符串。

现在生成摘要。"""


ROLLING_SUMMARY_PROMPT_ZH = """你负责维护用户当天的对话摘要，按 topic（话题）组织。

给你的输入：
1. 上一条摘要（PREVIOUS SUMMARY）—— 该用户今天最近的一条摘要（可能为空）。
2. 新对话轮次（NEW TURNS）—— 最近的对话，每条以行号 [n] 开头。

你的任务：把 NEW TURNS 按 topic 分组，对每组判断：
- 若有价值且内容已足够 → 为它写一条摘要。如果该组延续了「上一条摘要」的同一 topic，
  则改写一条「合并了上一条摘要 + 新对话」的更新摘要（"is_continuation": true）。
- 若有价值但还不够写摘要 → 把这些行号放进 "keep_line_ids"（留待下次再总结）。
- 若无价值 / 不值得记 → 直接省略这些行号（丢弃）。

上一条摘要（PREVIOUS SUMMARY）：
---
{prev_summary}
---

新对话轮次（NEW TURNS）：
---
{turns}
---

记忆日期: {memory_date}
当前日期: {current_date}

摘要撰写规则（与标准摘要一致）：
1. 第三人称「用户……」。2. 每条 1-3 句话，最多 200 字。3. 保留偏好/决定/变化。
4. 自包含。5. 不编造。6. 输出语言必须与对话语言一致。7. 相对时间按记忆日期解析。

## 输出约定 —— 严格 JSON，不要 markdown / 代码块：
{{
  "summary_list": [
    {{"content": "<摘要文本>", "topic": "<简短话题标签>", "line_ids": [<int>, ...], "is_continuation": <true|false>}}
  ],
  "keep_line_ids": [<int>, ...]
}}

输出规则：
- 每个行号最多出现在一个位置（某条摘要的 line_ids、或 keep_line_ids、或被丢弃而完全不出现），
  绝不在两个分组里重复出现。
- line_ids / keep_line_ids 只能引用 NEW TURNS 中出现过的行号。
- 如果当前还没有任何值得写摘要的内容，返回 {{"summary_list": [], "keep_line_ids": [...]}}。
只输出 JSON。"""
