# Hy-Memory Ultra × LoCoMo 小型评测

本目录使用 LoCoMo（ACL 2024）官方数据，在一个完整的 19-session 长期对话上抽取
25 道问题。数据来源：<https://github.com/snap-research/locomo>。

## 选样设计

- 5 道 multi-hop（类别 1）
- 5 道 temporal（类别 2）
- 5 道 open-domain（类别 3）
- 5 道 single-hop（类别 4）
- 5 道 adversarial / unanswerable（类别 5）

固定题号写在 `scripts/locomo_ultra_eval.py` 中，覆盖前期和后期会话、多证据问题、
相对时间推理、开放域推断，以及把 Caroline/Melanie 身份互换的对抗问题。
非对抗题统一询问映射到 Hy-Memory `user` 角色的 Caroline。原因是 LoCoMo 原始对话
由两位真人互聊，而 Hy-Memory 的人物记忆模型明确区分 `user` 与 `assistant`；把
Melanie 的真人经历按 assistant 消息写入后再当作用户事实提问，会混入角色适配误差。
官方类别 5 的题号 184 未采用：它把 Caroline 的乐器标成不可回答，但完整对话中
Caroline 明确说自己会弹木吉他，存在真值冲突；本子集改用题号 185。

## 运行

```powershell
.\.venv\Scripts\python.exe .\scripts\locomo_ultra_eval.py
```

中断后可通过报告文件名中的 run id 续跑：

```powershell
.\.venv\Scripts\python.exe .\scripts\locomo_ultra_eval.py --resume-run 20260813-120000
```

报告写入 `results/locomo-ultra-<run-id>.json`。核心指标：

- 端到端 QA accuracy：检索结果经过同一 LLM 生成答案，再由同一 LLM 按参考答案判分。
- evidence Hit@K：返回记忆的发生日期是否命中人工证据所在 session 的日期。
- 检索延迟、写入延迟、写入成功率和各记忆层数量。

## 结果解释边界

这是成本受控的 25 题分层子集，不等同于完整 LoCoMo 1,540 题官方成绩。
Hy-Memory 搜索结果不暴露原 session id，因此 evidence Hit@K 用记忆日期与证据
session 日期匹配，是近似检索指标。端到端 accuracy 使用模型裁判，也应抽查失败案例，
不能把它当作无误差的人工评分。

LoCoMo 原始数据遵循上游的 CC BY-NC 4.0 许可，仅用于非商业学习与评测。
