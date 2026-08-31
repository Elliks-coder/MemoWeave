# Hy-Memory Ultra 本地运行

## 已完成的环境

- Python 3.11.14：`D:\hy-memory\.venv`
- Hy-Memory 1.2.21
- Chroma 1.5.9：本地向量库
- Kuzu 0.11.3：本地图数据库
- SQLite：本地缓存、历史与观测数据
- pytest / pytest-asyncio：测试工具

激活环境：

```powershell
cd D:\hy-memory
.\.venv\Scripts\Activate.ps1
```

不激活也可以直接使用：

```powershell
.\.venv\Scripts\python.exe --version
```

## 1. 只检查本地 Ultra 后端

该命令不会调用 LLM 或 Embedding API，也不需要密钥：

```powershell
.\.venv\Scripts\python.exe .\scripts\ultra_smoke_test.py --init-only
```

预期输出包含：

```text
{"initialized": true, "mode": "ultra"}
```

## 2. 配置模型

`.env` 已创建且只包含占位符；直接编辑它并填写：

- `MEMORY_LLM_API_KEY`
- `MEMORY_LLM_BASE_URL`
- `MEMORY_EMBEDDER_API_KEY`
- `MEMORY_EMBEDDER_BASE_URL`

测试脚本检测到 DeepSeek V4 时会显式关闭 thinking 模式。记忆抽取主要是结构化 JSON
任务，关闭思考可以减少延迟与输出 Token，并避免把未使用的推理过程计入成本。

`.env` 已被 `.gitignore` 排除，任何日志和测试报告都不应该保存密钥。
如需恢复模板，可从 `.env.example` 重新复制。

## 3. 运行端到端测试

```powershell
.\.venv\Scripts\python.exe .\scripts\ultra_smoke_test.py
```

脚本将依次验证：

1. 三段带时间戳的历史会话写入；
2. 饮食偏好从“重辣川菜”更新为“清淡粤菜”；
3. 每周固定学习计划的跨会话召回；
4. Ultra `digest()` 对 System 2 的显式触发；
5. Chroma、Kuzu 与 SQLite 的持久化。

完整结果写入 `results/ultra-smoke-时间戳.json`。

## 4. GraphRAG Shadow 验证

GraphRAG 关系是在新对话写入时从 L1 候选中产生的，因此请使用一个新的
`MEMORY_DATA_DIR / MEMORY_PERSIST_DIR / MEMORY_GRAPH_DB_PATH`，重新写入评测对话。
不要直接拿旧的 500 题 runtime 测试，因为其中已有 L2 没有 Entity-Fact 关系。

第一轮建议：

```env
MEMORY_GRAPHRAG_ENABLED=true
MEMORY_GRAPHRAG_SHADOW_MODE=true
MEMORY_GRAPHRAG_MAX_HOPS=2
MEMORY_GRAPHRAG_MAX_FACTS=5
MEMORY_GRAPHRAG_MAX_DEGREE=25
MEMORY_GRAPHRAG_MIN_CONFIDENCE=0.8
```

Shadow 模式会在 `response.extra.graphrag` 和 `READ_GRAPH_RAG` trace 中记录图路径，
但不会把图证据加入回答。人工抽检关系准确率后，将
`MEMORY_GRAPHRAG_SHADOW_MODE=false` 才会让补充 L2 事实参与重排与回答。

## 已知提示

当前代码会在客户端关闭时取消并等待 MetricsCollector 后台协程。如果进程退出时再次出现：

```text
Task was destroyed but it is pending!
```

说明关闭清理出现了回归，应保留完整日志并运行回归测试，不应再把它视为正常提示。
