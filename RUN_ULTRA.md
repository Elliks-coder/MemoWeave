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

## 已知提示

Hy-Memory 1.2.21 的单客户端关闭流程没有取消 MetricsCollector 的两个后台协程，进程退出时可能打印：

```text
Task was destroyed but it is pending!
```

当前验证中这不会导致退出码失败或数据丢失，属于发布包的关闭清理问题，不代表 Ultra 初始化失败。
