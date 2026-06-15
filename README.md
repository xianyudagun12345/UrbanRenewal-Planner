# UrbanRenewal Planner Autonomous Agent

面向上海市杨浦区城市更新场景的自主 AI Agent。项目支持自然语言提问、任务预规划、信息澄清、工具调用、空间分析、政策 RAG、结构化报告、Streamlit 页面和 FastAPI 服务。

## 项目简介

UrbanRenewal Planner 聚焦上海市杨浦区的城市更新分析场景，适用于社区更新、15 分钟生活圈、老年友好、步行环境、设施短板和政策依据检索等问题。

用户可以直接提出自然语言问题，例如：

```text
请分析鞍山新村周边800米的老年友好问题
控江路和本溪路路口有哪些步行环境问题？
杨浦区平凉路社区15分钟生活圈设施是否完善？
刚才那个地方附近有没有适合老人休息的空间？
```

当前数据范围：上海市杨浦区。

## 已实现能力

- 自主 Agent：基于 LangGraph ReAct/tool-calling，让模型根据问题自主选择工具。
- 任务预规划：进入 Agent 前先识别地点、场景、半径、上下文指代和是否需要澄清。
- 澄清机制：地点缺失、地点不清、超出数据范围时优先追问或说明限制。
- 会话记忆：支持 `thread_id`，并通过 SQLite checkpointer 保存 LangGraph 会话状态。
- API 会话上下文：FastAPI 保存最近对话轮次，并把上下文注入下一轮 Agent 调用。
- 工具注册表：统一封装地理编码、POI、设施缺口、路网、街景、政策 RAG 和报告生成工具。
- 结构化报告：返回 Markdown 答案，同时生成 `PlanningReport`，用于前端展示问题、建议、政策依据、空间证据和地图图层。
- 成本边界：按用户等级定义 `AgentBudget`，限制递归轮数、工具调用、街景图片数、政策查询数和分析半径。
- 缓存：提供 TTL cache，并用于地理编码和政策 RAG 等重复查询场景。
- 服务层：提供同步、流式、异步 Agent API，以及会话、反馈、任务状态和健康检查接口。

## 技术栈

| 模块 | 技术 |
|---|---|
| Agent 编排 | LangGraph, LangChain |
| LLM / 多模态 | DashScope OpenAI-compatible Qwen / Qwen-VL |
| 工具 Schema | Pydantic |
| API | FastAPI |
| Web UI | Streamlit, Folium |
| RAG | ChromaDB, DashScope text-embedding-v4 |
| 空间分析 | GeoPandas, Shapely, NetworkX, OSMnx |
| 存储 | SQLite, 本地文件缓存 |
| 测试 | pytest, ruff |

## 项目结构

```text
UrbanReneweral-Planner/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── project.example.yaml
├── app.py
├── main.py
├── eval/
│   └── golden_questions.json
├── scripts/
│   ├── build_rag.py
│   ├── eval_task_planner.py
│   ├── ocr_scanned_pdfs.py
│   ├── run_agent_smoke.py
│   └── test_rag.py
├── src/
│   └── urbanrenewal/
│       ├── agent/
│       │   ├── autonomous.py
│       │   ├── budget.py
│       │   ├── checkpoint.py
│       │   ├── llm.py
│       │   ├── plan.py
│       │   ├── planner.py
│       │   └── report.py
│       ├── api/
│       │   ├── main.py
│       │   ├── observability.py
│       │   ├── rate_limit.py
│       │   ├── schemas.py
│       │   ├── store.py
│       │   ├── task_models.py
│       │   ├── task_store.py
│       │   └── tasks.py
│       ├── config/
│       │   └── settings.py
│       ├── rag/
│       │   └── build_policy_rag.py
│       ├── tools/
│       │   ├── geocode.py
│       │   ├── poi_query.py
│       │   ├── policy_rag.py
│       │   ├── registry.py
│       │   ├── road_query.py
│       │   └── streetview_query.py
│       └── utils/
│           └── ttl_cache.py
└── tests/
    ├── conftest.py
    └── unit/
```

说明：

- `app.py` 是 Streamlit 页面入口。
- `main.py` 是命令行入口。
- `src/urbanrenewal/api/main.py` 是 FastAPI 服务入口。
- `src/urbanrenewal/agent/planner.py` 是早期固定 workflow 实现，当前产品入口使用 `autonomous.py`。
- `notebooks/`、`docs/`、`outputs/`、本地配置和缓存文件不会上传到公开仓库。

## 配置

复制环境变量示例：

```bash
cp .env.example .env
```

填写 `.env`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL=qwen3.5-omni-plus
VISION_MODEL=qwen3.5-omni-plus
FAST_MODEL=qwen3.5-omni-flash

AMAP_API_KEY=your_amap_api_key_here
AMAP_BASE_URL=https://restapi.amap.com
```

复制项目配置示例：

```bash
cp project.example.yaml config/project.yaml
```

根据本机数据位置修改 `config/project.yaml` 中的 `paths`。真实 `.env` 和 `config/project.yaml` 已被 `.gitignore` 忽略。

## 数据依赖

数据文件不随代码仓库上传，需要本地准备并在 `config/project.yaml` 中配置路径。

| 数据 | 用途 |
|---|---|
| POI parquet | 周边设施查询和设施缺口诊断 |
| 路网 graphml/parquet | 步行等时圈、路口和慢行分析 |
| 街景元数据和图片 | 街景点位筛选和多模态分析 |
| 政策 PDF | 政策 RAG 原始文档 |
| ChromaDB policy collection | 政策语义检索 |

## 安装

推荐使用 uv：

```bash
uv sync
```

或使用 pip：

```bash
pip install -e .
```

## 运行

### CLI

交互模式：

```bash
uv run python main.py
```

单次查询：

```bash
uv run python main.py "请分析鞍山新村周边800米的老年友好问题"
```

查看工具调用轨迹：

```bash
uv run python scripts/run_agent_smoke.py -q "控江路和本溪路路口有哪些步行环境问题？"
```

### Streamlit

```bash
uv run streamlit run app.py
```

### FastAPI

```bash
uv run uvicorn src.urbanrenewal.api.main:app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/v1/health
```

同步聊天接口：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"请分析鞍山新村周边800米的老年友好问题\",\"session_id\":\"demo\"}"
```

## API 接口

| 接口 | 方法 | 用途 |
|---|---|---|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/chat` | POST | 同步 Agent 调用 |
| `/api/v1/chat/stream` | POST | NDJSON 流式进度 |
| `/api/v1/chat/async` | POST | 提交后台任务 |
| `/api/v1/chat/tasks/{task_id}` | GET | 查询异步任务状态 |
| `/api/v1/sessions/{session_id}` | GET | 查询会话状态 |
| `/api/v1/feedback` | POST | 提交用户反馈 |
| `/api/v1/service/status` | GET | 查询服务状态 |

## Python 调用

```python
from src.urbanrenewal.agent.autonomous import run_autonomous

result = run_autonomous(
    "请分析鞍山新村周边800米的老年友好问题",
    thread_id="demo-session",
)

print(result.answer)
print(result.tool_events)
print(result.report)
```

## 工具返回格式

自主 Agent 工具统一返回：

```json
{
  "ok": true,
  "data": {},
  "summary": "工具结果摘要",
  "error": "",
  "source": "数据来源",
  "cost_hint": "low"
}
```

这种结构便于 Agent 理解工具结果，也便于 API、前端、日志和测试复用。

## 构建政策 RAG

```bash
uv run python scripts/build_rag.py
```

强制重建：

```bash
uv run python scripts/build_rag.py --rebuild
```

RAG 构建会读取 `config/project.yaml` 中的政策 PDF 路径、处理输出路径和 ChromaDB 路径。

## 测试与评估

静态检查：

```bash
uv run ruff check src tests app.py main.py scripts/run_agent_smoke.py scripts/eval_task_planner.py
```

单元测试：

```bash
uv run pytest tests/unit -q
```

任务规划器评估：

```bash
uv run python scripts/eval_task_planner.py
```

当前测试覆盖包括：

- Agent checkpointer 工厂
- API schema、健康检查、会话上下文、异步任务
- 会话和任务 SQLite 存储
- 请求预算
- TTL cache
- 工具注册表返回契约
- 任务预规划
- 结构化 `PlanningReport`
- 结构化日志

## 公开仓库说明

以下内容不会上传到公开仓库：

- `.env`
- `config/project.yaml`
- `outputs/`
- `notebooks/`
- `docs/`
- `ROADMAP.md`
- `claude.md`
- `.venv/`
- `.uv-cache/`
- Python 缓存和测试缓存

公开仓库保留的是项目代码、示例配置、README、测试、评估集和运行脚本。
