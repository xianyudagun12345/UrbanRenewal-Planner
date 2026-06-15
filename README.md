# UrbanRenewal Planner Autonomous Agent

> 面向上海市杨浦区城市更新场景的自主规划 AI Agent。它可以理解用户问题、主动澄清缺失信息、自主选择工具、检索空间与政策证据，并生成可解释的规划诊断和更新建议。

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2.0-green)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-enabled-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-enabled-ff4b4b)](https://streamlit.io/)

## 项目简介

UrbanRenewal Planner 是一个城市更新垂直领域 AI Agent。当前版本聚焦上海市杨浦区，面向社区更新、15 分钟生活圈、老年友好、步行环境和政策依据检索等场景。

它不是固定流程问答机器人，而是一个自主 Agent：模型会根据用户问题判断下一步需要什么信息，并选择合适工具完成任务。例如用户只说“那附近有没有养老设施？”时，Agent 可以通过会话记忆理解“那附近”指上一轮讨论的地点。

典型问题：

- 请分析鞍山新村周边 800 米的老年友好问题
- 控江路和本溪路路口有哪些步行环境问题？
- 杨浦区平凉路社区 15 分钟生活圈设施是否完善？
- 从街景看，四平路这段人行环境有什么问题？
- 刚才那个地方附近有没有适合老人休息的空间？

当前数据范围：**上海市杨浦区**。

## 核心能力

- **自主任务规划**：基于 LangGraph ReAct/tool-calling loop，自主决定是否需要地理编码、POI、路网、街景或政策 RAG。
- **多轮对话记忆**：通过 `thread_id` 和 LangGraph checkpointer 维持会话上下文，支持地点继承和连续追问。
- **澄清机制**：地点缺失、地点模糊、问题过宽或超出数据范围时，优先向用户追问或说明限制。
- **统一工具注册表**：`src/urbanrenewal/tools/registry.py` 将专业工具包装为 LLM 可调用工具，并统一返回 `ok/data/summary/error/source/cost_hint`。
- **地理编码**：调用高德地图 API，将地名解析为坐标，并完成 GCJ-02 到 WGS84 转换。
- **POI 与设施缺口诊断**：基于杨浦区 POI 数据进行 buffer 查询，并按老年友好、生活圈、步行环境等场景诊断设施短板。
- **路网分析**：基于 NetworkX/OSMnx 路网数据计算步行等时圈、主要路口和可达性。
- **街景多模态分析**：按需调用 Qwen-VL 分析街景图像，使用本地缓存避免重复分析。
- **政策 RAG**：基于 ChromaDB 和 DashScope embedding 检索政策 PDF 片段，为建议提供依据。
- **产品入口**：支持 CLI、Streamlit UI 和 FastAPI API。

## 当前架构

```text
用户问题
  │
  ├── CLI: main.py
  ├── Web UI: app.py
  └── API: src/urbanrenewal/api/main.py
      │
      ▼
Autonomous Agent
src/urbanrenewal/agent/autonomous.py
      │
      ▼
ReAct / Tool Calling Loop
      │
      ├── geocode_tool
      ├── poi_query_tool
      ├── facility_gap_tool
      ├── isochrone_tool
      ├── intersection_tool
      ├── streetview_tool
      ├── policy_rag_tool
      └── report_generation_tool
      │
      ▼
结构化 Markdown 诊断与规划建议
```

## 技术栈

| 层次 | 技术 |
|---|---|
| Agent 编排 | LangGraph ReAct agent |
| LLM / VL | 阿里云 DashScope OpenAI-compatible Qwen / Qwen-VL |
| 工具封装 | LangChain tools + Pydantic schema |
| 向量检索 | ChromaDB + text-embedding-v4 |
| 空间分析 | GeoPandas + Shapely + NetworkX + OSMnx |
| 地理编码 | 高德地图 REST API |
| Web UI | Streamlit |
| API | FastAPI |
| 配置 | YAML + `.env` |

## 项目结构

```text
UrbanRenewal-Planner/
├── app.py                              # Streamlit 自主 Agent 对话界面
├── main.py                             # CLI 主入口，默认调用自主 Agent
├── config/
│   └── project.yaml                    # 本地数据路径、RAG 参数、场景配置
├── src/
│   └── urbanrenewal/
│       ├── agent/
│       │   ├── autonomous.py           # 自主 Agent 主实现
│       │   ├── llm.py                  # 共享 LLM 工厂
│       │   └── planner.py              # 旧固定 workflow，后续可移除
│       ├── api/
│       │   ├── main.py                 # FastAPI 服务
│       │   └── schemas.py              # API 请求/响应模型
│       ├── config/
│       │   └── settings.py             # 配置读取
│       ├── rag/
│       │   └── build_policy_rag.py     # 政策 PDF 向量库构建
│       └── tools/
│           ├── registry.py             # LLM 可调用工具注册表
│           ├── geocode.py
│           ├── poi_query.py
│           ├── road_query.py
│           ├── streetview_query.py
│           └── policy_rag.py
├── scripts/
│   ├── build_rag.py
│   ├── run_agent_smoke.py              # 自主 Agent CLI smoke test
│   ├── test_rag.py
│   └── ocr_scanned_pdfs.py
└── tests/
    └── unit/
```

## 数据依赖

数据文件不提交到代码仓库。请在 `config/project.yaml` 中配置本地路径。

| 数据 | 用途 |
|---|---|
| `poi_yangpu_clean.parquet` | POI 查询和设施缺口诊断 |
| `walk_bike_network.graphml` | 步行/骑行路网分析 |
| `image_metadata.parquet` | 街景图片索引 |
| `image_analysis_cache.parquet` | 街景多模态分析缓存 |
| 政策 PDF | RAG 原始文档 |
| ChromaDB policy collection | 政策语义检索 |

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

或：

```bash
pip install -e .
```

### 2. 配置环境变量

复制示例文件：

```bash
cp .env.example .env
```

填写：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL=qwen-plus
VISION_MODEL=qwen-vl-plus
AMAP_API_KEY=your_amap_api_key
```

### 3. 配置数据路径

复制并修改：

```bash
cp project.example.yaml config/project.yaml
```

根据本机数据位置修改 `paths`。

### 4. 构建政策 RAG

```bash
uv run python scripts/build_rag.py
```

强制重建：

```bash
uv run python scripts/build_rag.py --rebuild
```

### 5. 运行自主 Agent

CLI 交互：

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

Streamlit UI：

```bash
uv run streamlit run app.py
```

FastAPI：

```bash
uv run uvicorn src.urbanrenewal.api.main:app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/v1/health
```

## Python 调用示例

```python
from src.urbanrenewal.agent.autonomous import run_autonomous

result = run_autonomous(
    "请分析鞍山新村周边800米的老年友好问题",
    thread_id="demo-session",
)

print(result.answer)
print(result.tool_events)
```

## API 示例

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"请分析鞍山新村周边800米的老年友好问题\",\"session_id\":\"demo\"}"
```

## 工具返回契约

所有自主 Agent 工具统一返回：

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

这样做的目的：

- 让 LLM 更稳定地理解工具结果。
- 让 API 和前端可以复用同一份数据。
- 让失败结果可解释，而不是直接抛异常。
- 为后续成本控制和审计打基础。

## 成本控制与安全边界

当前自主 Agent 已内置基础限制：

- 最大空间查询半径：`2000m`
- 单次街景分析上限：`3` 张
- 单次政策 RAG 查询上限：`5` 条 query
- 单条 RAG query 返回上限：`5` 条
- Agent 最大递归轮次：默认 `18`

后续计划：

- Redis checkpointer，支持跨进程会话记忆。
- 地理编码缓存。
- RAG embedding 缓存。
- 街景分析预算按用户等级控制。
- 鉴权、限流和调用审计。

## 测试

```bash
uv run pytest tests/unit -q
uv run ruff check src tests app.py main.py scripts/run_agent_smoke.py scripts/eval_task_planner.py
```

当前基础测试覆盖：

- 工具注册表契约。
- API schema。
- API health endpoint。

## 已知限制

- 当前数据范围仅覆盖上海市杨浦区。
- 街景图片具有时间滞后，部分现状可能已变化。
- 当前结果主要是 Markdown + 工具轨迹，地图图层和结构化 `PlanningReport` 仍在后续开发计划中。
- 当前会话记忆使用内存 checkpointer，进程重启后会丢失。
- 旧 `planner.py` 固定 workflow 仍保留在代码中，但不再是产品主入口。

## 下一步开发重点

- 定义严格的 `PlanningReport` 输出结构。
- 从工具调用结果中提取地图图层、POI、政策证据和建议卡片。
- Streamlit 重新接入地图展示。
- API 流式接口返回更细粒度的工具事件。
- Redis 记忆、缓存和成本预算系统。
