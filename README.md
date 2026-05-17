# UrbanRenewal Planner Agent

> 面向上海市杨浦区城市更新的多模态规划诊断与建议生成 Agent

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2.0-green)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 项目简介

本项目是一个实验性 AI Agent，专为上海市杨浦区城市更新与社区规划场景设计。

Agent 的核心能力不是回答通用城市规划知识，而是**针对用户提出的具体空间问题，自动完成地点解析、空间数据检索、政策文献召回、问题诊断和规划建议生成**。

典型问题示例：

- *请分析鞍山新村周边 800 米的老年友好问题*
- *控江路和本溪路路口有哪些步行环境问题？*
- *杨浦区某小区周边 15 分钟生活圈设施是否完善？*
- *从街景来看，四平路这个街段有什么人行环境问题？*

**当前版本仅支持上海市杨浦区范围内的问题。**

---

## 功能特性

- **自然语言意图解析**：LLM 自动识别地名、分析场景（老年友好 / 15 分钟生活圈 / 步行环境 / 通用）、分析半径
- **地理编码**：调用高德地图 API，支持小区名、路口、医院等各类地名，自动将 GCJ-02 坐标转换为 WGS84
- **POI 空间查询**：35,120 条杨浦区 POI，圆形 buffer 查询 + 场景设施缺口诊断
- **路网分析**：OSMnx 步行路网，支持等时圈（步行 5/10/15 分钟）、最短路径、路口密度分析
- **街景多模态分析**：按需调用 Qwen-VL 分析 2,746 张杨浦区街景图片，结果本地缓存避免重复 API 调用
- **政策 RAG 检索**：16 份城市更新政策 PDF 构建的 ChromaDB 向量库，语义检索相关文件段落
- **结构化建议生成**：综合以上数据，输出带优先级、具体空间位置和政策依据的规划建议

---

## 技术架构

```
用户问题
    │
    ▼
parse_intent ──► geocode ──► query_spatial ──────────────────┐
                              (POI + 路网)                    │
                                  │                           │
                          need_streetview?                    │
                          ├── Yes ──► query_streetview ───►   │
                          └── No ──────────────────────────►  │
                                                              ▼
                                                       query_policy
                                                       (政策 RAG)
                                                              │
                                                              ▼
                                                          generate
                                                       (LLM 生成建议)
```

| 层次 | 技术选型 |
|---|---|
| Agent 框架 | LangGraph 1.2（有状态工作流） |
| LLM / VL 模型 | 阿里云 Qwen（DashScope OpenAI 兼容接口） |
| Embedding | DashScope text-embedding-v4 |
| 向量数据库 | ChromaDB（本地持久化，cosine 距离） |
| 空间分析 | GeoPandas + Shapely + OSMnx + NetworkX |
| 地理编码 | 高德地图 REST API v3 |
| 配置管理 | YAML + python-dotenv |

---

## 项目结构

```
UrbanRenewal-Planner/
├── src/
│   └── urbanrenewal/
│       ├── config/
│       │   └── settings.py          # 统一配置单例（读取 YAML + .env）
│       ├── agent/
│       │   └── planner.py           # LangGraph Agent 主工作流
│       ├── rag/
│       │   └── build_policy_rag.py  # 政策 PDF → ChromaDB 构建
│       └── tools/
│           ├── geocode.py           # 高德地理编码 + GCJ02→WGS84
│           ├── poi_query.py         # POI 空间查询与设施缺口诊断
│           ├── road_query.py        # 路网分析（等时圈/最短路径/路口）
│           ├── streetview_query.py  # 街景检索 + 多模态按需分析
│           └── policy_rag.py        # 政策 RAG 查询封装
├── scripts/
│   ├── build_rag.py                 # RAG 构建 CLI（支持 --rebuild）
│   ├── test_rag.py                  # RAG 检索质量测试
│   ├── test_agent.py                # Agent 端到端测试
│   └── ocr_scanned_pdfs.py         # 扫描件 PDF OCR 补充
├── notebooks/
│   ├── POI.ipynb                    # POI 数据探索
│   ├── street_road.ipynb            # 路网数据探索
│   └── street_view.ipynb            # 街景数据探索
├── config/
│   └── project.yaml                 # 路径、RAG 参数、场景定义
├── main.py                          # 交互式入口
├── pyproject.toml
├── .env.example
├── app.py
```

---

## 数据说明

本项目依赖以下**外部数据**（不在代码仓库中，存放于 `~/urban_renewal_data/`）：

| 数据 | 规模 | 路径 |
|---|---|---|
| 杨浦区 POI | `processed/POI data/poi_yangpu_clean.parquet` |
| 杨浦区路网（OSMnx） | `processed/road/walk_bike_network.graphml` |
| 街景图片 metadata | `processed/yangpu_processed/image_metadata.parquet` |
| 街景图片原文件 | `~D:/街景~/yangpu/` |
| 政策 PDF | `~/document/` |
| ChromaDB 向量库 | `vector_db/policy_chroma/` |

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd UrbanRenewal-Planner

# 安装依赖（使用 uv，推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入以下 API Key：

```env
# 阿里云 DashScope（用于 LLM / Embedding / 街景分析）
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 对话生成模型
MODEL=qwen3.5-omni-plus

# 街景图像分析模型（Qwen-VL 系列）
VISION_MODEL=qwen3.5-omni-plus

# 高德地图 API（用于地理编码）
AMAP_API_KEY=your_amap_api_key
```

### 3. 构建政策 RAG 向量库

```bash
# 首次运行（增量构建）
python scripts/build_rag.py

# 强制重建（清空后重新向量化）
python scripts/build_rag.py --rebuild
```

### 4. 验证 RAG 检索质量

```bash
python scripts/test_rag.py
# 可选：自定义查询
python scripts/test_rag.py --query "无障碍坡道设计标准"
```

### 5. 运行 Agent

```bash
# 交互模式
python main.py

# 单次查询
python main.py "请分析鞍山新村周边800米的老年友好问题"

# 端到端测试（含中间状态摘要）
python scripts/test_agent.py
python scripts/test_agent.py -q "控江路和本溪路路口有哪些步行环境问题？"
```

---

## 使用示例

```python
from src.urbanrenewal.agent.planner import run

answer = run("请分析鞍山新村周边800米的老年友好问题")
print(answer)
```

输出示例（节选）：

```
### 优先级 高 | 增设嵌入式综合为老服务站
- 问题：800m 范围内社区服务设施完全缺失，养老服务仅 1 处（距 638m）
- 措施：在鞍山新村委附近置换底层商铺，嵌入助餐、日托、康复等功能
- 依据：《完整居住社区建设指南》要求综合为老服务站服务半径≤500m

### 优先级 中 | 改善主要步行路径无障碍连续性
- 问题：118 个路口中大量缺少缘石坡道和盲道铺装
- 措施：对居委至公交站、医院的必经路径优先实施无障碍改造
- 依据：《上海市慢行交通规划设计导则》4.3.1 节无障碍系统要求
```

---

## 各工具模块说明

### `tools/geocode.py`
将自然语言地名解析为 WGS84 坐标。核心功能：优先返回杨浦区内结果；自动用"上海+地名"重试；GCJ-02→WGS84 坐标转换（标准偏移公式，无额外依赖）。

### `tools/poi_query.py`
圆形 buffer POI 查询。临时投影到 UTM Zone 51N（EPSG:32651）保证米制精度，`@lru_cache` 进程内只加载一次数据文件（加载 0.11s，查询 0.04s）。

### `tools/road_query.py`
路网分析工具。使用 `networkx.read_graphml` 加载（osmnx 存在版本兼容问题），numpy 向量化最近节点计算（<1ms），`nx.ego_graph` 实现等时圈。

### `tools/streetview_query.py`
街景检索与多模态分析。**按需调用**，不预先全量分析。分析结果写入本地 parquet 缓存，同一图片+场景组合只调用一次 API。

### `tools/policy_rag.py`
政策文献语义检索。支持场景关键词自动扩充（提升召回率）、多查询合并去重、`min_score` 阈值过滤，输出"依据《XXX》"格式引用段落。

---

## 配置文件

`config/project.yaml` 管理所有路径和分析参数：

```yaml
analysis:
  default_radius_m: 800      # 默认分析半径
  walk_speed_kmh: 4.8        # 步行速度（用于等时圈计算）
  default_streetview_limit: 8 # 每次最多分析街景数量

scenarios:
  elderly_friendly:
    poi_categories: [医疗, 药店, 养老服务, 公交站, ...]
    policy_keywords: [老年友好, 适老化, 无障碍, ...]
```

增加新场景只需在 YAML 中添加对应配置，无需修改代码。

---

## 开发

```bash
# 代码风格检查
ruff check src/

# 自动修复可修复的问题
ruff check src/ --fix

# 格式化
black src/
```

---

## 已知限制

- **地理范围**：第一阶段仅支持上海市杨浦区，不处理区外问题
- **街景时效**：街景图片采集于 2019 年，部分街道现状可能已发生变化
- **路网名称**：部分路段 `road_name=unknown`，来自 OSM 数据不完整

---

## License

MIT License — 详见 [LICENSE](LICENSE)
