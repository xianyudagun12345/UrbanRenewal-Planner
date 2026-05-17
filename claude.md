# UrbanRenewal-Planner Agent 项目说明

## 1. 项目概述

本项目是一个面向上海市杨浦区城市更新与社区规划场景的实验性 AI Agent。

Agent 的目标不是简单回答城市规划知识，而是根据用户提出的具体空间问题，自动完成地点解析、空间数据检索、政策规划依据检索、问题诊断和规划建议生成。

当前项目默认研究范围为上海市杨浦区。第一阶段为功能实验，不处理杨浦区以外的问题。若用户提出杨浦区以外的问题，系统应提示当前版本暂不支持。

## 2. 项目定位

本项目可理解为：

“面向杨浦区城市更新的多模态规划诊断与建议生成 Agent”。

它基于以下数据：

1. 杨浦区 POI 数据；
2. 杨浦区路网数据；
3. 杨浦区街景图像及其 metadata；
4. 城市更新、15 分钟生活圈、适老化、无障碍、街道设计、慢行交通等政策与规划参考文件；
5. 地图 API 动态获取的地点坐标与行政区信息。

Agent 应根据用户问题，围绕特定小区、街道、路口、医院、学校、公园、地铁站或其他空间对象，生成具有空间依据和政策依据的规划建议。

## 3. 当前 MVP 范围

当前版本优先实现以下功能：

1. 用户输入自然语言问题；
2. 解析用户问题中的地点、分析目标和空间范围；
3. 调用地图 API 获取地点坐标、地址和行政区信息；
4. 根据地点坐标构建分析范围，例如 500 米、800 米 buffer；
5. 从本地 POI 数据中检索分析范围内的设施；
6. 从本地路网数据中检索相关道路、路口和步行路径信息；
7. 从本地街景 metadata 和街景分析结果中检索相关街景图像；
8. 从政策与规划依据数据库中检索相关文件片段；
9. 综合空间数据和政策依据，诊断城市更新问题；
10. 输出结构化规划建议。

当前版本不实现正式工程预算、政府投资项目管理、财政预算申报等复杂功能。

## 4. 典型用户问题

Agent 应支持类似问题：

- 请分析鞍山新村周边 800 米的老年友好问题。
- 新华医院周边适合做哪些适老化微更新？
- 控江路和本溪路路口有哪些步行环境问题？
- 杨浦区某个小区周边 15 分钟生活圈设施是否完善？
- 某个社区周边是否缺少公园、药店、公厕或养老服务设施？
- 某个路口为什么适合做城市更新？
- 从街景来看，这个街段在人行环境上有什么问题？
- 请结合政策文件说明为什么建议这样更新。

## 5. Agent 工作流程

推荐工作流程如下：

1. 用户问题解析
   - 识别地点名称；
   - 识别规划目标，例如老年友好、步行友好、15 分钟生活圈、城市更新；
   - 识别空间范围，例如 500 米、800 米、15 分钟步行范围；
   - 识别输出需求，例如问题诊断、更新建议、政策依据、地图图层。

2. 地点解析
   - 调用地图 API；
   - 获取地点经纬度；
   - 获取地址和行政区；
   - 判断是否属于上海市杨浦区；
   - 如果地图 API 返回多个候选，应优先选择位于上海市杨浦区且名称匹配度最高的结果。

3. 分析范围构建
   - MVP 阶段使用圆形 buffer；
   - 默认半径可以为 800 米；
   - 后续可扩展为基于路网的 5 / 10 / 15 分钟步行圈。

4. 空间数据检索
   - 查询范围内 POI；
   - 查询相关路网和道路节点；
   - 查询范围内街景图像 metadata；
   - 查询街景图像分析标签。

5. 政策与规划依据检索
   - 根据用户问题和诊断结果，从政策文献向量库中检索相关内容；
   - 重点检索城市更新、15 分钟生活圈、老年友好、无障碍、慢行交通、街道设计、完整社区等相关内容；
   - 输出时应尽量说明建议对应的政策依据。

6. 问题诊断
   - 结合 POI、路网、街景和政策依据；
   - 识别设施配置问题、步行环境问题、无障碍问题、街道空间问题、公共服务短板；
   - 避免只输出泛泛而谈的建议。

7. 规划建议生成
   - 输出具体问题；
   - 输出具体空间位置或节点；
   - 输出对应的更新措施；
   - 输出优先级；
   - 输出政策依据；
   - 必要时输出 GeoJSON、CSV、Markdown 或地图结果。

## 6. 数据使用原则

项目数据分为两类：

### 6.1 空间数据

包括：

- POI 数据；
- 路网数据；
- 街景图像 metadata；
- 街景图像分析标签。

空间数据不应主要依赖向量数据库检索，而应使用空间查询方法，例如坐标、buffer、空间索引、距离计算、路网分析等。

### 6.2 文本知识数据

包括：

- 政策文件；
- 规划导则；
- 技术标准；
- 城市更新案例；
- 街景图像文字描述；
- 措施库说明。

这些数据可以进入向量数据库，用于 RAG 检索。

## 当前项目进度

### 数据底座（已完成）

1. 街景数据：
   - 已从百度地图 API 获取 2019 年上海街景图片（实际图片位于 `D:\街景\yangpu_2019`）；
   - 已筛选杨浦区街景并完成 metadata 标准化；
   - 标准化结果位于：`D:/AI/urban_renewal_data/processed/yangpu_processed_2019`
   - metadata parquet：3670 条，其中 image_exists=True 共 2746 条，缺失 924 条（忽略）；
   - 当前策略：不提前全量调用多模态模型，用户提问后按需检索图片并分析。

2. POI 数据：
   - 已获取 2024 年上海市 POI，已筛选和标准化杨浦区 POI；
   - 标准化结果位于：`D:/AI/urban_renewal_data/processed/POI data/poi_yangpu_clean.parquet`
   - POI 主检索方式为空间检索，不进入向量库。

3. 路网数据：
   - 已获取杨浦区 OSMnx 路网，已保存 nodes/edges parquet 与步行+骑行 GraphML；
   - 标准化结果位于：`D:/AI/urban_renewal_data/processed/road/`

4. 政策文件 RAG：
   - 已收集 16 个 PDF（位于 `D:/AI/document`）；
   - 已构建 ChromaDB 向量库（371 个 chunk，13 份成功提取文字）；
   - 向量库位于：`D:/AI/urban_renewal_data/vector_db/policy_chroma/`
   - 分块文件：`D:/AI/urban_renewal_data/processed/policy/policy_chunks.jsonl`
   - 文档元信息：`D:/AI/urban_renewal_data/processed/policy/policy_documents.jsonl`
   - 注：3 份扫描件 PDF（共 81 页）暂未 OCR，chunk_count=0，待后续补充。

### 项目目录结构（当前）

```
URBANRENEWAL-PLANNER/
├── src/
│   └── urbanrenewal/          ← 主包
│       ├── agent/             ← Agent 工作流（预留）
│       ├── config/
│       │   └── settings.py    ← 统一配置单例（读取 YAML + .env）
│       ├── rag/
│       │   └── build_policy_rag.py  ← PDF解析、分块、向量化、ChromaDB写入
│       ├── tools/
│       │   └── geocode.py     ← 高德地理编码 + GCJ02→WGS84 坐标转换
│       └── utils/             ← 通用工具（预留）
├── notebooks/                 ← 数据探索 Jupyter Notebook
│   ├── POI.ipynb
│   ├── street_road.ipynb
│   └── street_view.ipynb
├── scripts/                   ← 运维/一次性脚本
│   ├── build_rag.py           ← RAG 构建 CLI（支持 --rebuild）
│   ├── test_rag.py            ← RAG 检索质量测试
│   └── ocr_scanned_pdfs.py   ← 扫描件 OCR 补充（待运行）
├── tests/                     ← 单元测试（预留）
├── outputs/                   ← 生成结果文件
├── config/
│   └── project.yaml           ← 项目配置（路径、RAG参数、场景定义）
├── main.py
├── pyproject.toml
├── .env / .env.example
└── claude.md
```

所有模块通过 `from src.urbanrenewal.config import cfg` 获取配置。

## 当前优先任务

按顺序完成以下任务：

1. ~~编写政策 PDF RAG 构建脚本~~ ✅ 已完成；
2. ~~编写政策 RAG 检索测试脚本~~ ✅ 已完成；
3. ~~编写地理编码工具（高德API + GCJ02→WGS84）~~ ✅ 已完成；
4. ~~调整项目目录结构为标准 AI Agent 工程结构~~ ✅ 已完成；
5. ~~调整项目目录结构为标准 AI Agent 工程结构~~ ✅ 已完成；
6. ~~编写 POI 空间查询工具 `src/urbanrenewal/tools/poi_query.py`~~ ✅ 已完成；
7. ~~编写路网查询工具 `src/urbanrenewal/tools/road_query.py`~~ ✅ 已完成；
8. ~~编写街景查询工具 `src/urbanrenewal/tools/streetview_query.py`~~ ✅ 已完成；
9. ~~编写政策 RAG 查询封装 `src/urbanrenewal/tools/policy_rag.py`~~ ✅ 已完成；
10. ~~编排 Agent 工作流（LangGraph）`src/urbanrenewal/agent/planner.py`~~ ✅ 已完成。

### MVP 阶段全部完成 ✅

## 开发约束

- 不要重新建议全量预分析所有街景图片；
- 不要重新引入正式预算管理功能；
- 不要把 POI、路网、街景全部放入向量数据库；
- 空间数据使用空间查询；
- 政策 PDF 使用 RAG；
- 代码要尽量模块化，路径从 `config/project.yaml` 读取；
- 不要把 API Key 写入代码。