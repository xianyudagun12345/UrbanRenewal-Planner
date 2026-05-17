"""
UrbanRenewal Planner Agent 主工作流（LangGraph）。

工作流节点：
  parse_intent   → 解析用户问题中的地名、场景、半径、是否需要街景
  geocode        → 高德地理编码，获取 WGS84 坐标，判断是否在杨浦区
  query_spatial  → 并行查询 POI + 路网（等时圈 + 路口）
  query_sv       → 街景检索 + 按需多模态分析（条件节点，need_streetview=True 时执行）
  query_policy   → 政策 RAG 多条查询
  generate       → LLM 综合以上数据生成结构化规划建议

State 字段说明：
  question          用户原始问题
  intent            意图解析结果 dict
  geocode_result    GeocodeResult 对象
  poi_summary       DataFrame（category_planning 统计）
  gap_report        FacilityGapReport
  isochrone         IsochroneResult（步行10分钟等时圈）
  intersections     路口列表
  sv_results        街景分析结果列表
  policy_chunks     政策检索结果列表
  answer            最终输出文本
  error             节点异常信息（非空时跳转到 error_exit）
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.urbanrenewal.config import cfg
from src.urbanrenewal.tools.geocode import GeocodeResult, geocode
from src.urbanrenewal.tools.poi_query import (
    FacilityGapReport,
    check_facility_gaps,
    query_poi_by_buffer,
    summarize_poi,
)
from src.urbanrenewal.tools.road_query import (
    IsochroneResult,
    IntersectionResult,
    query_intersections,
    query_isochrone,
)
from src.urbanrenewal.tools.streetview_query import StreetviewAnalysis, query_and_analyze
from src.urbanrenewal.tools.policy_rag import PolicyChunk, format_citations, query_policy_multi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM 实例（模块级）
# ---------------------------------------------------------------------------

def _build_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.dashscope_api_key,
        base_url=cfg.dashscope_base_url,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class PlannerState(TypedDict):
    question: str
    intent: dict                              # parse_intent 输出
    geocode_result: Optional[GeocodeResult]
    poi_summary: Optional[list[dict]]         # summarize_poi 序列化为 list[dict]
    gap_report: Optional[FacilityGapReport]
    isochrone: Optional[IsochroneResult]
    intersections: list[IntersectionResult]
    sv_results: list[StreetviewAnalysis]
    policy_chunks: list[PolicyChunk]
    answer: str
    error: str


# ---------------------------------------------------------------------------
# 意图解析
# ---------------------------------------------------------------------------

_PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个城市规划意图解析器，专门服务于上海市杨浦区城市更新 Agent。
只能处理杨浦区范围内的问题。

从用户问题中提取以下字段，以 JSON 格式返回，不要输出任何其他内容：
{{
  "place": "地名原文（如 鞍山新村、控江路本溪路路口、新华医院）",
  "scenario": "只能是 elderly_friendly / life_circle / walkability / general 之一",
  "radius_m": 整数（用户未提及时默认800）,
  "need_streetview": true 或 false（用户明确要求分析街景时为true）,
  "out_of_scope": true 或 false（问题明显与杨浦区无关时为true）
}}

场景判断规则：
- elderly_friendly：包含老年、适老、无障碍、养老等
- life_circle：包含生活圈、设施配置、公共服务、15分钟等
- walkability：包含步行、慢行、街道环境、路口、骑行等
- general：无法归类"""),
    ("human", "{question}"),
])


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"无法从模型响应中提取 JSON：{text[:200]}")


# ---------------------------------------------------------------------------
# 节点函数
# ---------------------------------------------------------------------------

def node_parse_intent(state: PlannerState) -> dict:
    """解析用户问题的地点、场景、半径和街景需求。"""
    llm = _build_llm(temperature=0)
    chain = _PARSE_PROMPT | llm
    try:
        resp = chain.invoke({"question": state["question"]})
        intent = _parse_json(resp.content)
        logger.info("意图解析：%s", intent)
        return {"intent": intent, "error": ""}
    except Exception as e:
        return {"intent": {}, "error": f"[parse_intent] {e}"}


def node_geocode(state: PlannerState) -> dict:
    """地理编码：将地名转换为 WGS84 坐标。"""
    intent = state.get("intent", {})
    if state.get("error"):
        return {}

    # out_of_scope 检测
    if intent.get("out_of_scope"):
        return {
            "error": "out_of_scope",
            "answer": (
                f"您询问的地点「{intent.get('place', '')}」不在上海市杨浦区范围内，"
                "当前版本仅支持杨浦区城市更新分析。"
            ),
        }

    place = intent.get("place", "")
    if not place:
        return {"error": "[geocode] 未能从问题中识别地点名称"}

    try:
        result = geocode(place)
        if result is None:
            return {"error": f"[geocode] 未找到地点：{place}"}
        logger.info("地理编码：%s → WGS84(%.5f, %.5f) %s",
                    place, result.lon_wgs84, result.lat_wgs84,
                    "[杨浦]" if result.in_target_district else "[区外]")
        return {"geocode_result": result, "error": ""}
    except Exception as e:
        return {"error": f"[geocode] {e}"}


def node_query_spatial(state: PlannerState) -> dict:
    """POI + 路网并行查询（同一节点内执行，数据量不大无需异步）。"""
    if state.get("error"):
        return {}

    geo: GeocodeResult = state["geocode_result"]
    intent = state["intent"]
    radius_m: int = int(intent.get("radius_m", cfg.default_radius_m))
    scenario: str = intent.get("scenario", "general")
    lon, lat = geo.lon_wgs84, geo.lat_wgs84

    updates: dict[str, Any] = {}

    # --- POI 查询 ---
    try:
        poi_gdf = query_poi_by_buffer(lon, lat, radius_m=radius_m, scenario=scenario if scenario != "general" else None)
        summary_df = summarize_poi(poi_gdf)
        updates["poi_summary"] = summary_df.to_dict(orient="records")

        if scenario != "general":
            gap = check_facility_gaps(poi_gdf, scenario=scenario)
            updates["gap_report"] = gap
            logger.info("设施缺口诊断：缺失=%d 不足=%d", len(gap.missing), len(gap.insufficient))
        logger.info("POI 查询完成：%d 条（%dm 内）", len(poi_gdf), radius_m)
    except Exception as e:
        logger.warning("POI 查询失败：%s", e)

    # --- 路网查询 ---
    try:
        # 步行10分钟等时圈
        iso = query_isochrone(lon, lat, minutes=10, mode="walk")
        updates["isochrone"] = iso
        # 路口（500m 内，3 条以上道路）
        intersections = query_intersections(lon, lat, radius_m=min(radius_m, 600), min_street_count=3)
        updates["intersections"] = intersections
        logger.info("路网查询完成：等时圈节点=%d, 路口=%d", iso.node_count, len(intersections))
    except Exception as e:
        logger.warning("路网查询失败：%s", e)

    return updates


def node_query_streetview(state: PlannerState) -> dict:
    """街景查询与多模态分析（仅当 need_streetview=True 时执行）。"""
    if state.get("error"):
        return {}

    geo: GeocodeResult = state["geocode_result"]
    intent = state["intent"]
    scenario = intent.get("scenario", "general")
    lon, lat = geo.lon_wgs84, geo.lat_wgs84

    try:
        sv_results = query_and_analyze(
            lon=lon,
            lat=lat,
            radius_m=min(int(intent.get("radius_m", 800)), 500),
            scenario=scenario,
            limit=cfg.default_streetview_limit,
            use_cache=True,
        )
        logger.info("街景分析完成：%d 张", len(sv_results))
        return {"sv_results": sv_results}
    except Exception as e:
        logger.warning("街景查询失败：%s", e)
        return {"sv_results": []}


def node_query_policy(state: PlannerState) -> dict:
    """政策 RAG 多条查询，覆盖问题的不同侧面。"""
    if state.get("error"):
        return {}

    intent = state["intent"]
    scenario = intent.get("scenario", "general")
    place = intent.get("place", "")

    # 构造多条查询：原始问题 + 场景关键词拆解
    base_query = state["question"]
    extra_queries = cfg.scenario_policy_keywords(scenario) if scenario != "general" else []
    # 取前3条关键词，避免过多 API 调用
    queries = [base_query] + [f"{place} {kw}" for kw in extra_queries[:3]]

    try:
        chunks = query_policy_multi(
            queries=queries,
            top_k_per_query=3,
            scenario=scenario if scenario != "general" else None,
            min_score=0.52,
        )
        logger.info("政策检索完成：%d 条 chunk（去重后）", len(chunks))
        return {"policy_chunks": chunks}
    except Exception as e:
        logger.warning("政策查询失败：%s", e)
        return {"policy_chunks": []}


def node_generate(state: PlannerState) -> dict:
    """综合所有数据，用 LLM 生成结构化规划建议。"""
    if state.get("error") == "out_of_scope":
        return {}  # answer 已在 node_geocode 中设置
    if state.get("error"):
        return {"answer": f"处理过程中发生错误：{state['error']}"}

    geo: GeocodeResult = state.get("geocode_result")
    intent = state.get("intent", {})
    scenario = intent.get("scenario", "general")

    # --- 组装上下文 ---
    ctx_parts: list[str] = []

    # 1. 地点信息
    if geo:
        district_note = "（已确认在杨浦区）" if geo.in_target_district else "（行政区信息不确定，以坐标为准）"
        ctx_parts.append(f"**分析地点**：{geo.address} {district_note}\n坐标：WGS84({geo.lon_wgs84:.5f}, {geo.lat_wgs84:.5f})")

    # 2. POI 统计
    poi_summary: list[dict] = state.get("poi_summary") or []
    if poi_summary:
        radius_m = intent.get("radius_m", 800)
        lines = [f"**{radius_m}m 范围内设施统计**（{scenario} 场景）："]
        for row in poi_summary[:12]:
            lines.append(f"- {row['category_planning']}：{row['count']} 处，最近 {row['min_distance_m']}m")
        ctx_parts.append("\n".join(lines))

    # 3. 设施缺口
    gap: Optional[FacilityGapReport] = state.get("gap_report")
    if gap:
        parts = []
        if gap.missing:
            parts.append(f"完全缺失：{'、'.join(gap.missing)}")
        if gap.insufficient:
            parts.append(f"数量不足（<2处）：{'、'.join(gap.insufficient)}")
        if parts:
            ctx_parts.append("**设施缺口**：\n- " + "\n- ".join(parts))

    # 4. 路网信息
    iso: Optional[IsochroneResult] = state.get("isochrone")
    intersections: list[IntersectionResult] = state.get("intersections") or []
    if iso or intersections:
        road_lines = []
        if iso:
            road_lines.append(f"步行10分钟可达路网节点：{iso.node_count} 个")
        if intersections:
            top3 = intersections[:3]
            desc = "；".join([f"{r.street_count}路口（{r.distance_m}m）" for r in top3])
            road_lines.append(f"周边主要路口（500m内）：{len(intersections)} 处，含 {desc}")
        ctx_parts.append("**路网分析**：\n" + "\n".join(f"- {line}" for line in road_lines))

    # 5. 街景分析
    sv_results: list[StreetviewAnalysis] = state.get("sv_results") or []
    if sv_results:
        sv_lines = [f"**街景分析**（{len(sv_results)} 张，场景：{scenario}）："]
        for r in sv_results[:3]:
            if r.analysis_text and not r.analysis_text.startswith("["):
                # 截取前120字
                preview = r.analysis_text[:120].replace("\n", " ")
                sv_lines.append(f"- {r.meta.image_id}（{r.meta.direction_bucket}向，{r.meta.distance_m}m）：{preview}…")
        ctx_parts.append("\n".join(sv_lines))

    # 6. 政策依据
    policy_chunks: list[PolicyChunk] = state.get("policy_chunks") or []
    if policy_chunks:
        citations = format_citations(policy_chunks[:6], max_text_length=120, group_by_doc=True)
        ctx_parts.append(f"**政策依据**：\n{citations}")

    context_text = "\n\n".join(ctx_parts)

    # --- 生成建议 ---
    scenario_desc = {
        "elderly_friendly": "老年友好与适老化改造",
        "life_circle": "15分钟生活圈设施完善",
        "walkability": "步行友好与慢行交通",
        "general": "城市更新",
    }.get(scenario, "城市更新")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位上海市杨浦区城市更新规划专家。
根据提供的空间数据和政策依据，为用户生成具体、可操作的规划诊断与建议。

要求：
1. 针对具体空间问题，给出3-5条优先级排序的更新建议（标注优先级：高/中/低）
2. 每条建议须说明：①具体问题 ②建议措施 ③政策依据（引用提供的文献）
3. 避免空洞表述，建议要指向具体位置或节点
4. 最后给出一段50字以内的综合评价"""),
        ("human", """用户问题：{question}

分析场景：{scenario_desc}

空间与政策数据：
{context}

请输出结构化规划建议。"""),
    ])

    llm = _build_llm(temperature=0.3)
    chain = prompt | llm
    try:
        resp = chain.invoke({
            "question": state["question"],
            "scenario_desc": scenario_desc,
            "context": context_text,
        })
        return {"answer": resp.content}
    except Exception as e:
        return {"answer": f"建议生成失败：{e}"}


# ---------------------------------------------------------------------------
# 条件边：是否执行街景节点
# ---------------------------------------------------------------------------

def _route_after_spatial(state: PlannerState) -> str:
    if state.get("error"):
        return "generate"
    if state.get("intent", {}).get("need_streetview"):
        return "query_streetview"
    return "query_policy"


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """构建并编译 LangGraph 工作流图。"""
    builder = StateGraph(PlannerState)

    builder.add_node("parse_intent", node_parse_intent)
    builder.add_node("geocode", node_geocode)
    builder.add_node("query_spatial", node_query_spatial)
    builder.add_node("query_streetview", node_query_streetview)
    builder.add_node("query_policy", node_query_policy)
    builder.add_node("generate", node_generate)

    builder.add_edge(START, "parse_intent")
    builder.add_edge("parse_intent", "geocode")
    builder.add_edge("geocode", "query_spatial")
    builder.add_conditional_edges(
        "query_spatial",
        _route_after_spatial,
        {
            "query_streetview": "query_streetview",
            "query_policy": "query_policy",
            "generate": "generate",
        },
    )
    builder.add_edge("query_streetview", "query_policy")
    builder.add_edge("query_policy", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# 便捷调用入口
# ---------------------------------------------------------------------------

def run(question: str) -> str:
    """
    运行 Agent，返回规划建议文本。

    Args:
        question: 用户自然语言问题，如"请分析鞍山新村周边800米的老年友好问题"

    Returns:
        结构化规划建议 Markdown 文本
    """
    graph = build_graph()
    initial: PlannerState = {
        "question": question,
        "intent": {},
        "geocode_result": None,
        "poi_summary": None,
        "gap_report": None,
        "isochrone": None,
        "intersections": [],
        "sv_results": [],
        "policy_chunks": [],
        "answer": "",
        "error": "",
    }
    final_state = graph.invoke(initial)
    return final_state.get("answer", "")
