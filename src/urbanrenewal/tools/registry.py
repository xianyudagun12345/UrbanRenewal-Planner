"""
LangChain tool registry for the autonomous UrbanRenewal Agent.

The existing tool modules return rich Python objects that are convenient for the
fixed workflow, but awkward for LLM tool calling, API responses, and tests. This
module wraps them behind a small, serializable contract:

    {ok, data, summary, error, source, cost_hint}
"""

from __future__ import annotations

import json
import logging
from hashlib import sha256
from dataclasses import asdict, is_dataclass
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.urbanrenewal.agent.budget import budget_for_tier
from src.urbanrenewal.utils.ttl_cache import TTLCache

logger = logging.getLogger(__name__)

Scenario = Literal["elderly_friendly", "life_circle", "walkability", "general"]
AudienceMode = Literal["professional", "public", "government"]

_DEFAULT_BUDGET = budget_for_tier("registered")

MAX_RADIUS_M = _DEFAULT_BUDGET.max_radius_m
MAX_STREETVIEW_LIMIT = _DEFAULT_BUDGET.max_streetview_images
MAX_POLICY_QUERIES = _DEFAULT_BUDGET.max_policy_queries
MAX_POLICY_TOP_K = _DEFAULT_BUDGET.max_policy_top_k

_GEOCODE_CACHE: TTLCache[dict[str, Any]] = TTLCache(ttl_seconds=60 * 60 * 24 * 7, max_items=512)
_POLICY_CACHE: TTLCache[dict[str, Any]] = TTLCache(ttl_seconds=60 * 60 * 6, max_items=256)


def _jsonable(value: Any) -> Any:
    """Convert common project objects to JSON-friendly values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_geojson"):
        return _jsonable(value.to_geojson())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


def _ok(
    *,
    data: Any,
    summary: str,
    source: str,
    cost_hint: str = "low",
) -> dict[str, Any]:
    return {
        "ok": True,
        "data": _jsonable(data),
        "summary": summary,
        "error": "",
        "source": source,
        "cost_hint": cost_hint,
    }


def _err(
    *,
    error: str,
    source: str,
    summary: str = "",
    cost_hint: str = "low",
) -> dict[str, Any]:
    logger.warning("Tool failed: %s | %s", source, error)
    return {
        "ok": False,
        "data": None,
        "summary": summary or "工具调用失败，已返回结构化错误。",
        "error": error,
        "source": source,
        "cost_hint": cost_hint,
    }


def _clamp_radius(radius_m: int) -> int:
    return max(50, min(int(radius_m), MAX_RADIUS_M))


def _hash_key(prefix: str, payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}:{sha256(text.encode('utf-8')).hexdigest()}"


class GeocodeInput(BaseModel):
    place: str = Field(..., min_length=1, max_length=120, description="上海市杨浦区内的地名、道路或路口。")


class POIQueryInput(BaseModel):
    lon: float = Field(..., description="WGS84 经度。")
    lat: float = Field(..., description="WGS84 纬度。")
    radius_m: int = Field(800, ge=50, le=MAX_RADIUS_M, description="查询半径，单位米。")
    scenario: Scenario = Field("general", description="分析场景。")
    categories: list[str] | None = Field(None, description="可选 POI 规划类别过滤列表。")


class FacilityGapInput(BaseModel):
    lon: float = Field(..., description="WGS84 经度。")
    lat: float = Field(..., description="WGS84 纬度。")
    radius_m: int = Field(800, ge=50, le=MAX_RADIUS_M, description="诊断半径，单位米。")
    scenario: Literal["elderly_friendly", "life_circle", "walkability"] = Field(..., description="设施诊断场景。")


class IsochroneInput(BaseModel):
    lon: float = Field(..., description="WGS84 经度。")
    lat: float = Field(..., description="WGS84 纬度。")
    minutes: float = Field(10, ge=3, le=30, description="步行或骑行时间阈值，单位分钟。")
    mode: Literal["walk", "bike"] = Field("walk", description="出行方式。")


class IntersectionInput(BaseModel):
    lon: float = Field(..., description="WGS84 经度。")
    lat: float = Field(..., description="WGS84 纬度。")
    radius_m: int = Field(600, ge=50, le=MAX_RADIUS_M, description="查询半径，单位米。")
    min_street_count: int = Field(3, ge=2, le=6, description="最少连接道路数。")


class StreetviewInput(BaseModel):
    lon: float = Field(..., description="WGS84 经度。")
    lat: float = Field(..., description="WGS84 纬度。")
    radius_m: int = Field(500, ge=50, le=800, description="街景检索半径，单位米。")
    scenario: Scenario = Field("general", description="分析场景。")
    limit: int = Field(3, ge=1, le=MAX_STREETVIEW_LIMIT, description="最多分析图片数。")
    use_cache: bool = Field(True, description="是否优先读取本地分析缓存。")


class PolicyRAGInput(BaseModel):
    queries: list[str] = Field(..., min_length=1, max_length=MAX_POLICY_QUERIES, description="政策检索查询列表。")
    scenario: Scenario = Field("general", description="分析场景。")
    top_k_per_query: int = Field(3, ge=1, le=MAX_POLICY_TOP_K, description="每条查询最多返回片段数。")
    min_score: float = Field(0.52, ge=0.0, le=1.0, description="最低相似度阈值。")


class ReportGenerationInput(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="用户原始问题。")
    evidence_json: str = Field(..., min_length=2, description="工具证据 JSON 字符串。")
    audience: AudienceMode = Field("professional", description="输出面向对象。")


@tool("geocode_tool", args_schema=GeocodeInput)
def geocode_tool(place: str) -> dict[str, Any]:
    """将地名解析为 WGS84 坐标，并判断是否位于上海市杨浦区。"""
    try:
        from src.urbanrenewal.tools.geocode import geocode

        cache_key = _hash_key("geocode", {"place": place})
        cached = _GEOCODE_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True, "cost_hint": "cache"}

        result = geocode(place)
        if result is None:
            return _err(error=f"未找到地点：{place}", source="amap_geocode")
        flag = "位于杨浦区" if result.in_target_district else "不在杨浦区或行政区不确定"
        payload = _ok(
            data=result,
            summary=f"{result.address}，WGS84({result.lon_wgs84:.5f}, {result.lat_wgs84:.5f})，{flag}。",
            source="amap_geocode",
            cost_hint="network",
        )
        payload["cached"] = False
        _GEOCODE_CACHE.set(cache_key, payload)
        return payload
    except Exception as exc:
        return _err(error=str(exc), source="amap_geocode", cost_hint="network")


@tool("poi_query_tool", args_schema=POIQueryInput)
def poi_query_tool(
    lon: float,
    lat: float,
    radius_m: int = 800,
    scenario: Scenario = "general",
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """查询坐标周边 POI，并按规划类别返回数量、平均距离和最近距离。"""
    try:
        from src.urbanrenewal.tools.poi_query import query_poi_by_buffer, summarize_poi, to_geojson

        radius = _clamp_radius(radius_m)
        gdf = query_poi_by_buffer(
            lon,
            lat,
            radius_m=radius,
            scenario=None if scenario == "general" else scenario,
            categories=categories,
        )
        summary = summarize_poi(gdf).to_dict(orient="records")
        return _ok(
            data={
                "radius_m": radius,
                "scenario": scenario,
                "total_count": int(len(gdf)),
                "summary": summary,
                "geojson": to_geojson(gdf, max_features=200),
            },
            summary=f"{radius}m 范围内检索到 {len(gdf)} 个相关 POI，形成 {len(summary)} 类设施统计。",
            source="local_poi_parquet",
        )
    except Exception as exc:
        return _err(error=str(exc), source="local_poi_parquet")


@tool("facility_gap_tool", args_schema=FacilityGapInput)
def facility_gap_tool(
    lon: float,
    lat: float,
    radius_m: int = 800,
    scenario: Literal["elderly_friendly", "life_circle", "walkability"] = "elderly_friendly",
) -> dict[str, Any]:
    """按场景配置诊断设施缺失、数量不足和已配置类别。"""
    try:
        from src.urbanrenewal.tools.poi_query import check_facility_gaps, query_poi_by_buffer

        radius = _clamp_radius(radius_m)
        gdf = query_poi_by_buffer(lon, lat, radius_m=radius, scenario=scenario)
        gap = check_facility_gaps(gdf, scenario=scenario)
        return _ok(
            data=gap,
            summary=f"{scenario} 场景下：缺失 {len(gap.missing)} 类，不足 {len(gap.insufficient)} 类，已配置 {len(gap.present)} 类。",
            source="local_poi_parquet",
        )
    except Exception as exc:
        return _err(error=str(exc), source="local_poi_parquet")


@tool("isochrone_tool", args_schema=IsochroneInput)
def isochrone_tool(
    lon: float,
    lat: float,
    minutes: float = 10,
    mode: Literal["walk", "bike"] = "walk",
) -> dict[str, Any]:
    """计算步行或骑行等时圈，返回节点数量和 GeoJSON。"""
    try:
        from src.urbanrenewal.tools.road_query import query_isochrone

        result = query_isochrone(lon, lat, minutes=minutes, mode=mode)
        return _ok(
            data={
                "center_lon": lon,
                "center_lat": lat,
                "minutes": minutes,
                "mode": mode,
                "node_count": result.node_count,
                "geojson": result.to_geojson(),
            },
            summary=f"{mode} {minutes:g} 分钟等时圈包含 {result.node_count} 个路网节点。",
            source="local_road_graph",
        )
    except Exception as exc:
        return _err(error=str(exc), source="local_road_graph")


@tool("intersection_tool", args_schema=IntersectionInput)
def intersection_tool(
    lon: float,
    lat: float,
    radius_m: int = 600,
    min_street_count: int = 3,
) -> dict[str, Any]:
    """查询周边主要路口节点。"""
    try:
        from src.urbanrenewal.tools.road_query import query_intersections

        radius = _clamp_radius(radius_m)
        rows = query_intersections(lon, lat, radius_m=radius, min_street_count=min_street_count)
        return _ok(
            data={
                "radius_m": radius,
                "min_street_count": min_street_count,
                "intersections": rows[:50],
            },
            summary=f"{radius}m 范围内找到 {len(rows)} 个主要路口。",
            source="local_road_graph",
        )
    except Exception as exc:
        return _err(error=str(exc), source="local_road_graph")


@tool("streetview_tool", args_schema=StreetviewInput)
def streetview_tool(
    lon: float,
    lat: float,
    radius_m: int = 500,
    scenario: Scenario = "general",
    limit: int = MAX_STREETVIEW_LIMIT,
    use_cache: bool = True,
) -> dict[str, Any]:
    """按预算检索并分析周边街景图片，返回多模态环境诊断。"""
    try:
        from src.urbanrenewal.tools.streetview_query import query_and_analyze

        capped_limit = min(int(limit), MAX_STREETVIEW_LIMIT)
        rows = query_and_analyze(
            lon=lon,
            lat=lat,
            radius_m=min(int(radius_m), 800),
            scenario=scenario,
            limit=capped_limit,
            use_cache=use_cache,
        )
        cached = sum(1 for row in rows if row.from_cache)
        return _ok(
            data={
                "scenario": scenario,
                "limit": capped_limit,
                "count": len(rows),
                "cached_count": cached,
                "results": rows,
            },
            summary=f"分析街景 {len(rows)} 张，其中缓存命中 {cached} 张；单次上限 {MAX_STREETVIEW_LIMIT} 张。",
            source="local_streetview_cache_and_vl_model",
            cost_hint="high",
        )
    except Exception as exc:
        return _err(error=str(exc), source="local_streetview_cache_and_vl_model", cost_hint="high")


@tool("policy_rag_tool", args_schema=PolicyRAGInput)
def policy_rag_tool(
    queries: list[str],
    scenario: Scenario = "general",
    top_k_per_query: int = 3,
    min_score: float = 0.52,
) -> dict[str, Any]:
    """检索政策 RAG，返回可引用的政策片段。"""
    try:
        from src.urbanrenewal.tools.policy_rag import format_citations, query_policy_multi

        clean_queries = [q.strip() for q in queries if q and q.strip()][:MAX_POLICY_QUERIES]
        cache_key = _hash_key("policy_rag", {
            "queries": clean_queries,
            "scenario": scenario,
            "top_k_per_query": min(int(top_k_per_query), MAX_POLICY_TOP_K),
            "min_score": min_score,
        })
        cached = _POLICY_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True, "cost_hint": "cache"}

        chunks = query_policy_multi(
            queries=clean_queries,
            top_k_per_query=min(int(top_k_per_query), MAX_POLICY_TOP_K),
            scenario=None if scenario == "general" else scenario,
            min_score=min_score,
        )
        citations = format_citations(chunks[:8], max_text_length=160, group_by_doc=True)
        payload = _ok(
            data={
                "queries": clean_queries,
                "scenario": scenario,
                "count": len(chunks),
                "chunks": chunks[:12],
                "citations_markdown": citations,
                "low_confidence": len(chunks) == 0,
            },
            summary=f"政策 RAG 返回 {len(chunks)} 条去重片段。" if chunks else "未检索到高置信度政策片段。",
            source="policy_chroma_rag",
            cost_hint="network",
        )
        payload["cached"] = False
        _POLICY_CACHE.set(cache_key, payload)
        return payload
    except Exception as exc:
        return _err(error=str(exc), source="policy_chroma_rag", cost_hint="network")


@tool("report_generation_tool", args_schema=ReportGenerationInput)
def report_generation_tool(
    question: str,
    evidence_json: str,
    audience: AudienceMode = "professional",
) -> dict[str, Any]:
    """根据工具证据生成结构化规划报告。适合在完成必要工具调用后使用。"""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from src.urbanrenewal.agent.llm import build_llm

        try:
            parsed = json.loads(evidence_json)
        except json.JSONDecodeError:
            parsed = {"raw_evidence": evidence_json}

        audience_desc = {
            "professional": "面向城市规划师，保留专业术语、空间证据和政策依据。",
            "public": "面向居民，用清晰直白的语言解释问题、影响和改善建议。",
            "government": "面向管理者，突出优先级、实施主体、阶段安排和风险。",
        }[audience]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是上海市杨浦区城市更新规划专家。请严格依据工具证据生成建议，不要编造政策、坐标或设施数量。
输出 Markdown，包含：总体诊断、主要问题、空间证据、政策依据、更新建议、优先级、实施主体、预期效果、不确定性说明。"""),
            ("human", """用户问题：{question}

输出对象：{audience_desc}

工具证据 JSON：
{evidence}

请生成结构化规划报告。"""),
        ])
        chain = prompt | build_llm(temperature=0.25)
        resp = chain.invoke({
            "question": question,
            "audience_desc": audience_desc,
            "evidence": json.dumps(parsed, ensure_ascii=False, indent=2),
        })
        return _ok(
            data={"markdown": resp.content, "audience": audience},
            summary="已基于工具证据生成结构化规划报告。",
            source="llm_report_generation",
            cost_hint="network",
        )
    except Exception as exc:
        return _err(error=str(exc), source="llm_report_generation", cost_hint="network")


ALL_TOOLS = [
    geocode_tool,
    poi_query_tool,
    facility_gap_tool,
    isochrone_tool,
    intersection_tool,
    streetview_tool,
    policy_rag_tool,
    report_generation_tool,
]

TOOL_NAMES = [t.name for t in ALL_TOOLS]
