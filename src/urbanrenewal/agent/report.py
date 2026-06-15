"""Structured report assembly for the autonomous UrbanRenewal Agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """A compact evidence record that frontends can render outside Markdown."""

    kind: Literal["location", "poi", "gap", "road", "streetview", "policy", "tool"]
    title: str
    summary: str = ""
    source: str = ""
    data: dict[str, Any] | list[Any] | None = None


class MapLayer(BaseModel):
    """A map-renderable layer extracted from tool results."""

    layer_id: str
    name: str
    layer_type: Literal["point", "polygon", "line", "geojson"]
    geojson: dict[str, Any]
    visible: bool = True


class Issue(BaseModel):
    """A product-card issue inferred from deterministic tool outputs."""

    title: str
    severity: Literal["high", "medium", "low", "unknown"] = "unknown"
    summary: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """A structured recommendation card for frontend rendering."""

    title: str
    priority: Literal["high", "medium", "low", "unknown"] = "unknown"
    action: str = ""
    rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class PolicyCitation(BaseModel):
    """Policy citation metadata extracted from RAG tool results."""

    title: str = "政策 RAG 依据"
    excerpt_markdown: str = ""
    source: str = ""
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"


class PlanningReport(BaseModel):
    """Product-facing structured result for API and UI rendering."""

    answer_markdown: str = ""
    locations: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    spatial_evidence: list[EvidenceItem] = Field(default_factory=list)
    policy_evidence: list[EvidenceItem] = Field(default_factory=list)
    policy_citations: list[PolicyCitation] = Field(default_factory=list)
    map_layers: list[MapLayer] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


def _point_feature(lon: float, lat: float, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": properties,
    }


def _feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def _tool_result(event: dict[str, Any]) -> dict[str, Any]:
    result = event.get("result")
    return result if isinstance(result, dict) else {}


def _severity_from_gap(data: dict[str, Any]) -> Literal["high", "medium", "low", "unknown"]:
    missing = data.get("missing") or []
    insufficient = data.get("insufficient") or []
    if missing:
        return "high"
    if insufficient:
        return "medium"
    return "low"


def _add_gap_cards(report: PlanningReport, data: dict[str, Any], summary: str) -> None:
    missing = data.get("missing") or []
    insufficient = data.get("insufficient") or []
    severity = _severity_from_gap(data)
    if not missing and not insufficient:
        report.issues.append(Issue(
            title="设施配置未发现明显缺口",
            severity="low",
            summary=summary,
            evidence_refs=["facility_gap_tool"],
        ))
        return

    title_parts = []
    if missing:
        title_parts.append(f"缺失：{'、'.join(str(x) for x in missing[:4])}")
    if insufficient:
        title_parts.append(f"不足：{'、'.join(str(x) for x in insufficient[:4])}")
    title = "；".join(title_parts)
    report.issues.append(Issue(
        title=title,
        severity=severity,
        summary=summary,
        evidence_refs=["facility_gap_tool", "poi_query_tool"],
    ))
    report.recommendations.append(Recommendation(
        title="优先补齐场景关键公共服务设施",
        priority=severity if severity in ("high", "medium", "low") else "medium",
        action="结合缺失和不足类别，优先在分析范围内布置高频、刚需、步行可达的公共服务设施。",
        rationale=summary,
        evidence_refs=["facility_gap_tool", "policy_rag_tool"],
    ))


def build_planning_report(answer_markdown: str, tool_events: list[dict[str, Any]]) -> PlanningReport:
    """Build a structured report from the final answer and parsed tool events."""
    report = PlanningReport(answer_markdown=answer_markdown, tool_trace=tool_events)

    for event in tool_events:
        if event.get("status") != "completed":
            continue

        tool_name = event.get("tool_name", "unknown")
        result = _tool_result(event)
        if not result:
            continue

        if result.get("ok") is False and result.get("error"):
            report.uncertainties.append(f"{tool_name} 调用失败：{result['error']}")
            continue

        data = result.get("data") or {}
        summary = result.get("summary", "")
        source = result.get("source", "")

        if tool_name == "geocode_tool" and isinstance(data, dict):
            location = {
                "name": data.get("name") or data.get("address"),
                "address": data.get("address"),
                "district": data.get("district"),
                "city": data.get("city"),
                "lon_wgs84": data.get("lon_wgs84"),
                "lat_wgs84": data.get("lat_wgs84"),
                "in_target_district": data.get("in_target_district"),
            }
            report.locations.append(location)
            report.spatial_evidence.append(EvidenceItem(kind="location", title="地点解析", summary=summary, source=source, data=location))
            lon, lat = data.get("lon_wgs84"), data.get("lat_wgs84")
            if isinstance(lon, int | float) and isinstance(lat, int | float):
                report.map_layers.append(MapLayer(
                    layer_id="analysis_center",
                    name="分析中心点",
                    layer_type="point",
                    geojson=_feature_collection([_point_feature(lon, lat, {"name": location["name"], "type": "analysis_center"})]),
                ))

        elif tool_name == "poi_query_tool" and isinstance(data, dict):
            report.spatial_evidence.append(EvidenceItem(
                kind="poi",
                title="周边设施统计",
                summary=summary,
                source=source,
                data=data,
            ))
            geojson = data.get("geojson")
            if isinstance(geojson, dict) and geojson.get("features"):
                report.map_layers.append(MapLayer(
                    layer_id="poi",
                    name="周边设施 POI",
                    layer_type="point",
                    geojson=geojson,
                ))

        elif tool_name == "facility_gap_tool":
            report.spatial_evidence.append(EvidenceItem(
                kind="gap",
                title="设施缺口诊断",
                summary=summary,
                source=source,
                data=data if isinstance(data, dict) else None,
            ))
            if isinstance(data, dict):
                _add_gap_cards(report, data, summary)

        elif tool_name == "isochrone_tool" and isinstance(data, dict):
            report.spatial_evidence.append(EvidenceItem(kind="road", title="等时圈分析", summary=summary, source=source, data=data))
            geojson = data.get("geojson")
            if isinstance(geojson, dict):
                report.map_layers.append(MapLayer(
                    layer_id="isochrone",
                    name=f"{data.get('mode', 'walk')} {data.get('minutes', '')}分钟等时圈",
                    layer_type="geojson",
                    geojson=geojson,
                ))

        elif tool_name == "intersection_tool" and isinstance(data, dict):
            report.spatial_evidence.append(EvidenceItem(kind="road", title="主要路口", summary=summary, source=source, data=data))
            features = []
            for row in data.get("intersections", []) or []:
                if not isinstance(row, dict):
                    continue
                lon, lat = row.get("lon"), row.get("lat")
                if isinstance(lon, int | float) and isinstance(lat, int | float):
                    features.append(_point_feature(lon, lat, {"type": "intersection", **row}))
            if features:
                report.map_layers.append(MapLayer(
                    layer_id="intersections",
                    name="主要路口",
                    layer_type="point",
                    geojson=_feature_collection(features),
                    visible=False,
                ))

        elif tool_name == "streetview_tool" and isinstance(data, dict):
            report.spatial_evidence.append(EvidenceItem(kind="streetview", title="街景分析", summary=summary, source=source, data=data))
            features = []
            for row in data.get("results", []) or []:
                if not isinstance(row, dict):
                    continue
                meta = row.get("meta") if isinstance(row.get("meta"), dict) else row
                lon, lat = meta.get("lon_wgs84"), meta.get("lat_wgs84")
                if isinstance(lon, int | float) and isinstance(lat, int | float):
                    features.append(_point_feature(lon, lat, {
                        "type": "streetview",
                        "image_id": meta.get("image_id"),
                        "distance_m": meta.get("distance_m"),
                        "direction": meta.get("direction_bucket"),
                    }))
            if features:
                report.map_layers.append(MapLayer(
                    layer_id="streetview",
                    name="街景点位",
                    layer_type="point",
                    geojson=_feature_collection(features),
                    visible=False,
                ))

        elif tool_name == "policy_rag_tool" and isinstance(data, dict):
            citations_markdown = data.get("citations_markdown", "")
            low_confidence = data.get("low_confidence", False)
            report.policy_evidence.append(EvidenceItem(
                kind="policy",
                title="政策 RAG 依据",
                summary=summary,
                source=source,
                data={
                    "queries": data.get("queries", []),
                    "count": data.get("count", 0),
                    "citations_markdown": citations_markdown,
                    "low_confidence": low_confidence,
                },
            ))
            report.policy_citations.append(PolicyCitation(
                excerpt_markdown=citations_markdown,
                source=source,
                confidence="low" if low_confidence else "medium",
            ))
            if low_confidence:
                report.uncertainties.append("政策 RAG 未检索到高置信度片段，政策依据需要人工复核。")

    return report
