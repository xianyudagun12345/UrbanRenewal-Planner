"""
UrbanRenewal Planner Autonomous Agent Streamlit UI.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import folium
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent / ".env", override=False)

from src.urbanrenewal.agent.autonomous import run_autonomous  # noqa: E402
from src.urbanrenewal.agent.plan import TaskPlan  # noqa: E402
from src.urbanrenewal.agent.report import PlanningReport  # noqa: E402
from src.urbanrenewal.config import cfg  # noqa: E402


st.set_page_config(
    page_title="UrbanRenewal Planner",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background: #f6f8fb; }
.stApp { color: #172033; }
[data-testid="stHeader"] { background: rgba(246, 248, 251, .92); }
[data-testid="stSidebar"] { background: #eef3f8; border-right: 1px solid #d7dee8; }
[data-testid="stSidebar"] * { color: #172033 !important; }
.app-title { font-size: 1.75rem; font-weight: 750; color: #173b57; margin: 0 0 .2rem 0; letter-spacing: 0; }
.app-subtitle { font-size: .94rem; color: #526173; margin-bottom: 1rem; }
.stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption, label { color: #172033; }
.stChatMessage {
    background: #ffffff;
    border: 1px solid #dfe6ef;
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(20, 35, 55, .04);
}
[data-testid="stChatInput"] textarea {
    color: #172033 !important;
    background: #ffffff !important;
    border: 1px solid #c9d4e2 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #64748b !important; opacity: 1; }
.stButton > button {
    background: #ffffff;
    color: #173b57;
    border: 1px solid #c9d4e2;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover {
    border-color: #2f6f9f;
    color: #0f4f78;
    background: #f4f9fc;
}
.tool-card {
    border: 1px solid #d9e2ec;
    background: #ffffff;
    border-radius: 8px;
    padding: .55rem .7rem;
    margin: .35rem 0;
}
.tool-name { color: #0f5f8f; font-weight: 650; font-size: .86rem; }
.tool-status { color: #64748b; font-size: .76rem; margin-left: .35rem; }
.tool-summary { color: #263548; font-size: .82rem; margin-top: .25rem; }
.section-title { color: #173b57; font-size: 1.05rem; font-weight: 750; margin: .7rem 0 .35rem 0; }
.result-card {
    border: 1px solid #d9e2ec;
    background: #ffffff;
    border-radius: 8px;
    padding: .7rem .85rem;
    margin: .45rem 0;
    box-shadow: 0 1px 2px rgba(20, 35, 55, .04);
}
.card-title { color: #172033; font-weight: 700; font-size: .92rem; }
.card-meta { color: #64748b; font-size: .76rem; margin-top: .15rem; }
.card-body { color: #263548; font-size: .84rem; margin-top: .35rem; line-height: 1.55; }
.priority-high { color: #b42318; font-weight: 650; }
.priority-medium { color: #93640b; font-weight: 650; }
.priority-low { color: #087443; font-weight: 650; }
.priority-unknown { color: #64748b; font-weight: 650; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #172033 !important; }
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #dfe6ef;
    border-radius: 8px;
}
#MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


def _init_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_tool_events" not in st.session_state:
        st.session_state.last_tool_events = []


def _render_tool_events(events: list[dict]) -> None:
    if not events:
        st.caption("本轮没有工具调用，Agent 可能正在澄清问题或直接回答。")
        return
    for event in events:
        name = event.get("tool_name", "unknown")
        status = event.get("status", "")
        summary = (event.get("summary") or "").replace("\n", " ")
        if len(summary) > 360:
            summary = summary[:360] + "..."
        st.markdown(
            f"""
<div class="tool-card">
  <span class="tool-name">{name}</span><span class="tool-status">{status}</span>
  <div class="tool-summary">{summary}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def _coerce_task_plan(value: Any) -> TaskPlan | None:
    if value is None:
        return None
    if isinstance(value, TaskPlan):
        return value
    if isinstance(value, dict):
        return TaskPlan.model_validate(value)
    return None


def _render_task_plan(value: Any) -> None:
    plan = _coerce_task_plan(value)
    if plan is None:
        return
    with st.expander("任务预规划", expanded=plan.clarification.needed):
        cols = st.columns(4)
        cols[0].metric("场景", plan.scenario)
        cols[1].metric("半径", f"{plan.radius_m}m")
        cols[2].metric("地点数", len(plan.places))
        cols[3].metric("需澄清", "是" if plan.clarification.needed else "否")
        if plan.places:
            st.caption("识别地点：" + "、".join(plan.places))
        if plan.suggested_tools:
            st.caption("建议工具：" + "、".join(plan.suggested_tools))
        if plan.clarification.needed:
            st.warning(plan.clarification.question)
        st.caption(plan.reasoning)


def _coerce_report(value: Any) -> PlanningReport | None:
    if value is None:
        return None
    if isinstance(value, PlanningReport):
        return value
    if isinstance(value, dict):
        return PlanningReport.model_validate(value)
    return None


def _badge_class(value: str) -> str:
    if value in {"high", "medium", "low"}:
        return f"priority-{value}"
    return "priority-unknown"


def _render_issue_cards(report: PlanningReport) -> None:
    if not report.issues:
        return
    st.markdown('<div class="section-title">关键问题</div>', unsafe_allow_html=True)
    for issue in report.issues:
        severity = issue.severity
        refs = " / ".join(issue.evidence_refs)
        st.markdown(
            f"""
<div class="result-card">
  <div class="card-title">{issue.title}</div>
  <div class="card-meta">严重程度：<span class="{_badge_class(severity)}">{severity}</span>{' · 证据：' + refs if refs else ''}</div>
  <div class="card-body">{issue.summary}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def _render_recommendation_cards(report: PlanningReport) -> None:
    if not report.recommendations:
        return
    st.markdown('<div class="section-title">行动建议</div>', unsafe_allow_html=True)
    for rec in report.recommendations:
        refs = " / ".join(rec.evidence_refs)
        st.markdown(
            f"""
<div class="result-card">
  <div class="card-title">{rec.title}</div>
  <div class="card-meta">优先级：<span class="{_badge_class(rec.priority)}">{rec.priority}</span>{' · 证据：' + refs if refs else ''}</div>
  <div class="card-body"><b>措施：</b>{rec.action}<br><b>依据：</b>{rec.rationale}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def _render_policy_citations(report: PlanningReport) -> None:
    if not report.policy_citations:
        return
    st.markdown('<div class="section-title">政策依据</div>', unsafe_allow_html=True)
    for citation in report.policy_citations:
        with st.expander(f"{citation.title} · 置信度 {citation.confidence}", expanded=False):
            if citation.excerpt_markdown:
                st.markdown(citation.excerpt_markdown)
            st.caption(f"来源：{citation.source or '未标注'}")


def _render_uncertainties(report: PlanningReport) -> None:
    if not report.uncertainties:
        return
    with st.expander("不确定性与需复核事项", expanded=False):
        for item in report.uncertainties:
            st.warning(item)


def _first_location(report: PlanningReport) -> tuple[float, float] | None:
    for loc in report.locations:
        lon, lat = loc.get("lon_wgs84"), loc.get("lat_wgs84")
        if isinstance(lon, int | float) and isinstance(lat, int | float):
            return float(lat), float(lon)
    for layer in report.map_layers:
        for feature in layer.geojson.get("features", []):
            coords = feature.get("geometry", {}).get("coordinates")
            if isinstance(coords, list) and len(coords) >= 2:
                lon, lat = coords[0], coords[1]
                if isinstance(lon, int | float) and isinstance(lat, int | float):
                    return float(lat), float(lon)
    return None


def _render_report_map(report: PlanningReport) -> None:
    if not report.map_layers:
        return
    center = _first_location(report) or (31.27, 121.52)
    fmap = folium.Map(location=list(center), zoom_start=15, tiles="CartoDB positron", prefer_canvas=True)

    for layer in report.map_layers:
        feature_group = folium.FeatureGroup(name=layer.name, show=layer.visible)
        folium.GeoJson(
            layer.geojson,
            name=layer.name,
            tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["名称"], labels=True)
            if _geojson_has_property(layer.geojson, "name")
            else None,
        ).add_to(feature_group)
        feature_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    st.markdown('<div class="section-title">空间图层</div>', unsafe_allow_html=True)
    st_folium(fmap, width=None, height=420, returned_objects=[])


def _geojson_has_property(geojson: dict[str, Any], key: str) -> bool:
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        if key in props:
            return True
    return False


def _render_report(report_value: Any) -> None:
    report = _coerce_report(report_value)
    if report is None:
        return
    left, right = st.columns([1, 1], gap="large")
    with left:
        _render_issue_cards(report)
        _render_recommendation_cards(report)
        _render_policy_citations(report)
        _render_uncertainties(report)
    with right:
        _render_report_map(report)
        if report.spatial_evidence:
            with st.expander("空间证据摘要", expanded=False):
                for evidence in report.spatial_evidence:
                    st.markdown(f"**{evidence.title}**")
                    st.caption(f"{evidence.summary} · {evidence.source}")


_init_state()

with st.sidebar:
    st.markdown("### UrbanRenewal Planner")
    st.caption(f"{cfg.city}{cfg.district} 自主城市更新 Agent")
    st.divider()
    st.markdown("**会话**")
    st.code(st.session_state.thread_id, language="text")
    if st.button("新建会话", use_container_width=True):
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages = []
        st.session_state.last_tool_events = []
        st.rerun()
    if st.button("清空当前对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_tool_events = []
        st.rerun()
    st.divider()
    st.markdown("**运行模型**")
    st.caption(f"LLM: {cfg.llm_model}")
    st.caption(f"VL: {cfg.vl_model}")
    st.caption(f"Embedding: {cfg.rag_embedding_model}")
    st.divider()
    st.markdown("**说明**")
    st.caption(
        "Agent 会根据问题自主决定是否调用地理编码、POI、路网、街景和政策 RAG 工具。"
        "如果地点不清楚，它应该先追问，而不是盲目分析。"
    )


st.markdown('<p class="app-title">UrbanRenewal Planner Autonomous Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">面向大众与规划专业人员的杨浦区城市更新自主分析 Agent</p>',
    unsafe_allow_html=True,
)

examples = [
    "你能干什么，详细介绍一下你自己",
    "请分析鞍山新村周边800米的老年友好问题",
    "控江路和本溪路路口有哪些步行环境问题？",
    "杨浦区平凉路社区15分钟生活圈设施是否完善？",
]

if not st.session_state.messages:
    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            if st.button(example, key=f"example_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("task_plan"):
            _render_task_plan(msg["task_plan"])
        if msg["role"] == "assistant" and msg.get("report"):
            _render_report(msg["report"])
        if msg["role"] == "assistant" and msg.get("tool_events"):
            with st.expander("工具调用轨迹", expanded=False):
                _render_tool_events(msg["tool_events"])


question = st.chat_input("请输入你的城市更新问题，例如：鞍山新村周边适老化设施是否完善？")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.rerun()


if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    current_question = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        status = st.status("自主 Agent 正在规划任务并选择工具...", expanded=True)
        try:
            result = run_autonomous(current_question, thread_id=st.session_state.thread_id)
            status.update(label="分析完成", state="complete", expanded=False)
            st.markdown(result.answer)
            if result.task_plan:
                _render_task_plan(result.task_plan)
            if result.report:
                _render_report(result.report)
            if result.tool_events:
                with st.expander("工具调用轨迹", expanded=True):
                    _render_tool_events(result.tool_events)
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "tool_events": result.tool_events,
                "report": result.report.model_dump(mode="json") if result.report else None,
                "task_plan": result.task_plan.model_dump(mode="json") if result.task_plan else None,
            })
            st.session_state.last_tool_events = result.tool_events
        except Exception as exc:
            status.update(label="分析失败", state="error", expanded=True)
            error_text = f"Agent 运行失败：{exc}"
            st.error(error_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_text,
                "tool_events": [],
            })
    st.rerun()
