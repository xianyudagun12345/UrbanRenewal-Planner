"""Lightweight task planning and clarification before the ReAct loop."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

Scenario = Literal["elderly_friendly", "life_circle", "walkability", "general"]

_CONTEXT_REF_PATTERNS = ("刚才", "上次", "前面", "那里", "那边", "那附近", "这个地方", "该区域")
_OUT_OF_SCOPE_HINTS = ("浦东", "徐汇", "静安", "黄浦", "长宁", "普陀", "虹口", "闵行", "宝山", "松江", "北京", "深圳")
_PLACE_SUFFIXES = ("新村", "小区", "社区", "路", "街", "医院", "公园", "广场", "学校", "菜场", "地铁站", "路口")


class ClarificationDecision(BaseModel):
    needed: bool = False
    reason: Literal["missing_place", "ambiguous_place", "out_of_scope", "too_broad", "none"] = "none"
    question: str = ""


class TaskPlan(BaseModel):
    question: str
    scenario: Scenario = "general"
    radius_m: int = 800
    places: list[str] = Field(default_factory=list)
    has_context_reference: bool = False
    clarification: ClarificationDecision = Field(default_factory=ClarificationDecision)
    suggested_tools: list[str] = Field(default_factory=list)
    reasoning: str = ""


def _detect_scenario(question: str) -> Scenario:
    if any(keyword in question for keyword in ("老年", "老人", "养老", "适老", "无障碍", "为老", "休息")):
        return "elderly_friendly"
    if any(keyword in question for keyword in ("15分钟", "十五分钟", "生活圈", "公共服务", "设施配置", "完整社区")):
        return "life_circle"
    if any(keyword in question for keyword in ("步行", "慢行", "人行", "过街", "路口", "街道环境", "骑行")):
        return "walkability"
    return "general"


def _detect_radius(question: str) -> int:
    match = re.search(r"(\d{2,4})\s*(?:米|m|M)", question)
    if not match:
        return 800
    value = int(match.group(1))
    return max(50, min(value, 3000))


def _extract_places(question: str) -> list[str]:
    places: list[str] = []

    intersection = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]{1,10}?路)[和与、]([\u4e00-\u9fa5A-Za-z0-9]{1,10}?路)(?:路口)?", question)
    if intersection:
        places.append(f"{intersection.group(1)}和{intersection.group(2)}路口")

    for suffix in _PLACE_SUFFIXES:
        pattern = rf"([\u4e00-\u9fa5A-Za-z0-9]{{2,18}}{suffix})"
        for match in re.finditer(pattern, question):
            candidate = match.group(1)
            candidate = re.sub(r"^(请|帮我|分析|看看|评估|一下|从|在|对|关于)", "", candidate)
            candidate = re.sub(r"(周边|附近|周围|这个|这个区域|有哪些).*$", "", candidate)
            if 2 <= len(candidate) <= 20 and candidate not in places:
                places.append(candidate)

    filtered: list[str] = []
    for place in sorted(places, key=len, reverse=True):
        if not any(place != other and place in other for other in filtered):
            filtered.append(place)
    return filtered[:4]


def _suggest_tools(question: str, scenario: Scenario, places: list[str], has_context_reference: bool) -> list[str]:
    tools: list[str] = []
    if places or has_context_reference:
        tools.append("geocode_tool")
    if scenario in {"elderly_friendly", "life_circle"} or any(keyword in question for keyword in ("设施", "养老", "医疗", "菜场", "公园")):
        tools.extend(["poi_query_tool", "facility_gap_tool"])
    if scenario == "walkability" or any(keyword in question for keyword in ("步行", "路口", "过街", "慢行", "可达")):
        tools.extend(["isochrone_tool", "intersection_tool"])
    if any(keyword in question for keyword in ("街景", "图像", "照片", "可见", "人行道", "无障碍")):
        tools.append("streetview_tool")
    if any(keyword in question for keyword in ("政策", "依据", "标准", "规范", "导则", "建议", "规划")) or scenario != "general":
        tools.append("policy_rag_tool")
    return list(dict.fromkeys(tools))


def plan_task(question: str) -> TaskPlan:
    """Create a deterministic first-pass task plan for UX, safety, and tests."""
    stripped = question.strip()
    scenario = _detect_scenario(stripped)
    radius_m = _detect_radius(stripped)
    places = _extract_places(stripped)
    has_context_reference = any(pattern in stripped for pattern in _CONTEXT_REF_PATTERNS)
    suggested_tools = _suggest_tools(stripped, scenario, places, has_context_reference)

    clarification = ClarificationDecision()
    if any(hint in stripped for hint in _OUT_OF_SCOPE_HINTS):
        clarification = ClarificationDecision(
            needed=True,
            reason="out_of_scope",
            question="当前数据主要覆盖上海市杨浦区。请确认你希望分析杨浦区内的哪个地点，或说明是否只是做方法讨论。",
        )
    elif not places and not has_context_reference:
        clarification = ClarificationDecision(
            needed=True,
            reason="missing_place",
            question="请提供要分析的具体地点、小区、道路或路口名称，例如“鞍山新村”或“控江路和本溪路路口”。",
        )
    elif len(places) > 1 and "对比" not in stripped and "分别" not in stripped:
        clarification = ClarificationDecision(
            needed=True,
            reason="ambiguous_place",
            question=f"我识别到多个地点：{'、'.join(places)}。请确认是要做对比分析，还是分别分析其中一个地点？",
        )

    reasoning_parts = [
        f"场景={scenario}",
        f"半径={radius_m}m",
        f"地点={places or '未显式识别'}",
        f"上下文指代={'是' if has_context_reference else '否'}",
        f"建议工具={suggested_tools or '暂无'}",
    ]

    return TaskPlan(
        question=stripped,
        scenario=scenario,
        radius_m=radius_m,
        places=places,
        has_context_reference=has_context_reference,
        clarification=clarification,
        suggested_tools=suggested_tools,
        reasoning="；".join(reasoning_parts),
    )
