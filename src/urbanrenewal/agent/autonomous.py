"""
Autonomous ReAct Agent for UrbanRenewal Planner.

This is the main product-facing agent. It uses a ReAct/tool-calling loop to
choose tools, ask clarifying questions, and maintain conversation state through
a LangGraph checkpointer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ast
import json
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from src.urbanrenewal.agent.budget import AgentBudget, budget_for_tier
from src.urbanrenewal.agent.checkpoint import default_checkpointer
from src.urbanrenewal.agent.llm import build_llm
from src.urbanrenewal.agent.plan import TaskPlan, plan_task
from src.urbanrenewal.agent.report import PlanningReport, build_planning_report
from src.urbanrenewal.config import cfg
from src.urbanrenewal.tools.registry import ALL_TOOLS, TOOL_NAMES

SYSTEM_PROMPT = f"""你是 UrbanRenewal Planner，一个面向公众和专业人员的上海市杨浦区城市更新自主 Agent。

你必须遵守以下规则：
1. 当前数据范围仅支持{cfg.city}{cfg.district}。如果用户问题明显超出范围，请解释限制，并给出可支持的替代问题。
2. 不要编造空间数据、设施数量、坐标或政策依据；这些内容必须来自工具返回结果。
3. 用户未提供地点时，先追问地点；不要直接调用空间工具。
4. 用户提供模糊地点或多个地点时，优先用 geocode_tool 确认，再让用户确认不确定项。
5. 根据问题自主选择工具，不需要每次调用所有工具。
6. 高成本街景分析只在用户明确要求街景、图像、街道可见环境，或步行/适老化问题确有必要时调用。
7. 政策依据应通过 policy_rag_tool 检索。
8. 完成必要工具调用后，输出结构化 Markdown；必要时可调用 report_generation_tool 生成报告。

建议的工具选择：
- 地点定位：geocode_tool
- 设施配置、15分钟生活圈、养老服务：poi_query_tool + facility_gap_tool
- 步行可达、慢行、路口：isochrone_tool + intersection_tool
- 街景、可见环境、人行道、无障碍细节：streetview_tool
- 政策依据：policy_rag_tool

最终回答应包含：总体诊断、主要问题、证据、政策依据、更新建议、优先级和不确定性说明。
"""

DEFAULT_RECURSION_LIMIT = 18


@dataclass
class AutonomousRunResult:
    """Serializable wrapper returned by convenience runner functions."""

    session_id: str
    answer: str
    messages: list[BaseMessage] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    report: PlanningReport | None = None
    budget: AgentBudget | None = None
    task_plan: TaskPlan | None = None
    raw_state: dict[str, Any] = field(default_factory=dict)


def build_autonomous_agent(checkpointer: Any | None = None) -> Any:
    """Build the LangGraph autonomous ReAct agent."""
    llm = build_llm(temperature=0.2)
    try:
        return create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer or default_checkpointer(),
        )
    except TypeError:
        # Compatibility with older LangGraph releases that used state_modifier.
        return create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            state_modifier=SYSTEM_PROMPT,
            checkpointer=checkpointer or default_checkpointer(),
        )


def _last_ai_content(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str):
            return message.content
    if messages:
        content = messages[-1].content
        return content if isinstance(content, str) else str(content)
    return ""


def _parse_tool_content(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str) or not content.strip():
        return None
    text = content.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_tool_events(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Extract a compact, UI-friendly tool trace from LangChain messages."""
    events: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            events.append({
                "tool_name": call.get("name", "unknown"),
                "status": "started",
                "summary": "",
                "args": call.get("args", {}),
            })
        if isinstance(message, ToolMessage):
            content = message.content if isinstance(message.content, str) else str(message.content)
            parsed = _parse_tool_content(message.content)
            summary = parsed.get("summary", "") if parsed else content[:500]
            events.append({
                "tool_name": getattr(message, "name", "unknown") or "unknown",
                "status": "completed",
                "summary": summary[:500],
                "args": {},
                "result": parsed,
            })
    return events


def _compose_planned_question(question: str, task_plan: TaskPlan, conversation_context: str = "") -> str:
    context_block = ""
    if conversation_context.strip():
        context_block = (
            "Persistent API conversation context. Use it only to resolve references "
            "such as previous place, radius, scenario, and user preferences. "
            "Do not treat prior conclusions as fresh evidence unless tools verify them.\n"
            f"{conversation_context.strip()}\n\n"
        )
    return (
        f"{context_block}"
        f"User question: {question}\n\n"
        f"Deterministic pre-plan: {task_plan.model_dump_json(ensure_ascii=False)}\n\n"
        "Use the pre-plan as guidance, but make autonomous decisions from the "
        "actual context and tool observations."
    )


def run_autonomous(
    question: str,
    *,
    thread_id: str | None = None,
    recursion_limit: int | None = None,
    budget: AgentBudget | None = None,
    conversation_context: str = "",
) -> AutonomousRunResult:
    """Run the autonomous agent once and return the final answer."""
    session_id = thread_id or str(uuid4())
    effective_budget = budget or budget_for_tier("registered")
    effective_recursion_limit = recursion_limit or effective_budget.recursion_limit or DEFAULT_RECURSION_LIMIT
    task_plan = plan_task(question)
    can_use_persistent_context = bool(conversation_context.strip()) and task_plan.clarification.reason == "missing_place"
    if task_plan.clarification.needed and not can_use_persistent_context:
        answer = task_plan.clarification.question
        return AutonomousRunResult(
            session_id=session_id,
            answer=answer,
            messages=[HumanMessage(content=question), AIMessage(content=answer)],
            tool_events=[],
            report=build_planning_report(answer, []),
            budget=effective_budget,
            task_plan=task_plan,
            raw_state={"messages": []},
        )

    agent = build_autonomous_agent()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": effective_recursion_limit,
    }
    planned_question = (
        f"用户问题：{question}\n\n"
        f"系统预规划：{task_plan.model_dump_json(ensure_ascii=False)}\n\n"
        "请参考预规划，但仍需根据实际上下文和工具结果自主判断。"
    )
    if conversation_context.strip():
        planned_question = _compose_planned_question(question, task_plan, conversation_context)
    state = agent.invoke({"messages": [HumanMessage(content=planned_question)]}, config=config)
    messages = list(state.get("messages", []))
    answer = _last_ai_content(messages)
    tool_events = extract_tool_events(messages)
    return AutonomousRunResult(
        session_id=session_id,
        answer=answer,
        messages=messages,
        tool_events=tool_events,
        report=build_planning_report(answer, tool_events),
        budget=effective_budget,
        task_plan=task_plan,
        raw_state=state,
    )


def stream_autonomous(
    question: str,
    *,
    thread_id: str | None = None,
    recursion_limit: int | None = None,
    budget: AgentBudget | None = None,
    conversation_context: str = "",
):
    """Yield LangGraph stream updates for API and CLI integrations."""
    session_id = thread_id or str(uuid4())
    effective_budget = budget or budget_for_tier("registered")
    effective_recursion_limit = recursion_limit or effective_budget.recursion_limit or DEFAULT_RECURSION_LIMIT
    task_plan = plan_task(question)
    can_use_persistent_context = bool(conversation_context.strip()) and task_plan.clarification.reason == "missing_place"
    if task_plan.clarification.needed and not can_use_persistent_context:
        yield {
            "session_id": session_id,
            "event": {
                "type": "clarification",
                "task_plan": task_plan.model_dump(mode="json"),
                "message": task_plan.clarification.question,
            },
        }
        return

    agent = build_autonomous_agent()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": effective_recursion_limit,
    }
    planned_question = (
        f"用户问题：{question}\n\n"
        f"系统预规划：{task_plan.model_dump_json(ensure_ascii=False)}\n\n"
        "请参考预规划，但仍需根据实际上下文和工具结果自主判断。"
    )
    if conversation_context.strip():
        planned_question = _compose_planned_question(question, task_plan, conversation_context)
    for event in agent.stream({"messages": [HumanMessage(content=planned_question)]}, config=config, stream_mode="updates"):
        yield {"session_id": session_id, "event": event}


__all__ = [
    "ALL_TOOLS",
    "TOOL_NAMES",
    "SYSTEM_PROMPT",
    "AutonomousRunResult",
    "build_autonomous_agent",
    "run_autonomous",
    "stream_autonomous",
]
