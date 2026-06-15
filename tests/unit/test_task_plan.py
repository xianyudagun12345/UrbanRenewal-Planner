from src.urbanrenewal.agent.autonomous import run_autonomous
from src.urbanrenewal.agent.plan import plan_task


def test_plan_task_detects_elderly_friendly_place_and_tools():
    plan = plan_task("请分析鞍山新村周边800米的老年友好问题")

    assert plan.scenario == "elderly_friendly"
    assert plan.radius_m == 800
    assert "geocode_tool" in plan.suggested_tools
    assert "poi_query_tool" in plan.suggested_tools
    assert plan.clarification.needed is False


def test_plan_task_clarifies_missing_place():
    plan = plan_task("帮我看看附近设施怎么样")

    assert plan.clarification.needed is True
    assert plan.clarification.reason == "missing_place"


def test_plan_task_detects_intersection():
    plan = plan_task("控江路和本溪路路口有哪些步行环境问题？")

    assert "控江路和本溪路路口" in plan.places
    assert plan.scenario == "walkability"
    assert "intersection_tool" in plan.suggested_tools


def test_run_autonomous_short_circuits_clarification_without_llm():
    result = run_autonomous("帮我看看设施怎么样", thread_id="unit-clarify")

    assert result.task_plan is not None
    assert result.task_plan.clarification.needed is True
    assert "请提供" in result.answer
    assert result.tool_events == []
