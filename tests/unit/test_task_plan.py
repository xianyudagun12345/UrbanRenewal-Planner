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

    assert plan.interaction_type == "spatial_analysis"
    assert plan.clarification.needed is True
    assert plan.clarification.reason == "missing_place"


def test_plan_task_treats_capability_question_as_help_not_spatial_task():
    plan = plan_task("你能干什么，详细介绍一下你自己")

    assert plan.interaction_type == "capability_help"
    assert plan.clarification.needed is False
    assert plan.suggested_tools == []


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


def test_run_autonomous_answers_capability_question_without_tools():
    result = run_autonomous("你能干什么，详细介绍一下你自己", thread_id="unit-help")

    assert result.task_plan is not None
    assert result.task_plan.interaction_type == "capability_help"
    assert "UrbanRenewal Planner" in result.answer
    assert "地理编码" in result.answer
    assert result.tool_events == []
