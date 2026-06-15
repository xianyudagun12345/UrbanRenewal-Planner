from src.urbanrenewal.agent.report import PlanningReport
from src.urbanrenewal.agent.plan import plan_task
from src.urbanrenewal.api.schemas import AsyncChatResponse, ChatRequest, ChatResponse, FeedbackRequest, TaskStatusResponse


def test_chat_request_defaults_session_id():
    req = ChatRequest(question="请分析鞍山新村周边800米的老年友好问题")
    assert req.session_id
    assert req.audience == "professional"
    assert req.user_tier == "anonymous"


def test_feedback_rating_bounds():
    req = FeedbackRequest(session_id="demo", rating=5, comment="ok")
    assert req.rating == 5


def test_chat_response_accepts_structured_report():
    resp = ChatResponse(
        session_id="demo",
        answer="ok",
        duration_ms=1,
        report=PlanningReport(answer_markdown="ok"),
    )
    assert resp.report.answer_markdown == "ok"
    assert resp.report.issues == []


def test_chat_response_accepts_budget():
    from src.urbanrenewal.agent.budget import budget_for_tier

    resp = ChatResponse(
        session_id="demo",
        answer="ok",
        duration_ms=1,
        budget=budget_for_tier("anonymous"),
    )
    assert resp.budget.user_tier == "anonymous"


def test_chat_response_accepts_task_plan():
    resp = ChatResponse(
        session_id="demo",
        answer="ok",
        duration_ms=1,
        task_plan=plan_task("请分析鞍山新村周边800米的老年友好问题"),
    )
    assert resp.task_plan.scenario == "elderly_friendly"


def test_async_task_schemas():
    submitted = AsyncChatResponse(task_id="task-1", session_id="demo", status="queued", status_url="/tasks/task-1")
    status = TaskStatusResponse(task_id="task-1", session_id="demo", status="queued")

    assert submitted.status == "queued"
    assert status.result is None
