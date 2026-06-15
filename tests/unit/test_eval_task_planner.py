from pathlib import Path

from scripts.eval_task_planner import evaluate


def test_task_planner_golden_questions_pass():
    report = evaluate(Path("eval/golden_questions.json"))

    assert report["passed"] == report["total"]
    assert report["accuracy"] == 1.0
