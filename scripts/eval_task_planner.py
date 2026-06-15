"""
Evaluate deterministic task planning against golden questions.

This script does not call LLMs or external APIs. It is intended as a fast
regression check whenever prompt rules, planning heuristics, or tool routing
logic changes.

Usage:
    uv run python scripts/eval_task_planner.py
    uv run python scripts/eval_task_planner.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.urbanrenewal.agent.plan import plan_task  # noqa: E402

DEFAULT_DATASET = Path(__file__).parent.parent / "eval" / "golden_questions.json"


@dataclass
class EvalCaseResult:
    case_id: str
    passed: bool
    failures: list[str]
    plan: dict[str, Any]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _check_case(case: dict[str, Any]) -> EvalCaseResult:
    plan = plan_task(case["question"])
    failures: list[str] = []

    if plan.scenario != case.get("expected_scenario"):
        failures.append(f"scenario expected={case.get('expected_scenario')} actual={plan.scenario}")
    if plan.radius_m != case.get("expected_radius_m"):
        failures.append(f"radius expected={case.get('expected_radius_m')} actual={plan.radius_m}")
    if plan.clarification.needed != case.get("expected_clarification_needed"):
        failures.append(
            "clarification_needed "
            f"expected={case.get('expected_clarification_needed')} actual={plan.clarification.needed}"
        )

    expected_reason = case.get("expected_clarification_reason")
    if expected_reason and plan.clarification.reason != expected_reason:
        failures.append(f"clarification_reason expected={expected_reason} actual={plan.clarification.reason}")

    missing_tools = sorted(set(case.get("expected_tools", [])) - set(plan.suggested_tools))
    if missing_tools:
        failures.append(f"missing_tools={missing_tools}")

    for expected_place in case.get("expected_place_contains", []):
        if not any(expected_place in place for place in plan.places):
            failures.append(f"missing_place_contains={expected_place}; actual_places={plan.places}")

    return EvalCaseResult(
        case_id=case["id"],
        passed=not failures,
        failures=failures,
        plan=plan.model_dump(mode="json"),
    )


def evaluate(path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    cases = _load_cases(path)
    results = [_check_case(case) for case in cases]
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    return {
        "dataset": str(path),
        "passed": passed,
        "total": total,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "results": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate task planner golden questions")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    args = parser.parse_args()

    report = evaluate(args.dataset)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Task planner eval: {report['passed']}/{report['total']} passed, accuracy={report['accuracy']:.2%}")
        for result in report["results"]:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"- {status} {result['case_id']}")
            for failure in result["failures"]:
                print(f"  - {failure}")

    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
