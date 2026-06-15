"""
Autonomous Agent 端到端测试脚本。

用法：
    python scripts/run_agent_smoke.py
    python scripts/run_agent_smoke.py --question "控江路和本溪路路口有哪些步行环境问题？"
    python scripts/run_agent_smoke.py --question "..." --thread-id demo

输出自主 Agent 的工具调用轨迹和最终规划建议。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_agent_smoke")

from src.urbanrenewal.agent.autonomous import run_autonomous, stream_autonomous  # noqa: E402
from src.urbanrenewal.config import cfg  # noqa: E402

DEFAULT_QUESTION = "请分析鞍山新村周边800米的老年友好问题"

DIVIDER = "=" * 70


def print_tool_trace(tool_events: list[dict]) -> None:
    """打印自主 Agent 工具调用轨迹。"""
    print(f"\n{DIVIDER}")
    print("  Autonomous Agent 工具调用轨迹")
    print(DIVIDER)
    if not tool_events:
        print("  未检测到工具调用；可能是澄清问题或模型直接回答。")
        return
    for idx, event in enumerate(tool_events, start=1):
        summary = (event.get("summary") or "").replace("\n", " ")[:160]
        print(f"  {idx:02d}. [{event.get('status')}] {event.get('tool_name')} {summary}")


def run_with_timing(question: str, thread_id: str, stream: bool = False) -> None:
    print(DIVIDER)
    print(f"  UrbanRenewal Planner Agent — {cfg.city}{cfg.district}")
    print(DIVIDER)
    print(f"问题：{question}\n")

    if stream:
        print("[stream events]")
        for item in stream_autonomous(question, thread_id=thread_id):
            print(item)
        return

    result = run_autonomous(question, thread_id=thread_id)
    print_tool_trace(result.tool_events)

    # 打印最终建议
    print(f"\n{DIVIDER}")
    print("  最终规划建议")
    print(DIVIDER)
    print(result.answer or "（无输出）")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 端到端测试")
    parser.add_argument("--question", "-q", type=str, default=DEFAULT_QUESTION,
                        help=f"测试问题（默认：{DEFAULT_QUESTION}）")
    parser.add_argument("--thread-id", type=str, default="test-agent",
                        help="会话 thread_id，用于测试多轮对话记忆")
    parser.add_argument("--stream", action="store_true",
                        help="打印 LangGraph 流式事件")
    args = parser.parse_args()
    run_with_timing(args.question, thread_id=args.thread_id, stream=args.stream)


if __name__ == "__main__":
    main()
