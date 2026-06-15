"""
UrbanRenewal Planner 自主 Agent 主入口。

用法：
    python main.py                          # 交互模式
    python main.py "请分析鞍山新村周边800米的老年友好问题"  # 单次查询
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from src.urbanrenewal.agent.autonomous import run_autonomous  # noqa: E402
from src.urbanrenewal.config import cfg  # noqa: E402


def main() -> None:
    print(f"UrbanRenewal Planner Autonomous Agent — {cfg.city}{cfg.district}")
    print("输入问题开始分析，输入 'q' 退出\n")
    thread_id = "main-cli"

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"问题：{question}\n")
        result = run_autonomous(question, thread_id=thread_id)
        print(result.answer)
        return

    while True:
        try:
            question = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if question.lower() in ("q", "quit", "exit", ""):
            break
        print()
        result = run_autonomous(question, thread_id=thread_id)
        print(result.answer)
        print()


if __name__ == "__main__":
    main()
