"""
CLI 入口：构建政策 RAG 向量库。

用法：
    python scripts/build_rag.py            # 增量更新（已有chunk跳过）
    python scripts/build_rag.py --rebuild  # 清空重建
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.urbanrenewal.rag.build_policy_rag import PolicyRAGBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建政策 PDF RAG 向量库")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="清空已有向量库，强制重新构建",
    )
    args = parser.parse_args()

    builder = PolicyRAGBuilder()
    builder.build(force_rebuild=args.rebuild)


if __name__ == "__main__":
    main()
