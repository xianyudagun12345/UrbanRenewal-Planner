"""
政策 RAG 检索测试脚本。

用法：
    python scripts/test_rag.py
    python scripts/test_rag.py --query "无障碍坡道设计标准"
    python scripts/test_rag.py --top_k 5

前提：已运行 scripts/build_rag.py 完成向量库构建。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb

from src.urbanrenewal.config import cfg
from src.urbanrenewal.rag.build_policy_rag import _EmbeddingClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "老年人步行可达性与无障碍设施",
    "15分钟社区生活圈公共服务设施配置标准",
    "慢行交通街道断面设计要求",
    "城市更新微改造实施路径",
    "养老服务设施建设要求",
]


def search(query: str, top_k: int) -> list[dict]:
    client = chromadb.PersistentClient(path=str(cfg.policy_vector_db_dir))
    collection = client.get_collection(cfg.rag_collection_name)

    embedder = _EmbeddingClient()
    q_vec = embedder.embed_batch([query])[0]

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "score": round(1 - dist, 4),
                "doc_name": meta["doc_name"],
                "chunk_index": meta["chunk_index"],
                "text_preview": doc[:120].replace("\n", " "),
            }
        )
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="政策 RAG 检索测试")
    parser.add_argument("--query", type=str, default=None, help="自定义检索问题")
    parser.add_argument("--top_k", type=int, default=cfg.rag_top_k, help="返回条数")
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        t0 = time.time()
        hits = search(q, args.top_k)
        elapsed = time.time() - t0
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] score={h['score']:.4f}  来源: {h['doc_name']}  chunk#{h['chunk_index']}")
            print(f"       {h['text_preview']}…")
        print(f"  (耗时 {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
