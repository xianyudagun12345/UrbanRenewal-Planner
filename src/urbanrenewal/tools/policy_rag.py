"""
政策 RAG 查询封装。

核心功能：
- query_policy:       单条查询，返回结构化 PolicyChunk 列表
- query_policy_multi: 多条查询后合并去重，覆盖不同侧面
- format_citations:   将检索结果格式化为规划建议引用段落

设计要点：
- ChromaDB 和 EmbeddingClient 均在模块级懒加载并缓存，进程内复用
- 支持按场景自动追加关键词（augment_with_scenario），提升召回率
- 支持最低相似度阈值过滤（min_score），剔除低质量结果
- 支持按 doc_name 白名单过滤，精确限定参考文献范围
- format_citations 输出"依据《XXX》"格式，供 Agent 直接写入规划建议

使用：
    from src.urbanrenewal.tools.policy_rag import query_policy, format_citations

    chunks = query_policy("老年人步行无障碍设施配置要求", top_k=5, scenario="elderly_friendly")
    print(format_citations(chunks))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import chromadb

from src.urbanrenewal.config import cfg
from src.urbanrenewal.rag.build_policy_rag import _EmbeddingClient

logger = logging.getLogger(__name__)

# 场景关键词扩充表：查询时自动追加，提升相关政策召回
_SCENARIO_KEYWORDS: dict[str, list[str]] = {
    "elderly_friendly": ["适老化", "无障碍", "老年友好", "步行可达"],
    "life_circle": ["15分钟生活圈", "社区生活圈", "公共服务设施", "完整社区"],
    "walkability": ["慢行交通", "步行友好", "街道设计", "人行道"],
}


# ---------------------------------------------------------------------------
# 模块级懒加载缓存
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(cfg.policy_vector_db_dir))
    return client.get_collection(cfg.rag_collection_name)


@lru_cache(maxsize=1)
def _get_embedder() -> _EmbeddingClient:
    return _EmbeddingClient()


@lru_cache(maxsize=1)
def _load_doc_index() -> dict[str, str]:
    """
    加载 policy_documents.jsonl，返回 {doc_name: source_file} 映射。
    用于格式化引用时查找完整文件标题。
    """
    index: dict[str, str] = {}
    try:
        with open(cfg.policy_documents_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                index[rec["doc_name"]] = rec["source_file"]
    except FileNotFoundError:
        logger.warning("policy_documents.jsonl 不存在，文献引用将使用 doc_name 代替")
    return index


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class PolicyChunk:
    """单条政策检索结果。"""
    chunk_id: str          # "{doc_name}__chunk_{idx:04d}"
    doc_name: str          # 文档名（不含扩展名），用于引用
    chunk_index: int
    source_file: str
    text: str
    score: float           # cosine 相似度 [0, 1]，越高越相关
    query: str             # 触发此结果的查询文本

    def short_preview(self, length: int = 80) -> str:
        return self.text[:length].replace("\n", " ") + "…"


# ---------------------------------------------------------------------------
# 核心查询
# ---------------------------------------------------------------------------

def query_policy(
    query_text: str,
    top_k: Optional[int] = None,
    scenario: Optional[str] = None,
    min_score: float = 0.50,
    doc_names: Optional[list[str]] = None,
    augment_with_scenario: bool = True,
) -> list[PolicyChunk]:
    """
    查询政策 RAG，返回相关文档片段列表。

    Args:
        query_text:             查询问题或关键词
        top_k:                  返回条数，None 时使用 config 默认值
        scenario:               分析场景（"elderly_friendly"/"life_circle"/"walkability"）；
                                指定时自动在查询末尾追加场景关键词（augment_with_scenario=True）
        min_score:              最低相似度阈值，默认 0.50；低于此分值的结果被过滤
        doc_names:              白名单过滤，仅返回指定文档名中的片段；None 则不限制
        augment_with_scenario:  是否追加场景关键词到查询文本，默认 True

    Returns:
        按相似度降序排列的 PolicyChunk 列表
    """
    k = top_k or cfg.rag_top_k

    # 场景关键词扩充
    effective_query = query_text
    if augment_with_scenario and scenario and scenario in _SCENARIO_KEYWORDS:
        keywords = " ".join(_SCENARIO_KEYWORDS[scenario])
        effective_query = f"{query_text} {keywords}"

    collection = _get_collection()
    embedder = _get_embedder()
    query_vec = embedder.embed_batch([effective_query])[0]

    raw = collection.query(
        query_embeddings=[query_vec],
        n_results=min(k * 2, collection.count()),  # 多取一些，过滤后再截断
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[PolicyChunk] = []
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        score = round(1.0 - dist, 4)
        if score < min_score:
            continue

        doc_name = meta["doc_name"]
        if doc_names and doc_name not in doc_names:
            continue

        chunk_index = int(meta["chunk_index"])
        chunks.append(PolicyChunk(
            chunk_id=f"{doc_name}__chunk_{chunk_index:04d}",
            doc_name=doc_name,
            chunk_index=chunk_index,
            source_file=meta.get("source_file", ""),
            text=doc,
            score=score,
            query=query_text,
        ))

    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks[:k]


def query_policy_multi(
    queries: list[str],
    top_k_per_query: int = 4,
    scenario: Optional[str] = None,
    min_score: float = 0.50,
    dedupe: bool = True,
) -> list[PolicyChunk]:
    """
    多条查询合并去重，用于覆盖用户问题的不同侧面。

    例如对于"老年步行环境改善"可以拆分为：
      ["老年人步行无障碍设施", "慢行交通街道设计", "养老服务设施配置"]

    Args:
        queries:         查询文本列表
        top_k_per_query: 每条查询取多少条结果
        scenario:        场景名
        min_score:       最低分阈值
        dedupe:          True 时按 chunk_id 去重，保留最高分那次

    Returns:
        合并后按相似度降序排列的 PolicyChunk 列表
    """
    all_chunks: list[PolicyChunk] = []
    for q in queries:
        results = query_policy(
            q,
            top_k=top_k_per_query,
            scenario=scenario,
            min_score=min_score,
        )
        all_chunks.extend(results)

    if not dedupe:
        return sorted(all_chunks, key=lambda c: c.score, reverse=True)

    # 去重：同一 chunk_id 保留最高分
    best: dict[str, PolicyChunk] = {}
    for chunk in all_chunks:
        if chunk.chunk_id not in best or chunk.score > best[chunk.chunk_id].score:
            best[chunk.chunk_id] = chunk

    return sorted(best.values(), key=lambda c: c.score, reverse=True)


# ---------------------------------------------------------------------------
# 格式化输出
# ---------------------------------------------------------------------------

def format_citations(
    chunks: list[PolicyChunk],
    max_text_length: int = 150,
    group_by_doc: bool = True,
) -> str:
    """
    将检索结果格式化为"依据《XXX》"引用段落，供 Agent 写入规划建议。

    Args:
        chunks:          query_policy / query_policy_multi 返回的结果
        max_text_length: 每条摘录的最大字符数
        group_by_doc:    True 时按文档分组展示；False 时按分值顺序展示

    Returns:
        格式化的 Markdown 引用文本
    """
    if not chunks:
        return "（未检索到相关政策依据）"

    doc_index = _load_doc_index()

    if group_by_doc:
        # 按 doc_name 分组
        groups: dict[str, list[PolicyChunk]] = {}
        for c in chunks:
            groups.setdefault(c.doc_name, []).append(c)

        lines: list[str] = []
        for doc_name, doc_chunks in groups.items():
            # 用完整文件名（去扩展名）做标题
            title = doc_index.get(doc_name, doc_name)
            title = title.replace(".pdf", "").replace("（", "(").replace("）", ")")
            lines.append(f"**依据《{title}》**")
            for c in sorted(doc_chunks, key=lambda x: x.chunk_index):
                excerpt = c.text[:max_text_length].replace("\n", " ") + "…"
                lines.append(f"- {excerpt}")
            lines.append("")
        return "\n".join(lines).strip()

    else:
        lines = []
        for c in chunks:
            title = doc_index.get(c.doc_name, c.doc_name).replace(".pdf", "")
            excerpt = c.text[:max_text_length].replace("\n", " ") + "…"
            lines.append(f"- 【{title}】{excerpt}  _(相似度 {c.score:.3f})_")
        return "\n".join(lines)
