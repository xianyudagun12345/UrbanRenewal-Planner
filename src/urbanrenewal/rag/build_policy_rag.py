"""
政策 PDF RAG 构建模块。

功能：
1. 扫描 policy_raw_dir 下的所有 PDF 文件
2. 使用 pypdf 提取文本
3. 按 chunk_size / chunk_overlap 分块，保留文档来源 metadata
4. 用阿里云 DashScope text-embedding-v4 生成向量
5. 存入本地 ChromaDB 持久化向量库

调用方式：
    from src.urbanrenewal.rag.build_policy_rag import PolicyRAGBuilder
    builder = PolicyRAGBuilder()
    builder.build(force_rebuild=False)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Generator

import chromadb
from openai import OpenAI
from pypdf import PdfReader

from src.urbanrenewal.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 文本分块
# ---------------------------------------------------------------------------

def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按字符数滑动窗口分块，在句号/换行处对齐切分边界。"""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for punct in ("。", "；", "\n\n", "\n", "，"):
                idx = text.rfind(punct, start + chunk_size // 2, end)
                if idx != -1:
                    end = idx + 1
                    break
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
    return chunks


# ---------------------------------------------------------------------------
# PDF 文本提取
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: Path) -> str:
    """提取 PDF 全文，跳过无法读取的页。加密 PDF 尝试空密码解密。"""
    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        if not reader.decrypt(""):
            raise ValueError(f"{pdf_path.name} 已加密且无法用空密码解密，请手动解密后重试")
    pages: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
            pages.append(t)
        except Exception:
            continue
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Embedding（DashScope OpenAI-compatible）
# ---------------------------------------------------------------------------

class _EmbeddingClient:
    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=cfg.dashscope_api_key,
            base_url=cfg.dashscope_base_url,
        )
        self._model = cfg.rag_embedding_model

    def embed_batch(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """批量生成嵌入向量，带简单限速重试。"""
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(3):
                try:
                    resp = self._client.embeddings.create(
                        model=self._model,
                        input=batch,
                        encoding_format="float",
                    )
                    all_vectors.extend([d.embedding for d in resp.data])
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning("Embedding 请求失败（%s），1秒后重试…", e)
                    time.sleep(1)
            if i + batch_size < len(texts):
                time.sleep(0.3)
        return all_vectors


# ---------------------------------------------------------------------------
# 主构建类
# ---------------------------------------------------------------------------

class PolicyRAGBuilder:
    def __init__(self) -> None:
        self._raw_dir = cfg.policy_raw_dir
        self._processed_dir = cfg.policy_processed_dir
        self._chunks_path = cfg.policy_chunks_path
        self._vector_db_dir = cfg.policy_vector_db_dir
        self._collection_name = cfg.rag_collection_name
        self._chunk_size = cfg.rag_chunk_size
        self._overlap = cfg.rag_chunk_overlap

    def build(self, force_rebuild: bool = False) -> None:
        """完整构建流程：解析 → 分块 → 向量化 → 存库。"""
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        self._vector_db_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(self._raw_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"在 {self._raw_dir} 未找到任何 PDF 文件")
        logger.info("发现 %d 个 PDF 文件", len(pdf_files))

        all_chunks: list[dict] = []
        doc_records: list[dict] = []
        for chunk in self._parse_and_chunk(pdf_files, doc_records):
            all_chunks.append(chunk)
        logger.info("共生成 %d 个文本块", len(all_chunks))
        self._save_chunks(all_chunks)
        self._save_documents(doc_records)

        self._build_chroma(all_chunks, force_rebuild=force_rebuild)
        logger.info("政策 RAG 向量库构建完成 → %s", self._vector_db_dir)

    def _parse_and_chunk(
        self, pdf_files: list[Path], doc_records: list[dict]
    ) -> Generator[dict, None, None]:
        for pdf_path in pdf_files:
            logger.info("正在解析：%s", pdf_path.name)
            try:
                full_text = _extract_pdf_text(pdf_path)
            except Exception as e:
                logger.warning("解析失败，跳过 %s：%s", pdf_path.name, e)
                doc_records.append({
                    "doc_name": pdf_path.stem,
                    "source_file": pdf_path.name,
                    "file_size_kb": round(pdf_path.stat().st_size / 1024, 1),
                    "chunk_count": 0,
                    "parse_status": "failed",
                    "parse_error": str(e),
                })
                continue

            chunks = _split_text(full_text, self._chunk_size, self._overlap)
            doc_name = pdf_path.stem
            doc_records.append({
                "doc_name": doc_name,
                "source_file": pdf_path.name,
                "file_size_kb": round(pdf_path.stat().st_size / 1024, 1),
                "chunk_count": len(chunks),
                "parse_status": "ok",
                "parse_error": None,
            })
            for idx, chunk_text in enumerate(chunks):
                yield {
                    "doc_name": doc_name,
                    "chunk_index": idx,
                    "chunk_id": f"{doc_name}__chunk_{idx:04d}",
                    "text": chunk_text,
                    "source_file": pdf_path.name,
                }

    def _save_chunks(self, chunks: list[dict]) -> None:
        with open(self._chunks_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logger.info("分块结果已保存 → %s", self._chunks_path)

    def _save_documents(self, doc_records: list[dict]) -> None:
        docs_path = cfg.policy_documents_path
        with open(docs_path, "w", encoding="utf-8") as f:
            for rec in doc_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("文档元信息已保存 → %s", docs_path)

    def _build_chroma(self, chunks: list[dict], force_rebuild: bool) -> None:
        client = chromadb.PersistentClient(path=str(self._vector_db_dir))

        if force_rebuild:
            try:
                client.delete_collection(self._collection_name)
                logger.info("已删除旧向量库 collection：%s", self._collection_name)
            except Exception:
                pass

        collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        existing_ids: set[str] = set()
        try:
            existing_ids = set(collection.get(include=[])["ids"])
        except Exception:
            pass

        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if not new_chunks:
            logger.info("所有文本块已存在于向量库，无需更新")
            return

        logger.info("需向量化的新文本块：%d 个", len(new_chunks))
        embedder = _EmbeddingClient()
        texts = [c["text"] for c in new_chunks]
        vectors = embedder.embed_batch(texts)

        collection.add(
            ids=[c["chunk_id"] for c in new_chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "doc_name": c["doc_name"],
                    "chunk_index": c["chunk_index"],
                    "source_file": c["source_file"],
                }
                for c in new_chunks
            ],
        )
        logger.info("已写入 %d 个向量到 ChromaDB", len(new_chunks))
