"""
对扫描件 PDF 进行 OCR，将识别文本追加进政策 RAG 向量库。

流程：
1. 读取 policy_documents.jsonl，找出 chunk_count=0 的文件
2. 用 pymupdf 将每页渲染为图像
3. 用 rapidocr-onnxruntime 识别文字
4. 分块 → 写入 policy_chunks.jsonl（追加）→ 写入 ChromaDB

用法：
    python scripts/ocr_scanned_pdfs.py
    python scripts/ocr_scanned_pdfs.py --dpi 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import fitz  # pymupdf
import numpy as np
from rapidocr_onnxruntime import RapidOCR

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb

from src.urbanrenewal.config import cfg
from src.urbanrenewal.rag.build_policy_rag import _EmbeddingClient, _split_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def pdf_page_to_image(page: fitz.Page, dpi: int) -> np.ndarray:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img


def ocr_pdf(pdf_path: Path, dpi: int = 150) -> str:
    engine = RapidOCR()
    doc = fitz.open(str(pdf_path))
    pages_text: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        img = pdf_page_to_image(page, dpi)
        result, _ = engine(img)
        if result:
            page_text = "\n".join([line[1] for line in result])
        else:
            page_text = ""
        pages_text.append(page_text)
        logger.info("  第 %d/%d 页 → %d 字符", page_num + 1, len(doc), len(page_text))

    doc.close()
    return "\n".join(pages_text)


def main(dpi: int) -> None:
    docs_path = cfg.policy_documents_path
    if not docs_path.exists():
        raise FileNotFoundError(f"未找到 {docs_path}，请先运行 scripts/build_rag.py")

    scanned: list[dict] = []
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["chunk_count"] == 0 and rec["parse_status"] == "ok":
                scanned.append(rec)

    if not scanned:
        logger.info("没有需要 OCR 的文档，退出")
        return

    logger.info("需要 OCR 的文档：%d 份", len(scanned))

    chroma_client = chromadb.PersistentClient(path=str(cfg.policy_vector_db_dir))
    collection = chroma_client.get_or_create_collection(
        name=cfg.rag_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    existing_ids: set[str] = set(collection.get(include=[])["ids"])

    embedder = _EmbeddingClient()
    all_new_chunks: list[dict] = []

    for rec in scanned:
        pdf_path = cfg.policy_raw_dir / rec["source_file"]
        if not pdf_path.exists():
            logger.warning("文件不存在，跳过：%s", pdf_path)
            continue

        logger.info("开始 OCR：%s", rec["source_file"])
        try:
            full_text = ocr_pdf(pdf_path, dpi=dpi)
        except Exception as e:
            logger.error("OCR 失败：%s → %s", rec["source_file"], e)
            continue

        if not full_text.strip():
            logger.warning("OCR 结果为空：%s", rec["source_file"])
            continue

        chunks = _split_text(full_text, cfg.rag_chunk_size, cfg.rag_chunk_overlap)
        doc_name = rec["doc_name"]
        logger.info("  OCR 完成，生成 %d 个文本块", len(chunks))

        new_chunks = []
        for idx, text in enumerate(chunks):
            chunk_id = f"{doc_name}__chunk_{idx:04d}"
            if chunk_id in existing_ids:
                continue
            new_chunks.append({
                "chunk_id": chunk_id,
                "doc_name": doc_name,
                "chunk_index": idx,
                "source_file": rec["source_file"],
                "text": text,
            })

        if not new_chunks:
            logger.info("  所有 chunk 已存在，跳过")
            continue

        logger.info("  向量化 %d 个新文本块…", len(new_chunks))
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
        all_new_chunks.extend(new_chunks)
        logger.info("  已写入 %d 个向量", len(new_chunks))
        rec["chunk_count"] = len(chunks)
        time.sleep(0.5)

    if all_new_chunks:
        with open(cfg.policy_chunks_path, "a", encoding="utf-8") as f:
            for c in all_new_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        logger.info("已追加 %d 个 chunk 到 %s", len(all_new_chunks), cfg.policy_chunks_path)

    all_docs: list[dict] = []
    updated = {r["source_file"]: r for r in scanned}
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["source_file"] in updated:
                rec["chunk_count"] = updated[rec["source_file"]]["chunk_count"]
            all_docs.append(rec)
    with open(docs_path, "w", encoding="utf-8") as f:
        for rec in all_docs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("policy_documents.jsonl 已更新")
    logger.info("向量库现有总 chunk 数：%d", collection.count())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对扫描件 PDF 执行 OCR 并补充进 RAG 向量库")
    parser.add_argument("--dpi", type=int, default=150, help="渲染分辨率（默认150）")
    args = parser.parse_args()
    main(dpi=args.dpi)
