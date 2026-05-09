#!/usr/bin/env python3
"""Rebuild the FAISS index from docs/ — run this before run_eval.py in CI.

Usage: python tests/eval/build_index.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = ROOT / "docs"
INDEX_PATH = ROOT / "index" / "faiss_skyline_v1"


def load_docs(data_dir: Path) -> list[Document]:
    docs = []
    for md_path in sorted(data_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
        doc_id_match = re.search(r"\*\*Document ID:\*\* (\S+)", text)
        docs.append(Document(
            page_content=text,
            metadata={
                "source": md_path.name,
                "title": title_match.group(1) if title_match else md_path.stem,
                "doc_id": doc_id_match.group(1) if doc_id_match else md_path.stem,
            },
        ))
    return docs


def main():
    print(f"Loading docs from {DOCS_DIR}...")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    ).split_documents(load_docs(DOCS_DIR))
    print(f"Split into {len(chunks)} chunks.")

    print("Loading embedding model (downloads ~90MB on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(INDEX_PATH))
    print(f"Index saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()
