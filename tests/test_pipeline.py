"""Unit tests for rag/pipeline.py — no FAISS index or OpenAI key required."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import rag.pipeline as pipeline_mod
from rag.pipeline import PipelineError, _dedup_sources, _load_pipeline


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons before and after every test."""
    pipeline_mod._vectorstore = None
    pipeline_mod._retriever = None
    pipeline_mod._single_chain = None
    pipeline_mod._conv_chain = None
    yield
    pipeline_mod._vectorstore = None
    pipeline_mod._retriever = None
    pipeline_mod._single_chain = None
    pipeline_mod._conv_chain = None


# ---------------------------------------------------------------------------
# _dedup_sources
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str, title: str):
    doc = MagicMock()
    doc.metadata = {"doc_id": doc_id, "title": title}
    return doc


def test_dedup_sources_empty():
    assert _dedup_sources([]) == []


def test_dedup_sources_single():
    result = _dedup_sources([_make_doc("A", "Doc A")])
    assert result == [{"doc_id": "A", "title": "Doc A"}]


def test_dedup_sources_removes_duplicate_doc_ids():
    docs = [
        _make_doc("A", "Doc A"),
        _make_doc("A", "Doc A"),   # duplicate
        _make_doc("B", "Doc B"),
    ]
    result = _dedup_sources(docs)
    assert len(result) == 2
    assert result[0]["doc_id"] == "A"
    assert result[1]["doc_id"] == "B"


def test_dedup_sources_preserves_first_occurrence():
    docs = [
        _make_doc("X", "First"),
        _make_doc("X", "Second"),  # same id, different title — first wins
    ]
    result = _dedup_sources(docs)
    assert result == [{"doc_id": "X", "title": "First"}]


# ---------------------------------------------------------------------------
# _load_pipeline — startup validation
# ---------------------------------------------------------------------------

def test_load_pipeline_raises_if_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(PipelineError, match="OPENAI_API_KEY"):
        _load_pipeline()


def test_load_pipeline_raises_if_index_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(pipeline_mod, "INDEX_PATH", tmp_path / "nonexistent")
    with pytest.raises(PipelineError, match="FAISS index not found"):
        _load_pipeline()


def test_load_pipeline_skips_if_already_loaded():
    # Simulate a pipeline that is already loaded
    pipeline_mod._conv_chain = MagicMock()
    # Should return immediately without touching env vars or disk
    _load_pipeline()   # no exception = pass


# ---------------------------------------------------------------------------
# answer_question / answer_with_history — error wrapping
# ---------------------------------------------------------------------------

def test_answer_question_wraps_unexpected_exception():
    """Any crash inside the chain should become a PipelineError."""
    pipeline_mod._conv_chain = MagicMock()         # mark pipeline as loaded
    pipeline_mod._retriever = MagicMock()
    pipeline_mod._retriever.invoke.side_effect = RuntimeError("boom")
    pipeline_mod._single_chain = MagicMock()

    with pytest.raises(PipelineError, match="Failed to answer question"):
        from rag.pipeline import answer_question
        answer_question("test question")


def test_answer_with_history_wraps_unexpected_exception():
    pipeline_mod._conv_chain = MagicMock()
    pipeline_mod._conv_chain.invoke.side_effect = RuntimeError("kaboom")

    with pytest.raises(PipelineError, match="Failed to answer question"):
        from rag.pipeline import answer_with_history
        answer_with_history("test", [])
