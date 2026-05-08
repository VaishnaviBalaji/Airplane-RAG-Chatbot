"""Integration tests for api/main.py — no FAISS index or OpenAI key required."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import api.main as api_mod
from api.main import app
from rag.pipeline import PipelineError

VALID_KEY = "test-secret-key"
FAKE_ANSWER = {
    "question": "What is the carry-on limit?",
    "answer": "7 kg for Economy. [POL-BAG-001]",
    "sources": [{"doc_id": "POL-BAG-001", "title": "Carry-On Baggage Policy"}],
}


@pytest.fixture
def client(monkeypatch):
    """TestClient with pipeline loading mocked and a test API key set."""
    monkeypatch.setattr(api_mod, "_API_KEY", VALID_KEY)
    with patch("api.main._load_pipeline"):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def auth(client):
    """(client, headers) pair for authenticated requests."""
    return client, {"X-API-Key": VALID_KEY}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /chat — authentication
# ---------------------------------------------------------------------------

def test_chat_missing_api_key_returns_403(client):
    resp = client.post("/chat", json={"question": "What is the baggage limit?"})
    assert resp.status_code == 403


def test_chat_wrong_api_key_returns_401(client):
    resp = client.post(
        "/chat",
        json={"question": "What is the baggage limit?"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /chat — input validation
# ---------------------------------------------------------------------------

def test_chat_empty_question_returns_422(auth):
    client, headers = auth
    resp = client.post("/chat", json={"question": ""}, headers=headers)
    assert resp.status_code == 422


def test_chat_whitespace_only_question_returns_422(auth):
    client, headers = auth
    resp = client.post("/chat", json={"question": "   "}, headers=headers)
    assert resp.status_code == 422


def test_chat_too_long_question_returns_422(auth):
    client, headers = auth
    resp = client.post("/chat", json={"question": "x" * 501}, headers=headers)
    assert resp.status_code == 422


def test_chat_exactly_500_chars_is_accepted(auth):
    client, headers = auth
    with patch("api.main.answer_question", return_value=FAKE_ANSWER):
        resp = client.post("/chat", json={"question": "x" * 500}, headers=headers)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /chat — success path
# ---------------------------------------------------------------------------

def test_chat_returns_answer(auth):
    client, headers = auth
    with patch("api.main.answer_question", return_value=FAKE_ANSWER):
        resp = client.post(
            "/chat",
            json={"question": "What is the carry-on limit?"},
            headers=headers,
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == FAKE_ANSWER["answer"]
    assert body["sources"][0]["doc_id"] == "POL-BAG-001"


# ---------------------------------------------------------------------------
# /chat — error handling
# ---------------------------------------------------------------------------

def test_chat_pipeline_error_returns_500(auth):
    client, headers = auth
    with patch("api.main.answer_question", side_effect=PipelineError("index down")):
        resp = client.post(
            "/chat",
            json={"question": "What is the baggage limit?"},
            headers=headers,
        )
    assert resp.status_code == 500
    assert "try again" in resp.json()["detail"].lower()
