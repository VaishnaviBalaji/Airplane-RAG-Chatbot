import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag.pipeline import PipelineError, answer_question, _load_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# --- Rate limiting: 10 requests per minute per IP ---
limiter = Limiter(key_func=get_remote_address)

# --- API key auth ---
_API_KEY = os.getenv("API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def _verify_api_key(key: str = Security(_api_key_header)) -> str:
    if not _API_KEY:
        # Warn loudly but don't block — lets you develop without a key set
        logger.warning("API_KEY not configured — endpoint is unprotected")
        return key
    if key != _API_KEY:
        logger.warning("Rejected request with invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return key


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading RAG pipeline...")
    try:
        _load_pipeline()
        logger.info("Pipeline loaded successfully.")
    except PipelineError as e:
        logger.critical("Pipeline failed to load at startup: %s", e)
        raise
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SkyLine Airways RAG API",
    description="Policy Q&A powered by retrieval-augmented generation.",
    version="0.2.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class ChatRequest(BaseModel):
    question: str
    k: int = 4

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty.")
        if len(v) > 500:
            raise ValueError("Question must be 500 characters or fewer.")
        return v


class Source(BaseModel):
    doc_id: str
    title: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat(request: Request, body: ChatRequest, api_key: str = Depends(_verify_api_key)):
    start = time.perf_counter()
    logger.info("Question received (length=%d): %r", len(body.question), body.question[:80])
    try:
        result = answer_question(body.question, k=body.k)
    except PipelineError as e:
        logger.error("Pipeline error: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to answer your question. Please try again.",
        )
    elapsed = time.perf_counter() - start
    logger.info(
        "Answered in %.2fs | sources: %s",
        elapsed,
        [s["doc_id"] for s in result["sources"]],
    )
    return result
