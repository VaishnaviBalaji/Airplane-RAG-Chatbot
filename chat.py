import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chainlit as cl
from dotenv import load_dotenv

from rag.pipeline import PipelineError, answer_with_history, _load_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

FEEDBACK_FILE = Path("feedback.jsonl")

_CHAT_USERNAME = os.getenv("CHAT_USERNAME", "admin")
_CHAT_PASSWORD = os.getenv("CHAT_PASSWORD")


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if not _CHAT_PASSWORD:
        # No password configured — allow through and warn (useful in local dev)
        logger.warning("CHAT_PASSWORD not set — chat UI has no authentication")
        return cl.User(identifier=username)
    if username == _CHAT_USERNAME and password == _CHAT_PASSWORD:
        logger.info("User logged in: %r", username)
        return cl.User(identifier=username, metadata={"role": "user"})
    logger.warning("Failed login attempt for username: %r", username)
    return None

_ERROR_MESSAGE = (
    "Sorry, I ran into a problem answering your question. "
    "Please try again, or contact SkyLine Airways customer services directly."
)


@cl.on_chat_start
async def on_chat_start():
    try:
        _load_pipeline()
    except PipelineError as e:
        logger.critical("Pipeline failed at chat start: %s", e)
        await cl.Message(
            content="Sorry, the assistant is currently unavailable. Please try again later."
        ).send()
        return
    cl.user_session.set("history", [])
    await cl.Message(
        content=(
            "Hello! I'm the **SkyLine Airways** support assistant. "
            "Ask me anything about our policies — baggage, check-in, pets, disruptions, and more."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    history: list[tuple[str, str]] = cl.user_session.get("history")
    logger.info("Message received (length=%d)", len(message.content))

    try:
        result = answer_with_history(message.content, history)
    except PipelineError as e:
        logger.error("Pipeline error: %s", e)
        await cl.Message(content=_ERROR_MESSAGE).send()
        return
    except Exception:
        logger.exception("Unexpected error handling message")
        await cl.Message(content=_ERROR_MESSAGE).send()
        return

    history.append((message.content, result["answer"]))
    cl.user_session.set("history", history)

    # Store this turn so the feedback callback can reference it
    cl.user_session.set("last_turn", {
        "question": message.content,
        "answer": result["answer"],
        "sources": result["sources"],
    })

    sources_md = "\n".join(
        f"- `{s['doc_id']}` — {s['title']}" for s in result["sources"]
    )

    await cl.Message(
        content=f"{result['answer']}\n\n---\n**Sources consulted:**\n{sources_md}",
        actions=[
            cl.Action(name="feedback", payload={"rating": 1}, label="👍 Helpful"),
            cl.Action(name="feedback", payload={"rating": -1}, label="👎 Not helpful"),
        ],
    ).send()


@cl.action_callback("feedback")
async def on_feedback(action: cl.Action):
    last_turn = cl.user_session.get("last_turn")
    if not last_turn:
        return

    rating = action.payload["rating"]
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "question": last_turn["question"],
        "answer": last_turn["answer"],
        "sources": last_turn["sources"],
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Feedback recorded: rating=%d for %r", rating, last_turn["question"][:60])

    emoji = "🙏" if rating == 1 else "🙏 We'll work on improving this."
    await cl.Message(content=f"Thanks for your feedback! {emoji}").send()
    await action.remove()
