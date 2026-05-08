import logging

import chainlit as cl

from rag.pipeline import PipelineError, answer_with_history, _load_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

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

    sources_md = "\n".join(
        f"- `{s['doc_id']}` — {s['title']}" for s in result["sources"]
    )

    await cl.Message(
        content=f"{result['answer']}\n\n---\n**Sources consulted:**\n{sources_md}"
    ).send()
