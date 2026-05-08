import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI

from rag.retriever import HybridRetriever

load_dotenv()

logger = logging.getLogger(__name__)

INDEX_PATH = Path(__file__).parent.parent / "index" / "faiss_skyline_v1"

SYSTEM_PROMPT = """You are a customer support assistant for SkyLine Airways.

Answer the customer's question using ONLY the policy context provided below.
Follow these rules strictly:

1. If the answer is in the context, answer concisely and cite the source doc_id \
in square brackets, e.g. [POL-BAG-001].
2. If the context does not contain the answer, say: "I don't have that information \
in our policies. Please contact SkyLine Airways customer services for assistance." \
Do not guess or use outside knowledge.
3. If the question is partially answered, answer what you can from the context \
and explicitly note what is missing.
4. Use a polite, professional tone. Do not invent policies, prices, or numbers.

Context:
{context}"""

CONTEXTUALIZE_SYSTEM = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question that can be "
    "understood without the chat history. Do NOT answer — just reformulate if "
    "needed, otherwise return it as is."
)

# Module-level singletons — loaded once, reused across requests
_vectorstore = None
_retriever = None      # hybrid BM25 + FAISS + cross-encoder
_single_chain = None   # single-turn chain (FastAPI)
_conv_chain = None     # multi-turn chain (Chainlit)


class PipelineError(RuntimeError):
    """Raised when the RAG pipeline fails during loading or inference."""


def _load_pipeline():
    global _vectorstore, _retriever, _single_chain, _conv_chain
    if _conv_chain is not None:
        return

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise PipelineError("OPENAI_API_KEY is not set. Add it to your .env file.")

        logger.info("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if not INDEX_PATH.exists():
            raise PipelineError(
                f"FAISS index not found at {INDEX_PATH}. Run the indexing notebook first."
            )

        logger.info("Loading FAISS index from %s", INDEX_PATH)
        _vectorstore = FAISS.load_local(
            str(INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        chunks = list(_vectorstore.docstore._dict.values())
        logger.info("Loaded %d chunks from FAISS index.", len(chunks))

        logger.info("Building hybrid retriever (may download cross-encoder on first run ~80MB)...")
        _retriever = HybridRetriever(
            vectorstore=_vectorstore,
            chunks=chunks,
            top_k=10,
            final_k=4,
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            timeout=30,      # give up if OpenAI hasn't responded in 30 s
            max_retries=2,   # retry transient errors (rate limits, 5xx) automatically
        )

        # --- Single-turn chain (FastAPI) ---
        single_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        _single_chain = single_prompt | llm | StrOutputParser()

        # --- Multi-turn chain (Chainlit) ---
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm, _retriever, contextualize_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        doc_prompt = PromptTemplate.from_template("[{doc_id}] {title}\n{page_content}")
        qa_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=doc_prompt)

        _conv_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
        logger.info("Pipeline ready.")

    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(f"Failed to initialize pipeline: {e}") from e


def _dedup_sources(docs) -> list[dict]:
    seen = {}
    for d in docs:
        doc_id = d.metadata["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "title": d.metadata["title"]}
    return list(seen.values())


def answer_question(question: str, k: int = 4) -> dict:
    """Single-turn — used by the FastAPI endpoint."""
    _load_pipeline()
    try:
        retrieved = _retriever.invoke(question)
        context = "\n\n".join(
            f"[{d.metadata['doc_id']}] {d.metadata['title']}\n{d.page_content}"
            for d in retrieved
        )
        answer = _single_chain.invoke({"context": context, "question": question})
        return {
            "question": question,
            "answer": answer,
            "sources": _dedup_sources(retrieved),
        }
    except PipelineError:
        raise
    except Exception as e:
        logger.exception("Error in answer_question for %r", question[:80])
        raise PipelineError(f"Failed to answer question: {e}") from e


def answer_with_history(question: str, chat_history: list[tuple[str, str]]) -> dict:
    """Multi-turn — chat_history is a list of (human_msg, ai_msg) tuples."""
    _load_pipeline()
    try:
        messages = []
        for human, ai in chat_history:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))

        result = _conv_chain.invoke({
            "input": question,
            "chat_history": messages,
        })

        return {
            "question": question,
            "answer": result["answer"],
            "sources": _dedup_sources(result["context"]),
        }
    except PipelineError:
        raise
    except Exception as e:
        logger.exception("Error in answer_with_history for %r", question[:80])
        raise PipelineError(f"Failed to answer question: {e}") from e
