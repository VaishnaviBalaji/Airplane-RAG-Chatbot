# SkyLine Airways RAG Chatbot

A customer-support chatbot for a fictional airline, built as a portfolio project to demonstrate end-to-end RAG (Retrieval Augmented Generation) — from ingestion through to evaluation and deployment.

## Why this project

Most RAG tutorials stop at "I built a demo." This project goes further: each stage is independently shippable, each design choice is justified by trade-offs, and the final pipeline is evaluated with concrete metrics rather than vibes.

## Stages

- **Stage 1 — Vanilla RAG (this commit)**: ingestion, chunking, embedding, FAISS retrieval, grounded generation with `gpt-4o-mini`. Single-turn, notebook-based.
- **Stage 2 — API + UI**: FastAPI backend, Chainlit frontend.
- **Stage 3 — Conversational**: session memory, query rewriting for follow-ups.
- **Stage 4 — Citations**: structured citation output, rendered as source links.
- **Stage 5 — Evaluation**: RAGAS harness, synthetic eval set, faithfulness / answer relevance / context relevance metrics.
- **Stage 6 — Advanced retrieval**: cross-encoder re-ranking, before/after measured against the eval set.
- **Stage 7 — Deployment**: Docker + Cloud Run.

## Project structure

```
rag-chatbot/
├── data/                # 10 synthetic SkyLine Airways policy docs (markdown)
├── notebooks/
│   └── 01_stage1_rag_pipeline.ipynb
├── index/               # Persisted FAISS index (created on first run)
├── src/                 # Stage 2+ code lives here
└── .env                 # OPENAI_API_KEY (not committed)
```

## Stack

- **Orchestration:** LangChain
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **Vector store:** FAISS (Stage 1) → ChromaDB (Stage 6+)
- **LLM:** OpenAI `gpt-4o-mini`
- **Eval:** RAGAS (Stage 5)
- **API:** FastAPI (Stage 2)
- **UI:** Chainlit (Stage 2)
- **Deploy:** Docker + Cloud Run (Stage 7)

## Running Stage 1

1. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=sk-...
   ```
2. Open `notebooks/01_stage1_rag_pipeline.ipynb`.
3. Uncomment the `pip install` line in the setup cell on first run.
4. Run all cells.

The first run will download the embedding model (~80MB) and build the FAISS index. Subsequent runs load from disk.

## Knowledge base

Synthetic SkyLine Airways policy documents covering:

- Carry-on and checked baggage
- Booking changes and cancellations
- Check-in and boarding
- Flight delays and disruptions (UK261 / EU261)
- Special assistance
- Pet travel
- SkyMiles loyalty programme
- Refunds
- Sports equipment

All content is fictional and was generated for this project.
