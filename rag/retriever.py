import numpy as np
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr


class HybridRetriever(BaseRetriever):
    """
    Retrieval pipeline:
      1. BM25 keyword search        → top_k candidates
      2. FAISS semantic search      → top_k candidates
      3. Reciprocal Rank Fusion     → unified ranked list
      4. Cross-encoder re-ranking   → final_k docs

    Why each step:
    - BM25 catches exact keyword matches (policy codes, numbers) that embeddings miss
    - FAISS catches semantic similarity that BM25 misses
    - RRF merges the two ranked lists without needing score normalisation
    - Cross-encoder scores query-doc pairs jointly (much stronger signal than bi-encoder)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Any
    chunks: list[Document]
    top_k: int = 10    # candidates pulled from each source before re-ranking
    final_k: int = 4   # docs returned to the LLM after re-ranking

    _bm25: Any = PrivateAttr(default=None)
    _cross_encoder: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder

        tokenized = [c.page_content.lower().split() for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # --- Step 1: BM25 ---
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][: self.top_k]

        # --- Step 2: FAISS ---
        faiss_docs = self.vectorstore.similarity_search(query, k=self.top_k)

        # --- Step 3: Reciprocal Rank Fusion ---
        # RRF score = Σ 1 / (K + rank)  where K=60 is the standard constant
        K = 60
        rrf: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, i in enumerate(bm25_top):
            doc = self.chunks[i]
            key = doc.page_content
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (K + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(faiss_docs):
            key = doc.page_content
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (K + rank + 1)
            doc_map[key] = doc

        candidates = [
            doc_map[k]
            for k in sorted(rrf, key=rrf.get, reverse=True)[: self.top_k]
        ]

        if not candidates:
            return []

        # --- Step 4: Cross-encoder re-ranking ---
        # Cross-encoder sees (query, passage) together — much stronger relevance signal
        pairs = [(query, doc.page_content) for doc in candidates]
        ce_scores = self._cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[: self.final_k]]
