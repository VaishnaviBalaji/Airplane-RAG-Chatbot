#!/usr/bin/env python3
"""RAGAS evaluation script — runs the full pipeline and checks quality thresholds.

Usage:
    python tests/eval/build_index.py   # build FAISS index first (CI step 1)
    python tests/eval/run_eval.py      # run evaluation (CI step 2)

Requires: OPENAI_API_KEY set in environment or .env file.
Exits with code 1 if any metric falls below its threshold.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Thresholds — fail CI if any metric drops below these values.
# Based on baseline scores: faithfulness=0.983, relevancy=0.870,
# precision=0.758, recall=0.900. Set slightly below to allow normal variance.
THRESHOLDS = {
    "faithfulness": 0.90,
    "answer_relevancy": 0.80,
    "context_precision": 0.65,
    "context_recall": 0.85,
}


def main():
    from rag.pipeline import _load_pipeline, answer_question, _vectorstore

    print("Loading pipeline...")
    _load_pipeline()

    test_set_path = Path(__file__).parent / "test_set.json"
    test_set = json.loads(test_set_path.read_text())
    print(f"Loaded {len(test_set)} test questions.")

    questions, answers, contexts, ground_truths = [], [], [], []

    print("\nRunning pipeline on test questions...")
    for item in test_set:
        result = answer_question(item["question"], k=4)
        retrieved = _vectorstore.similarity_search(item["question"], k=4)
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in retrieved])
        ground_truths.append(item["ground_truth"])
        print(f"  ✓  {item['question'][:65]}")

    print("\nRunning RAGAS evaluation (makes LLM calls — ~2 min)...")

    import nest_asyncio
    nest_asyncio.apply()

    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=gt,
        )
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
    ]

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = result.to_pandas()
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores = df[metric_cols].mean().to_dict()

    print("\n=== RAGAS Evaluation Results ===")
    failed = []
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        if status == "FAIL":
            failed.append(metric)
        flag = "✓" if status == "PASS" else "✗"
        print(f"  {flag}  {metric:<22} {score:.3f}  (threshold ≥ {threshold})")

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        print("Fix retrieval or prompts, then re-run before merging.")
        sys.exit(1)

    print("\nAll metrics passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
