"""
Dataset-Based RAGAS Evaluation

This script demonstrates how to evaluate multiple samples at once using
RAGAS's evaluate() function with an EvaluationDataset.

Phase 1: Setup & Basic Evaluation
"""

import asyncio
import os

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy


def create_llm_and_embeddings():
    """Create LLM and embeddings wrappers for RAGAS."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    return llm, embeddings


def create_evaluation_dataset():
    """Create an EvaluationDataset with multiple samples."""
    samples = [
        # Good answer with proper context
        SingleTurnSample(
            user_input="What is Python?",
            response="Python is a high-level, interpreted programming language "
            "known for its readability and versatility.",
            retrieved_contexts=[
                "Python is a high-level, general-purpose programming language. "
                "Its design philosophy emphasizes code readability.",
                "Python is dynamically typed and garbage-collected. It supports "
                "multiple programming paradigms.",
            ],
            reference="Python is a high-level programming language.",
        ),
        # Answer with some hallucination
        SingleTurnSample(
            user_input="Who created Python?",
            response="Python was created by Guido van Rossum in 1991. He invented "
            "it while working at Google.",  # Hallucination: Not at Google
            retrieved_contexts=[
                "Python was conceived in the late 1980s by Guido van Rossum.",
                "The first version of Python (0.9.0) was released in 1991.",
            ],
            reference="Guido van Rossum created Python.",
        ),
        # Good answer matching context
        SingleTurnSample(
            user_input="What is machine learning?",
            response="Machine learning is a subset of artificial intelligence "
            "that enables systems to learn from data.",
            retrieved_contexts=[
                "Machine learning (ML) is a subset of artificial intelligence (AI) "
                "that allows systems to automatically learn and improve from experience.",
                "ML algorithms build models based on sample data to make predictions "
                "without being explicitly programmed.",
            ],
            reference="Machine learning is a branch of AI focused on learning from data.",
        ),
        # Answer with missing information
        SingleTurnSample(
            user_input="What are the main features of JavaScript?",
            response="JavaScript is a programming language.",  # Incomplete
            retrieved_contexts=[
                "JavaScript is a dynamic, interpreted programming language. "
                "It is one of the core technologies of the World Wide Web.",
                "JavaScript supports event-driven, functional, and imperative "
                "programming styles. It has first-class functions.",
            ],
            reference="JavaScript is a dynamic language with first-class functions.",
        ),
    ]

    return EvaluationDataset(samples=samples)


def run_synchronous_evaluation(dataset, llm, embeddings):
    """Run synchronous evaluation using evaluate()."""
    print("\n" + "=" * 60)
    print("Synchronous Dataset Evaluation")
    print("=" * 60)

    # Configure metrics with LLM and embeddings
    metrics = [faithfulness, answer_relevancy]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    print(f"\nEvaluating {len(dataset.samples)} samples...")
    print(f"Metrics: {[m.name for m in metrics]}")

    # Run evaluation
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return results


async def run_async_evaluation(dataset, llm, embeddings):
    """Run asynchronous evaluation using aevaluate()."""
    from ragas import aevaluate

    print("\n" + "=" * 60)
    print("Asynchronous Dataset Evaluation")
    print("=" * 60)

    # Configure metrics
    metrics = [faithfulness, answer_relevancy]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    print(f"\nEvaluating {len(dataset.samples)} samples asynchronously...")

    # Run async evaluation (faster for large datasets)
    results = await aevaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return results


def display_results(results):
    """Display evaluation results in a formatted table."""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    # Get the results dataframe
    df = results.to_pandas()

    print("\nPer-sample scores:")
    print("-" * 60)

    for idx, row in df.iterrows():
        question = row.get("user_input", "N/A")[:40]
        faith = row.get("faithfulness", float("nan"))
        relev = row.get("answer_relevancy", float("nan"))

        print(f"\nSample {idx + 1}: {question}...")
        print(f"  Faithfulness:      {faith:.4f}")
        print(f"  Answer Relevancy:  {relev:.4f}")

    # Aggregate scores
    print("\n" + "-" * 60)
    print("Aggregate Scores:")
    print("-" * 60)

    for col in ["faithfulness", "answer_relevancy"]:
        if col in df.columns:
            mean_score = df[col].mean()
            print(f"  Mean {col}: {mean_score:.4f}")

    return df


def main():
    """Run dataset evaluation examples."""
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    print("\n" + "#" * 60)
    print("# RAGAS Dataset Evaluation Demo")
    print("#" * 60)

    # Setup
    print("\n1. Setting up LLM and embeddings...")
    llm, embeddings = create_llm_and_embeddings()

    # Create dataset
    print("\n2. Creating evaluation dataset...")
    dataset = create_evaluation_dataset()
    print(f"   Created dataset with {len(dataset.samples)} samples")

    # Show sample structure
    print("\n3. Sample structure:")
    sample = dataset.samples[0]
    print(f"   - user_input: {sample.user_input[:50]}...")
    print(f"   - response: {sample.response[:50]}...")
    print(f"   - retrieved_contexts: {len(sample.retrieved_contexts)} items")
    print(f"   - reference: {sample.reference[:50]}...")

    # Run synchronous evaluation
    print("\n4. Running evaluation...")
    results = run_synchronous_evaluation(dataset, llm, embeddings)

    # Display results
    df = display_results(results)

    # Export options
    print("\n" + "=" * 60)
    print("Export Options")
    print("=" * 60)
    print("  • results.to_pandas()  - Get as DataFrame")
    print("  • df.to_csv('results.csv')  - Save to CSV")
    print("  • df.to_json('results.json')  - Save to JSON")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext: Run examples/03_rag_evaluation/ for advanced metrics.")


if __name__ == "__main__":
    main()
