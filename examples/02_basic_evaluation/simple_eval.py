"""
Simple RAGAS Evaluation with SingleTurnSample

This script demonstrates basic evaluation using RAGAS with a single
question-answer pair. This is the simplest form of RAGAS evaluation.

Phase 1: Setup & Basic Evaluation
"""

import asyncio
import os

from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy


def create_llm_and_embeddings():
    """Create LLM and embeddings wrappers for RAGAS."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    return llm, embeddings


def create_sample_data():
    """Create a simple SingleTurnSample for evaluation."""
    # Create a sample with question, answer, and retrieved contexts
    sample = SingleTurnSample(
        user_input="What is the capital of France?",
        response="Paris is the capital of France. It is known for the Eiffel Tower.",
        retrieved_contexts=[
            "Paris is the capital and most populous city of France. "
            "It has been an important city since the 3rd century.",
            "The Eiffel Tower is a wrought-iron lattice tower on the "
            "Champ de Mars in Paris, France.",
        ],
    )
    return sample


async def evaluate_single_sample():
    """Evaluate a single sample using RAGAS metrics."""
    print("=" * 60)
    print("Simple RAGAS Evaluation - SingleTurnSample")
    print("=" * 60)

    # Create LLM and embeddings
    print("\n1. Setting up LLM and embeddings...")
    llm, embeddings = create_llm_and_embeddings()

    # Create sample
    print("\n2. Creating sample data...")
    sample = create_sample_data()

    print(f"\n   Question: {sample.user_input}")
    print(f"   Answer: {sample.response}")
    print(f"   Contexts: {len(sample.retrieved_contexts)} retrieved")

    # Evaluate faithfulness
    print("\n3. Evaluating faithfulness...")
    faithfulness_metric = faithfulness
    faithfulness_metric.llm = llm

    faithfulness_score = await faithfulness_metric.single_turn_ascore(sample)
    print(f"   Faithfulness Score: {faithfulness_score:.4f}")

    # Evaluate answer relevancy
    print("\n4. Evaluating answer relevancy...")
    relevancy_metric = answer_relevancy
    relevancy_metric.llm = llm
    relevancy_metric.embeddings = embeddings

    relevancy_score = await relevancy_metric.single_turn_ascore(sample)
    print(f"   Answer Relevancy Score: {relevancy_score:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  • Faithfulness:      {faithfulness_score:.4f}")
    print(f"  • Answer Relevancy:  {relevancy_score:.4f}")

    # Interpretation
    print("\n" + "-" * 60)
    print("Score Interpretation:")
    print("-" * 60)
    print("  Scores range from 0.0 to 1.0")
    print("  • 0.8 - 1.0: Excellent")
    print("  • 0.6 - 0.8: Good")
    print("  • 0.4 - 0.6: Fair")
    print("  • 0.0 - 0.4: Poor")

    return {
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
    }


def evaluate_with_hallucination():
    """Demonstrate evaluation with a hallucinated answer."""
    print("\n\n" + "=" * 60)
    print("Evaluation with Hallucinated Answer")
    print("=" * 60)

    # Create a sample with hallucination (answer contains info not in context)
    sample = SingleTurnSample(
        user_input="What is the population of Paris?",
        response="Paris has a population of 12 million people and is the "
        "largest city in Europe.",  # Hallucination!
        retrieved_contexts=[
            "Paris is the capital of France.",
            "Paris is known for its art museums and the Eiffel Tower.",
        ],
    )

    print(f"\n   Question: {sample.user_input}")
    print(f"   Answer: {sample.response}")
    print("   Note: Answer contains hallucinated information!")

    return sample


async def main():
    """Run simple evaluation examples."""
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    print("\n" + "#" * 60)
    print("# RAGAS Simple Evaluation Demo")
    print("#" * 60)

    # Run basic evaluation
    results = await evaluate_single_sample()

    # Show hallucination example structure (without running to save API calls)
    hallucination_sample = evaluate_with_hallucination()
    print("\n   (Skipping actual evaluation to save API calls)")
    print("   To evaluate, use: faithfulness.single_turn_ascore(sample)")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext: Run examples/02_basic_evaluation/dataset_eval.py")
    print("      for dataset-based evaluation.")


if __name__ == "__main__":
    asyncio.run(main())
