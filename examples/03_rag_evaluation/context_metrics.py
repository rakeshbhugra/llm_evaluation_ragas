"""
Context Metrics Evaluation with RAGAS

This script demonstrates context-focused metrics:
- Context Precision: Are the retrieved contexts relevant to the question?
- Context Recall: Are all relevant contexts retrieved?
- Context Entity Recall: Entity-level recall in contexts

Phase 2: RAG Metrics Deep Dive
"""

import asyncio
import os

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    context_precision,
    context_recall,
    context_entity_recall,
)


def create_llm_and_embeddings():
    """Create LLM and embeddings wrappers for RAGAS."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    return llm, embeddings


def create_context_precision_samples():
    """
    Create samples to demonstrate Context Precision.

    Context Precision measures how many of the retrieved contexts are
    actually relevant to answering the question. It penalizes retrieval
    systems that return irrelevant documents.
    """
    samples = [
        # High precision: All contexts are relevant
        SingleTurnSample(
            user_input="What is the Eiffel Tower made of?",
            response="The Eiffel Tower is made of wrought iron.",
            retrieved_contexts=[
                "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
                "The tower is 330 meters tall and weighs 10,100 tonnes.",
                "Gustave Eiffel's company designed and built the tower.",
            ],
            reference="The Eiffel Tower is made of wrought iron.",
        ),
        # Low precision: Most contexts are irrelevant
        SingleTurnSample(
            user_input="What year was Python created?",
            response="Python was created in 1991.",
            retrieved_contexts=[
                "Java was created by James Gosling in 1995.",  # Irrelevant
                "Python was first released in 1991 by Guido van Rossum.",
                "JavaScript was created in 1995 by Brendan Eich.",  # Irrelevant
                "Ruby was created in 1995 by Yukihiro Matsumoto.",  # Irrelevant
            ],
            reference="Python was created in 1991.",
        ),
    ]

    return EvaluationDataset(samples=samples)


def create_context_recall_samples():
    """
    Create samples to demonstrate Context Recall.

    Context Recall measures whether the retrieved contexts contain all
    the information needed to answer the question completely. It requires
    a reference answer to compare against.
    """
    samples = [
        # High recall: All information from reference is in contexts
        SingleTurnSample(
            user_input="What are the three branches of the US government?",
            response="The three branches are legislative, executive, and judicial.",
            retrieved_contexts=[
                "The US government has three branches: the legislative branch "
                "(Congress), the executive branch (President), and the judicial "
                "branch (Supreme Court).",
                "This separation of powers is designed to provide checks and balances.",
            ],
            reference="The three branches of the US government are the legislative "
            "branch, the executive branch, and the judicial branch.",
        ),
        # Low recall: Contexts missing key information
        SingleTurnSample(
            user_input="Name the planets in our solar system.",
            response="The planets are Mercury, Venus, Earth, and Mars.",
            retrieved_contexts=[
                "Mercury is the closest planet to the Sun.",
                "Venus is the second planet from the Sun.",
                "Earth is the third planet and the only one known to support life.",
                "Mars is often called the Red Planet.",
            ],  # Missing Jupiter, Saturn, Uranus, Neptune
            reference="The eight planets are Mercury, Venus, Earth, Mars, "
            "Jupiter, Saturn, Uranus, and Neptune.",
        ),
    ]

    return EvaluationDataset(samples=samples)


def create_entity_recall_samples():
    """
    Create samples to demonstrate Context Entity Recall.

    Context Entity Recall focuses on whether important entities (names,
    places, dates, etc.) from the reference are present in the contexts.
    """
    samples = [
        # High entity recall: All key entities present
        SingleTurnSample(
            user_input="Who founded Microsoft and when?",
            response="Bill Gates and Paul Allen founded Microsoft in 1975.",
            retrieved_contexts=[
                "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
                "The company started in Albuquerque, New Mexico.",
                "Gates and Allen were childhood friends from Seattle.",
            ],
            reference="Bill Gates and Paul Allen founded Microsoft in 1975 in Albuquerque.",
        ),
        # Low entity recall: Missing key entities
        SingleTurnSample(
            user_input="Who were the main scientists behind the theory of evolution?",
            response="Darwin developed the theory of evolution.",
            retrieved_contexts=[
                "The theory of evolution explains how species change over time.",
                "Natural selection is a key mechanism of evolution.",
            ],  # Missing "Darwin" and "Wallace"
            reference="Charles Darwin and Alfred Russel Wallace independently developed "
            "the theory of evolution by natural selection.",
        ),
    ]

    return EvaluationDataset(samples=samples)


def evaluate_context_precision(llm, embeddings):
    """Evaluate context precision metric."""
    print("\n" + "=" * 60)
    print("Context Precision Evaluation")
    print("=" * 60)
    print("\nMeasures: Proportion of retrieved contexts that are relevant")
    print("High score = Few irrelevant contexts retrieved")

    dataset = create_context_precision_samples()

    # Configure metric
    context_precision.llm = llm

    results = evaluate(dataset=dataset, metrics=[context_precision])
    df = results.to_pandas()

    print("\nResults:")
    for idx, row in df.iterrows():
        question = row["user_input"][:50]
        score = row["context_precision"]
        print(f"  Sample {idx + 1}: {score:.4f} - {question}...")

    return df


def evaluate_context_recall(llm, embeddings):
    """Evaluate context recall metric."""
    print("\n" + "=" * 60)
    print("Context Recall Evaluation")
    print("=" * 60)
    print("\nMeasures: Whether contexts contain all info from reference")
    print("High score = Retrieved contexts cover complete answer")

    dataset = create_context_recall_samples()

    # Configure metric
    context_recall.llm = llm

    results = evaluate(dataset=dataset, metrics=[context_recall])
    df = results.to_pandas()

    print("\nResults:")
    for idx, row in df.iterrows():
        question = row["user_input"][:50]
        score = row["context_recall"]
        print(f"  Sample {idx + 1}: {score:.4f} - {question}...")

    return df


def evaluate_entity_recall(llm, embeddings):
    """Evaluate context entity recall metric."""
    print("\n" + "=" * 60)
    print("Context Entity Recall Evaluation")
    print("=" * 60)
    print("\nMeasures: Whether key entities from reference are in contexts")
    print("High score = All important entities captured")

    dataset = create_entity_recall_samples()

    # Configure metric
    context_entity_recall.llm = llm

    results = evaluate(dataset=dataset, metrics=[context_entity_recall])
    df = results.to_pandas()

    print("\nResults:")
    for idx, row in df.iterrows():
        question = row["user_input"][:50]
        score = row["context_entity_recall"]
        print(f"  Sample {idx + 1}: {score:.4f} - {question}...")

    return df


def main():
    """Run context metrics evaluation."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    print("\n" + "#" * 60)
    print("# RAGAS Context Metrics Demo")
    print("#" * 60)

    # Setup
    print("\nSetting up LLM and embeddings...")
    llm, embeddings = create_llm_and_embeddings()

    # Run evaluations
    precision_df = evaluate_context_precision(llm, embeddings)
    recall_df = evaluate_context_recall(llm, embeddings)
    entity_df = evaluate_entity_recall(llm, embeddings)

    # Summary
    print("\n" + "=" * 60)
    print("Metric Comparison Summary")
    print("=" * 60)
    print("""
    ┌─────────────────────────┬─────────────────────────────────────┐
    │ Metric                  │ What it Measures                    │
    ├─────────────────────────┼─────────────────────────────────────┤
    │ Context Precision       │ Relevance of retrieved contexts     │
    │ Context Recall          │ Completeness of context coverage    │
    │ Context Entity Recall   │ Entity-level information capture    │
    └─────────────────────────┴─────────────────────────────────────┘

    Use these metrics to evaluate your retrieval system:
    - Low Precision → Retriever returns too much noise
    - Low Recall → Retriever misses relevant documents
    - Low Entity Recall → Key facts not captured in retrieval
    """)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext: Run examples/03_rag_evaluation/faithfulness.py")


if __name__ == "__main__":
    main()
