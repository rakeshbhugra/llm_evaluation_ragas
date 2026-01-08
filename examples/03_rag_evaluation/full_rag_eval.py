"""
Full RAG Evaluation Pipeline with RAGAS

This script demonstrates a comprehensive RAG evaluation using all major
metrics together, simulating a production RAG system evaluation.

Phase 2: RAG Metrics Deep Dive
"""

import asyncio
import os
from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
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


def create_comprehensive_dataset():
    """
    Create a comprehensive dataset simulating real RAG outputs.

    This dataset covers various quality levels to demonstrate
    how different metrics respond to different issues.
    """
    samples = [
        # Excellent RAG output
        SingleTurnSample(
            user_input="What is Docker and what are its main benefits?",
            response="Docker is a platform for developing, shipping, and running "
            "applications in containers. Its main benefits include: 1) Consistent "
            "environments across development and production, 2) Resource efficiency "
            "compared to VMs, and 3) Easy application isolation and deployment.",
            retrieved_contexts=[
                "Docker is an open-source platform that automates the deployment "
                "of applications inside software containers.",
                "Containers provide consistent environments from development to "
                "production, eliminating 'it works on my machine' problems.",
                "Docker containers are more lightweight than virtual machines, "
                "sharing the host OS kernel for resource efficiency.",
                "Docker enables easy application isolation, making deployment "
                "and scaling straightforward.",
            ],
            reference="Docker is a containerization platform that provides "
            "consistent environments, resource efficiency, and easy deployment.",
        ),
        # Good retrieval, hallucinated answer
        SingleTurnSample(
            user_input="What is Kubernetes used for?",
            response="Kubernetes is a container orchestration platform originally "
            "developed by Google. It automates deployment, scaling, and management "
            "of containerized applications. Kubernetes was released in 2010 and "
            "is written in Go language.",  # 2010 is wrong (actually 2014)
            retrieved_contexts=[
                "Kubernetes (K8s) is an open-source container orchestration system.",
                "It was originally designed by Google and is now maintained by CNCF.",
                "Kubernetes automates deploying, scaling, and operating application "
                "containers across clusters of hosts.",
            ],
            reference="Kubernetes orchestrates containerized application deployment, "
            "scaling, and management.",
        ),
        # Poor retrieval, answer based on parametric knowledge
        SingleTurnSample(
            user_input="What is GraphQL and how does it differ from REST?",
            response="GraphQL is a query language for APIs that allows clients to "
            "request exactly the data they need. Unlike REST which has fixed endpoints, "
            "GraphQL uses a single endpoint. It prevents over-fetching and under-fetching "
            "of data, making it more efficient for complex applications.",
            retrieved_contexts=[
                "APIs are interfaces that allow software applications to communicate.",
                "REST is an architectural style for distributed systems.",
            ],  # Contexts don't mention GraphQL at all
            reference="GraphQL is a query language that differs from REST by allowing "
            "precise data fetching through a single endpoint.",
        ),
        # Good answer but irrelevant context retrieved
        SingleTurnSample(
            user_input="What is Redis used for?",
            response="Redis is an in-memory data store used for caching, session "
            "management, and real-time analytics.",
            retrieved_contexts=[
                "Redis is an open-source, in-memory data structure store.",
                "PostgreSQL is a powerful relational database system.",  # Irrelevant
                "MongoDB is a document-oriented NoSQL database.",  # Irrelevant
                "Redis supports data structures like strings, hashes, and lists.",
                "MySQL is widely used for web applications.",  # Irrelevant
            ],
            reference="Redis is an in-memory data store for caching and real-time data.",
        ),
        # Incomplete answer despite good context
        SingleTurnSample(
            user_input="What are the SOLID principles in software development?",
            response="SOLID principles are design principles for object-oriented "
            "programming. They include Single Responsibility and Open/Closed principles.",
            # Missing Liskov, Interface Segregation, Dependency Inversion
            retrieved_contexts=[
                "SOLID is an acronym for five design principles in OOP.",
                "S - Single Responsibility Principle: A class should have one reason to change.",
                "O - Open/Closed Principle: Open for extension, closed for modification.",
                "L - Liskov Substitution Principle: Subtypes must be substitutable.",
                "I - Interface Segregation Principle: Many specific interfaces are better.",
                "D - Dependency Inversion Principle: Depend on abstractions, not concretions.",
            ],
            reference="SOLID stands for Single Responsibility, Open/Closed, Liskov "
            "Substitution, Interface Segregation, and Dependency Inversion principles.",
        ),
    ]

    return EvaluationDataset(samples=samples)


def run_full_evaluation(dataset, llm, embeddings):
    """Run evaluation with all RAG metrics."""
    print("\n" + "=" * 60)
    print("Running Full RAG Evaluation")
    print("=" * 60)

    # Define all metrics
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
    ]

    # Configure metrics
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    print(f"\nDataset: {len(dataset.samples)} samples")
    print(f"Metrics: {[m.name for m in metrics]}")
    print("\nEvaluating... (this may take a minute)")

    results = evaluate(dataset=dataset, metrics=metrics)

    return results


def display_detailed_results(results):
    """Display results with detailed analysis."""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    df = results.to_pandas()

    # Per-sample results
    print("\n" + "-" * 60)
    print("Per-Sample Scores:")
    print("-" * 60)

    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_entity_recall",
    ]

    for idx, row in df.iterrows():
        question = row["user_input"][:45]
        print(f"\n[Sample {idx + 1}] {question}...")
        for col in metric_cols:
            if col in df.columns:
                score = row[col]
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                print(f"  {col:25s}: {bar} {score:.3f}")

    # Aggregate statistics
    print("\n" + "-" * 60)
    print("Aggregate Statistics:")
    print("-" * 60)

    print(f"\n{'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)

    for col in metric_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"{col:<25} {mean:>8.3f} {std:>8.3f} {min_val:>8.3f} {max_val:>8.3f}")

    return df


def analyze_issues(df):
    """Analyze common issues based on scores."""
    print("\n" + "=" * 60)
    print("Issue Analysis")
    print("=" * 60)

    issues = []

    # Check for faithfulness issues
    low_faith = df[df["faithfulness"] < 0.7]
    if len(low_faith) > 0:
        issues.append(
            f"⚠️  {len(low_faith)} samples have low faithfulness (<0.7)\n"
            "   → LLM may be hallucinating or adding unsupported information"
        )

    # Check for context precision issues
    low_precision = df[df["context_precision"] < 0.7]
    if len(low_precision) > 0:
        issues.append(
            f"⚠️  {len(low_precision)} samples have low context precision (<0.7)\n"
            "   → Retriever is returning too many irrelevant documents"
        )

    # Check for context recall issues
    low_recall = df[df["context_recall"] < 0.7]
    if len(low_recall) > 0:
        issues.append(
            f"⚠️  {len(low_recall)} samples have low context recall (<0.7)\n"
            "   → Retriever is missing relevant documents"
        )

    # Check for answer relevancy issues
    low_relevancy = df[df["answer_relevancy"] < 0.7]
    if len(low_relevancy) > 0:
        issues.append(
            f"⚠️  {len(low_relevancy)} samples have low answer relevancy (<0.7)\n"
            "   → Generated answers don't address the questions properly"
        )

    if issues:
        print("\nIdentified Issues:")
        for issue in issues:
            print(f"\n{issue}")
    else:
        print("\n✓ All metrics are above threshold (0.7)")

    # Recommendations
    print("\n" + "-" * 60)
    print("Recommendations:")
    print("-" * 60)

    avg_precision = df["context_precision"].mean()
    avg_recall = df["context_recall"].mean()
    avg_faith = df["faithfulness"].mean()

    if avg_precision < avg_recall:
        print("• Focus on improving retrieval precision (fewer irrelevant docs)")
    if avg_recall < avg_precision:
        print("• Focus on improving retrieval recall (capture more relevant docs)")
    if avg_faith < 0.8:
        print("• Consider adjusting LLM prompts to reduce hallucination")
        print("• Lower temperature settings may improve faithfulness")

    return issues


def export_results(df, output_dir: str = "data"):
    """Export results to files."""
    print("\n" + "=" * 60)
    print("Export Options")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Show export commands
    print(f"\nTo export results:")
    print(f"  df.to_csv('{output_dir}/rag_eval_results.csv', index=False)")
    print(f"  df.to_json('{output_dir}/rag_eval_results.json', orient='records')")

    # Summary stats
    summary = df[
        [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_entity_recall",
        ]
    ].describe()
    print(f"\nSummary statistics available via: df.describe()")


def main():
    """Run comprehensive RAG evaluation."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    print("\n" + "#" * 60)
    print("# Full RAG Evaluation Pipeline")
    print("#" * 60)

    # Setup
    print("\n1. Setting up LLM and embeddings...")
    llm, embeddings = create_llm_and_embeddings()

    # Create dataset
    print("\n2. Creating evaluation dataset...")
    dataset = create_comprehensive_dataset()
    print(f"   Loaded {len(dataset.samples)} samples")

    # Run evaluation
    print("\n3. Running evaluation...")
    results = run_full_evaluation(dataset, llm, embeddings)

    # Display results
    df = display_detailed_results(results)

    # Analyze issues
    analyze_issues(df)

    # Export options
    export_results(df)

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(
        """
    This evaluation covered 5 key RAG metrics:

    1. FAITHFULNESS - Is the answer grounded in context?
    2. ANSWER RELEVANCY - Does the answer address the question?
    3. CONTEXT PRECISION - Are retrieved docs relevant?
    4. CONTEXT RECALL - Did we retrieve all needed info?
    5. CONTEXT ENTITY RECALL - Are key entities captured?

    Use these metrics to:
    • Monitor RAG system quality over time
    • Compare different retrieval strategies
    • A/B test LLM configurations
    • Identify specific failure modes
    """
    )

    print("Next steps:")
    print("  • Explore Phase 3: Test data generation")
    print("  • Explore Phase 4: Custom metrics")


if __name__ == "__main__":
    main()
