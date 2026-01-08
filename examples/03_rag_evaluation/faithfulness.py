"""
Faithfulness Evaluation with RAGAS

This script provides a deep dive into the Faithfulness metric, which measures
whether the generated answer is grounded in the provided context without
introducing hallucinations.

Phase 2: RAG Metrics Deep Dive
"""

import asyncio
import os

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness


def create_llm():
    """Create LLM wrapper for RAGAS."""
    from langchain_openai import ChatOpenAI

    return LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))


def create_faithfulness_samples():
    """
    Create samples demonstrating different faithfulness scenarios.

    Faithfulness measures whether claims in the answer can be inferred
    from the given context. It detects hallucinations and unsupported claims.
    """
    samples = [
        # High faithfulness: All claims supported by context
        SingleTurnSample(
            user_input="What is the Great Wall of China?",
            response="The Great Wall of China is a series of fortifications "
            "made of stone, brick, and other materials. It was built along "
            "the historical northern borders of China.",
            retrieved_contexts=[
                "The Great Wall of China is a series of fortifications and "
                "walls made of stone, brick, tamped earth, and other materials.",
                "Built along the historical northern borders of China, the wall "
                "was constructed to protect against various nomadic groups.",
                "The Great Wall spans over 13,000 miles and is visible from space.",
            ],
        ),
        # Low faithfulness: Contains hallucinated information
        SingleTurnSample(
            user_input="When was the Eiffel Tower built?",
            response="The Eiffel Tower was built in 1889 for the World's Fair. "
            "It was designed by Gustave Eiffel and took only 6 months to build. "
            "It was originally painted red.",  # Hallucinations!
            retrieved_contexts=[
                "The Eiffel Tower was constructed from 1887 to 1889 as the "
                "centerpiece of the 1889 World's Fair.",
                "Gustave Eiffel's engineering company designed and built the tower.",
                "Construction took 2 years, 2 months, and 5 days.",
            ],
            # Note: "6 months" and "painted red" are not in context
        ),
        # Medium faithfulness: Partially supported claims
        SingleTurnSample(
            user_input="What is Amazon?",
            response="Amazon is an American technology company founded by Jeff Bezos. "
            "It started as an online bookstore and is now the world's largest "
            "online retailer. Amazon also owns Whole Foods and MGM Studios.",
            retrieved_contexts=[
                "Amazon.com, Inc. is an American multinational technology company.",
                "It was founded by Jeff Bezos in 1994 in Seattle.",
                "Amazon started as an online marketplace for books.",
                "It has since expanded into cloud computing, streaming, and AI.",
            ],
            # "Whole Foods" and "MGM Studios" not in context (though true)
        ),
        # Perfect faithfulness: Direct quotes from context
        SingleTurnSample(
            user_input="What is the speed of light?",
            response="The speed of light in a vacuum is approximately "
            "299,792 kilometers per second.",
            retrieved_contexts=[
                "The speed of light in a vacuum is exactly 299,792,458 meters "
                "per second, or approximately 299,792 kilometers per second.",
                "This speed is denoted by the letter 'c' in physics equations.",
            ],
        ),
        # Zero faithfulness: Completely unsupported answer
        SingleTurnSample(
            user_input="What is the capital of Australia?",
            response="The capital of Australia is Sydney, which is famous for "
            "its Opera House and Harbour Bridge.",  # Wrong! It's Canberra
            retrieved_contexts=[
                "Australia is a country in the Southern Hemisphere.",
                "It is known for its unique wildlife including kangaroos and koalas.",
            ],
            # Context doesn't mention any capital city
        ),
    ]

    return EvaluationDataset(samples=samples)


def explain_faithfulness_scoring():
    """Explain how faithfulness scoring works."""
    print("\n" + "=" * 60)
    print("How Faithfulness Scoring Works")
    print("=" * 60)
    print("""
    RAGAS Faithfulness metric works in three steps:

    1. CLAIM EXTRACTION
       - The answer is broken down into individual claims
       - Example: "Paris is the capital of France and has 2M people"
         → Claim 1: "Paris is the capital of France"
         → Claim 2: "Paris has 2 million people"

    2. CLAIM VERIFICATION
       - Each claim is checked against the context
       - Can this claim be inferred from the given context?
       - Result: Supported (1) or Not Supported (0)

    3. SCORE CALCULATION
       Faithfulness = (Number of Supported Claims) / (Total Claims)

    EXAMPLE CALCULATION:
    ┌─────────────────────────────────┬────────────┐
    │ Claim                           │ Supported? │
    ├─────────────────────────────────┼────────────┤
    │ "The tower was built in 1889"   │ Yes (1)    │
    │ "It was designed by Eiffel"     │ Yes (1)    │
    │ "Construction took 6 months"    │ No (0)     │
    │ "It was originally red"         │ No (0)     │
    └─────────────────────────────────┴────────────┘
    Faithfulness = 2/4 = 0.5
    """)


async def detailed_faithfulness_analysis(llm):
    """Perform detailed faithfulness analysis with explanations."""
    print("\n" + "=" * 60)
    print("Detailed Faithfulness Analysis")
    print("=" * 60)

    dataset = create_faithfulness_samples()

    # Configure metric
    faithfulness.llm = llm

    print(f"\nAnalyzing {len(dataset.samples)} samples...")

    results = evaluate(dataset=dataset, metrics=[faithfulness])
    df = results.to_pandas()

    print("\n" + "-" * 60)
    print("Results with Interpretation:")
    print("-" * 60)

    interpretations = [
        "All claims supported - Perfect grounding",
        "Contains hallucinations - Some claims not in context",
        "Partially supported - Some external knowledge used",
        "Direct quotes - Perfect factual accuracy",
        "Completely unsupported - Major hallucination",
    ]

    for idx, (_, row) in enumerate(df.iterrows()):
        score = row["faithfulness"]
        question = row["user_input"]

        # Determine rating
        if score >= 0.9:
            rating = "EXCELLENT"
        elif score >= 0.7:
            rating = "GOOD"
        elif score >= 0.5:
            rating = "FAIR"
        else:
            rating = "POOR"

        print(f"\nSample {idx + 1}: {question}")
        print(f"  Score: {score:.4f} ({rating})")
        print(f"  Analysis: {interpretations[idx]}")

    return df


def demonstrate_hallucination_types():
    """Show different types of hallucinations that faithfulness catches."""
    print("\n" + "=" * 60)
    print("Types of Hallucinations Detected")
    print("=" * 60)
    print("""
    1. FACTUAL HALLUCINATION
       Context: "Python was created in 1991"
       Answer: "Python was created in 1989"  ← Wrong date

    2. ATTRIBUTION HALLUCINATION
       Context: "The study found increased productivity"
       Answer: "Harvard researchers found..."  ← Source not mentioned

    3. EXTRINSIC HALLUCINATION
       Context: "Paris is the capital of France"
       Answer: "Paris has the best restaurants"  ← Not in context

    4. INTRINSIC HALLUCINATION
       Context: "The product costs $100"
       Answer: "The affordable product costs $50"  ← Contradicts context

    5. FABRICATED DETAILS
       Context: "The meeting was successful"
       Answer: "The 2-hour meeting had 15 attendees"  ← Made up numbers
    """)


def main():
    """Run faithfulness evaluation demo."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    print("\n" + "#" * 60)
    print("# RAGAS Faithfulness Deep Dive")
    print("#" * 60)

    # Explain the metric
    explain_faithfulness_scoring()

    # Setup LLM
    print("\nSetting up LLM...")
    llm = create_llm()

    # Run detailed analysis
    asyncio.run(detailed_faithfulness_analysis(llm))

    # Show hallucination types
    demonstrate_hallucination_types()

    # Best practices
    print("\n" + "=" * 60)
    print("Best Practices for High Faithfulness")
    print("=" * 60)
    print("""
    1. RETRIEVAL QUALITY
       - Ensure retrieved contexts contain the needed information
       - Better retrieval = easier for LLM to be faithful

    2. PROMPT ENGINEERING
       - Instruct the LLM to only use provided context
       - Add: "Answer based only on the given information"

    3. TEMPERATURE SETTINGS
       - Lower temperature (0.0-0.3) reduces creative hallucination
       - Higher temperature increases risk of fabrication

    4. OUTPUT VERIFICATION
       - Use faithfulness checks before serving responses
       - Flag low-scoring answers for human review

    5. CONTEXT WINDOW
       - Don't overload with too much context
       - Focused, relevant contexts improve faithfulness
    """)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext: Run examples/03_rag_evaluation/full_rag_eval.py")


if __name__ == "__main__":
    main()
