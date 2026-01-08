"""
Evaluation From Scratch - Using LiteLLM + Pydantic

This script shows how to build a simple LLM evaluation from scratch,
without using RAGAS abstractions. Same movie review sentiment classification
task as simple_eval.py, but implemented manually to teach the concepts.

What you'll learn:
1. How to use litellm with pydantic structured outputs
2. How to build a simple evaluation loop
3. How metrics work under the hood
"""

from dotenv import load_dotenv
from typing import Literal

import litellm
from pydantic import BaseModel, Field
load_dotenv()



# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class SentimentOutput(BaseModel):
    """Structured output for sentiment classification."""
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the movie review"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reason: str = Field(
        description="Brief explanation for the classification"
    )


# =============================================================================
# LLM Call with Structured Output
# =============================================================================

def classify_sentiment(text: str, model: str = "gpt-4o-mini") -> SentimentOutput:
    """
    Classify sentiment using litellm with pydantic structured output.

    Just pass the pydantic model directly to response_format.
    """
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment classifier. Classify movie reviews as positive or negative."
            },
            {
                "role": "user",
                "content": f"Classify this movie review:\n\n{text}"
            }
        ],
        response_format=SentimentOutput,
    )

    return SentimentOutput.model_validate_json(response.choices[0].message.content)


# =============================================================================
# Evaluation Logic
# =============================================================================

class EvaluationResult(BaseModel):
    """Result for a single evaluation."""
    text: str
    expected: str
    predicted: str
    correct: bool
    confidence: float
    reason: str


def evaluate_single(text: str, expected_label: str, model: str = "gpt-4o-mini") -> EvaluationResult:
    """Evaluate a single example."""
    output = classify_sentiment(text, model)

    return EvaluationResult(
        text=text,
        expected=expected_label,
        predicted=output.sentiment,
        correct=(output.sentiment == expected_label),
        confidence=output.confidence,
        reason=output.reason
    )


def run_evaluation(dataset: list[dict], model: str = "gpt-4o-mini") -> dict:
    """
    Run evaluation on entire dataset.

    This is what RAGAS does under the hood - loop through data,
    call LLM, compare results, calculate metrics.
    """
    results = []
    correct_count = 0

    print("=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)

    for i, sample in enumerate(dataset, 1):
        print(f"\n[{i}/{len(dataset)}] Evaluating...")
        print(f"  Text: {sample['text'][:50]}...")

        result = evaluate_single(sample["text"], sample["label"], model)
        results.append(result)

        if result.correct:
            correct_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"  Expected: {result.expected}")
        print(f"  Predicted: {result.predicted} (confidence: {result.confidence:.2f})")
        print(f"  Reason: {result.reason}")
        print(f"  Result: {status}")

    accuracy = correct_count / len(dataset)

    return {
        "results": results,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(dataset)
    }


# =============================================================================
# Dataset (same as simple_eval.py)
# =============================================================================

MOVIE_REVIEWS = [
    {"text": "I loved the movie! It was fantastic.", "label": "positive"},
    {"text": "The movie was terrible and boring.", "label": "negative"},
    {"text": "It was an average film, nothing special.", "label": "negative"},
    {"text": "Absolutely amazing! Best movie of the year.", "label": "positive"},
    {"text": "Waste of time. Do not watch.", "label": "negative"},
    {"text": "A masterpiece of storytelling.", "label": "positive"},
]


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Movie Review Sentiment Evaluation - From Scratch")
    print("# Using: litellm + pydantic structured output")
    print("#" * 60)

    # Run evaluation
    summary = run_evaluation(MOVIE_REVIEWS)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total']})")

    print("\nDetailed Results:")
    for r in summary["results"]:
        status = "PASS" if r.correct else "FAIL"
        print(f"  [{status}] {r.text[:40]}... -> {r.predicted}")
