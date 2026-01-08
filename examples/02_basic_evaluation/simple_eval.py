"""
Simple Prompt Evaluation with LiteLLM + Pydantic

This script demonstrates a basic evaluation approach using
litellm with pydantic structured output.

Example: Movie review sentiment classifier
"""

from dotenv import load_dotenv
from typing import Literal

import litellm
from pydantic import BaseModel, Field
load_dotenv()



class SentimentOutput(BaseModel):
    """Structured output for sentiment classification."""
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the movie review"
    )


def run_prompt(text: str) -> str:
    """Run the sentiment classification prompt with structured output."""
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment classifier. Classify movie reviews as positive or negative.",
            },
            {"role": "user", "content": text},
        ],
        response_format=SentimentOutput,
    )
    output = SentimentOutput.model_validate_json(response.choices[0].message.content)
    return output.sentiment


def evaluate(dataset: list[dict]) -> dict:
    """Run evaluation on dataset."""
    results = []
    correct = 0

    print("=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)

    for i, sample in enumerate(dataset, 1):
        print(f"\n[{i}/{len(dataset)}] Evaluating...")
        print(f"  Text: {sample['text'][:50]}...")

        prediction = run_prompt(sample["text"])
        is_correct = prediction == sample["label"]

        if is_correct:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"  Expected: {sample['label']}")
        print(f"  Predicted: {prediction}")
        print(f"  Result: {status}")

        results.append({
            "text": sample["text"],
            "expected": sample["label"],
            "predicted": prediction,
            "correct": is_correct
        })

    return {
        "results": results,
        "accuracy": correct / len(dataset),
        "correct": correct,
        "total": len(dataset)
    }


MOVIE_REVIEWS = [
    {"text": "I loved the movie! It was fantastic.", "label": "positive"},
    {"text": "The movie was terrible and boring.", "label": "negative"},
    {"text": "It was an average film, nothing special.", "label": "negative"},
    {"text": "Absolutely amazing! Best movie of the year.", "label": "positive"},
    {"text": "Waste of time. Do not watch.", "label": "negative"},
    {"text": "A masterpiece of storytelling.", "label": "positive"},
]


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Movie Review Sentiment Evaluation")
    print("# Using: litellm + pydantic structured output")
    print("#" * 60)

    summary = evaluate(MOVIE_REVIEWS)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total']})")

    print("\nDetailed Results:")
    for r in summary["results"]:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"  [{status}] {r['text'][:40]}... -> {r['predicted']}")
