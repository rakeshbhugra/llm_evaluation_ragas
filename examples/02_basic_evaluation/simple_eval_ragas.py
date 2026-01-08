"""
Simple Prompt Evaluation with RAGAS

This script uses RAGAS's @experiment() decorator pattern - the official
way to evaluate prompts with RAGAS framework.

Reference: https://docs.ragas.io/en/stable/getstarted/tutorials/prompt_eval/
"""

from dotenv import load_dotenv
from typing import Literal

import pandas as pd
import litellm
from pydantic import BaseModel, Field

from ragas import experiment, Dataset
from ragas.metrics import discrete_metric
from ragas.metrics.result import MetricResult
load_dotenv()



# =============================================================================
# Pydantic Model for Structured Output
# =============================================================================

class SentimentOutput(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the movie review"
    )


# =============================================================================
# Prompt to Evaluate
# =============================================================================

def run_prompt(text: str) -> str:
    """Run the sentiment classification prompt."""
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


# =============================================================================
# RAGAS Metric Definition (Custom)
# =============================================================================

@discrete_metric(name="accuracy", allowed_values=["pass", "fail"])
def accuracy_metric(prediction: str, actual: str) -> MetricResult:
    """Calculate accuracy of the prediction."""
    if prediction == actual:
        return MetricResult(value="pass", reason="Prediction matches actual")
    return MetricResult(value="fail", reason=f"Expected {actual}, got {prediction}")


# =============================================================================
# RAGAS Built-in Metrics (for reference)
# =============================================================================
# RAGAS provides several built-in metrics in ragas.metrics:
#
# Non-LLM Metrics (no API calls needed):
#   - ExactMatch: checks if response == reference (returns 0 or 1)
#   - StringPresence: checks if reference string is in response
#   - NonLLMStringSimilarity: Levenshtein/Jaro distance similarity
#   - BleuScore, RougeScore, ChrfScore: NLP similarity scores
#
# LLM-based Metrics (require LLM):
#   - Faithfulness: is answer grounded in context?
#   - AnswerRelevancy: is answer relevant to question?
#   - ContextPrecision: are retrieved contexts relevant?
#   - ContextRecall: did we retrieve all relevant contexts?
#   - AnswerCorrectness: is answer factually correct?
#
# Usage example with ExactMatch:
#   from ragas.metrics import ExactMatch
#   exact_match = ExactMatch()
#   # Requires: sample.reference and sample.response fields
#
# Location: .venv/lib/python3.12/site-packages/ragas/metrics/_string.py


# =============================================================================
# RAGAS Experiment
# =============================================================================

@experiment()
async def run_experiment(row):
    """Run experiment on a single row using RAGAS pattern."""
    response = run_prompt(row["text"])
    score = accuracy_metric.score(prediction=response, actual=row["label"])

    experiment_view = {
        **row,
        "response": response,
        "score": score.value,
    }
    return experiment_view


# =============================================================================
# Dataset
# =============================================================================

def create_dataset():
    """Create test dataset with sample movie reviews."""
    samples = [
        {"text": "I loved the movie! It was fantastic.", "label": "positive"},
        {"text": "The movie was terrible and boring.", "label": "negative"},
        {"text": "It was an average film, nothing special.", "label": "negative"},
        {"text": "Absolutely amazing! Best movie of the year.", "label": "positive"},
        {"text": "Waste of time. Do not watch.", "label": "negative"},
        {"text": "A masterpiece of storytelling.", "label": "positive"},
    ]
    df = pd.DataFrame(samples)
    return Dataset.from_pandas(df, name="sentiment_eval", backend="local/csv", root_dir="./data")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n" + "#" * 60)
    print("# Movie Review Sentiment Evaluation")
    print("# Using: RAGAS @experiment() pattern")
    print("#" * 60)

    # Create dataset
    dataset = create_dataset()
    print("\nDataset:")
    print(dataset)
    print()

    # Run experiment
    print("Running RAGAS experiment...")
    try:
        asyncio.run(run_experiment.arun(dataset))
    except IndexError:
        # Known issue with RAGAS async cleanup
        pass

    print("\nExperiment complete! Check ./data/ for results.")
