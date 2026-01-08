"""
Correctness Metric - From Scratch

Mirrors: ragas/metrics/discrete.py (DiscreteMetric)

This is a simple LLM-based metric that checks if the response
contains points mentioned in the grading notes.
"""

from typing import Literal
import litellm
from pydantic import BaseModel, Field


class CorrectnessOutput(BaseModel):
    result: Literal["pass", "fail"] = Field(
        description="pass if response contains grading notes points, fail otherwise"
    )
    reason: str = Field(description="Explanation for the result")


def calculate_correctness(response: str, grading_notes: str, model: str = "gpt-4o-mini") -> str:
    """
    Calculate correctness score.

    This replicates RAGAS DiscreteMetric with a grading_notes prompt.

    Args:
        response: The generated response to evaluate
        grading_notes: Expected points that should be in the response
        model: LLM model to use

    Returns:
        "pass" or "fail"
    """
    prompt = f"""Check if the response contains points mentioned in the grading notes and return 'pass' or 'fail'.
Response: {response}
Grading Notes: {grading_notes}"""

    llm_response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=CorrectnessOutput,
    )
    output = CorrectnessOutput.model_validate_json(llm_response.choices[0].message.content)
    return output.result
