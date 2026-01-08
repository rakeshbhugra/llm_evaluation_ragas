"""
Faithfulness Metric - From Scratch

Mirrors: ragas/metrics/_faithfulness.py

Algorithm (2 LLM calls):
1. StatementGeneratorPrompt - Break answer into atomic statements
2. NLIStatementPrompt - Check if each statement is supported by context
3. Score = # faithful statements / total statements
"""

from typing import List
import litellm
from pydantic import BaseModel, Field


# =============================================================================
# Step 1: Statement Generator
# Location: ragas/metrics/_faithfulness.py lines 34-55
# =============================================================================

class StatementsOutput(BaseModel):
    statements: List[str] = Field(description="List of atomic statements from the answer")


def generate_statements(question: str, answer: str, model: str = "gpt-4o-mini") -> StatementsOutput:
    """Break answer into atomic statements (no pronouns)."""
    prompt = """Given a question and an answer, analyze the complexity of each sentence in the answer.
Break down each sentence into one or more fully understandable statements.
Ensure that no pronouns are used in any statement.

Example:
Input: {"question": "Who was Albert Einstein?", "answer": "He was a German-born theoretical physicist."}
Output: {"statements": ["Albert Einstein was a German-born theoretical physicist."]}

Now perform the same:
Input: {"question": "%s", "answer": "%s"}
Output:""" % (question, answer)

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=StatementsOutput,
    )
    return StatementsOutput.model_validate_json(response.choices[0].message.content)


# =============================================================================
# Step 2: NLI Check
# Location: ragas/metrics/_faithfulness.py lines 73-130
# =============================================================================

class StatementVerdict(BaseModel):
    statement: str
    reason: str
    verdict: int = Field(description="1 if faithful, 0 if not")


class NLIOutput(BaseModel):
    verdicts: List[StatementVerdict]


def check_nli(context: str, statements: List[str], model: str = "gpt-4o-mini") -> NLIOutput:
    """Check if each statement can be inferred from context."""
    prompt = """Judge the faithfulness of statements based on context.
Return verdict=1 if statement can be directly inferred from context, 0 if not.

Context: %s
Statements: %s

Output:""" % (context, statements)

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=NLIOutput,
    )
    return NLIOutput.model_validate_json(response.choices[0].message.content)


# =============================================================================
# Main Function
# Location: ragas/metrics/_faithfulness.py lines 202-214
# =============================================================================

def calculate_faithfulness(question: str, answer: str, contexts: List[str], model: str = "gpt-4o-mini") -> float:
    """
    Calculate faithfulness score.

    Args:
        question: The user's question
        answer: The generated answer
        contexts: List of retrieved context strings
        model: LLM model to use

    Returns:
        Faithfulness score between 0 and 1
    """
    context = "\n".join(contexts)

    # Step 1: Generate statements
    statements = generate_statements(question, answer, model).statements
    if not statements:
        return 0.0

    # Step 2: NLI check
    nli_output = check_nli(context, statements, model)

    # Step 3: Calculate score
    faithful_count = sum(1 for v in nli_output.verdicts if v.verdict == 1)
    return faithful_count / len(statements)
