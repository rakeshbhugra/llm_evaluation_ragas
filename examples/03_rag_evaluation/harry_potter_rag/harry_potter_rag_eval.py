"""
Harry Potter RAG Evaluation

Evaluate the Harry Potter RAG system using RAGAS @experiment() pattern.
"""

from dotenv import load_dotenv
import asyncio
import pandas as pd
from openai import OpenAI

from ragas import experiment, SingleTurnSample, Dataset
from ragas.metrics import DiscreteMetric
from ragas.metrics._faithfulness import Faithfulness
from ragas.llms import llm_factory

from harry_potter_rag import retrieve_from_vector_db, augment, generate

load_dotenv()


# Initialize LLM
openai_client = OpenAI()
llm = llm_factory("gpt-4o-mini", client=openai_client)

# Initialize built-in metrics
faithfulness = Faithfulness(llm=llm)

# Define custom LLM-based metric
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned in the grading notes and return 'pass' or 'fail'.\nResponse: {response}\nGrading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)


def query_rag(query: str) -> dict:
    """Query the Harry Potter RAG system."""
    results = retrieve_from_vector_db(query)
    retrieved_docs = "\n\n".join(results["documents"][0])
    user_prompt = augment(retrieved_docs, query)
    answer = generate(user_prompt)
    return {"answer": answer, "contexts": results["documents"][0]}


@experiment()
async def run_experiment(row):
    """Run experiment on a single row."""
    response = query_rag(row["query"])

    # Create sample for built-in metrics
    sample = SingleTurnSample(
        user_input=row["query"],
        response=response.get("answer", ""),
        retrieved_contexts=response.get("contexts", []),
    )

    # Compute metrics
    faithfulness_score = await faithfulness.single_turn_ascore(sample)
    correctness_score = correctness_metric.score(
        llm=llm,
        response=response.get("answer", ""),
        grading_notes=row["grading_notes"],
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "faithfulness": faithfulness_score,
        "correctness": correctness_score.value,
    }
    return experiment_view


def create_dataset():
    """Create test dataset with Harry Potter questions."""
    samples = [
        {
            "query": "What is the address where the Dursleys live?",
            "grading_notes": "- 4 Privet Drive, Little Whinging, Surrey",
        },
        {
            "query": "What is Harry Potter's birthday?",
            "grading_notes": "- July 31",
        },
        {
            "query": "Who is Harry Potter's best friend?",
            "grading_notes": "- Ron Weasley or Hermione Granger",
        },
    ]
    df = pd.DataFrame(samples)
    return Dataset.from_pandas(df, name="harry_potter_eval", backend="local/csv", root_dir="./data")


if __name__ == "__main__":
    # Create dataset
    dataset = create_dataset()
    print("Dataset:")
    print(dataset)
    print()

    # Run experiment
    print("Running Harry Potter RAG evaluation...")
    asyncio.run(run_experiment.arun(dataset))
