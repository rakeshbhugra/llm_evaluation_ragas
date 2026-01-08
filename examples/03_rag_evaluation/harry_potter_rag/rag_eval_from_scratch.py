"""
Harry Potter RAG Evaluation - From Scratch

Evaluate the Harry Potter RAG system using custom metrics (no RAGAS).
"""

from dotenv import load_dotenv
import pandas as pd

from harry_potter_rag import retrieve_from_vector_db, augment, generate
from metrics import calculate_faithfulness, calculate_correctness
load_dotenv()



# Initialize metrics (mirrors RAGAS pattern)
# In RAGAS: faithfulness = Faithfulness(llm=llm)
# Here: we use functions instead of classes
MODEL = "gpt-4o-mini"


def query_rag(query: str) -> dict:
    """Query the Harry Potter RAG system."""
    results = retrieve_from_vector_db(query)
    retrieved_docs = "\n\n".join(results["documents"][0])
    user_prompt = augment(retrieved_docs, query)
    answer = generate(user_prompt)
    return {"answer": answer, "contexts": results["documents"][0]}


def run_experiment(row):
    """Run experiment on a single row."""
    response = query_rag(row["query"])

    # Compute metrics (mirrors RAGAS pattern)
    # In RAGAS: faithfulness_score = await faithfulness.single_turn_ascore(sample)
    # Here: faithfulness_score = calculate_faithfulness(...)
    faithfulness_score = calculate_faithfulness(
        question=row["query"],
        answer=response.get("answer", ""),
        contexts=response.get("contexts", []),
        model=MODEL,
    )
    correctness_score = calculate_correctness(
        response=response.get("answer", ""),
        grading_notes=row["grading_notes"],
        model=MODEL,
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "faithfulness": faithfulness_score,
        "correctness": correctness_score,
    }
    return experiment_view


# Mirrors: run_experiment.arun(dataset) in RAGAS
def run(dataset):
    """Run experiment on entire dataset."""
    results = []
    for _, row in dataset.iterrows():
        result = run_experiment(row.to_dict())
        results.append(result)
        print(f"  {row['query'][:40]}... -> faithfulness: {result['faithfulness']:.2f}, correctness: {result['correctness']}")

    # Summary
    print()
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    pass_count = sum(1 for r in results if r["correctness"] == "pass")
    print(f"Average Faithfulness: {avg_faithfulness:.2f}")
    print(f"Correctness: {pass_count}/{len(results)} passed")
    return results


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
    return pd.DataFrame(samples)


if __name__ == "__main__":
    # Create dataset
    dataset = create_dataset()
    print("Dataset:")
    print(dataset)
    print()

    # Run experiment
    # RAGAS: asyncio.run(run_experiment.arun(dataset))
    # Here: run(dataset)
    print("Running Harry Potter RAG evaluation...")
    run(dataset)
