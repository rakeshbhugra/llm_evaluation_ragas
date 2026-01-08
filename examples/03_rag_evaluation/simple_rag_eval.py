"""
Simple RAG Evaluation with RAGAS

This script demonstrates how to evaluate a RAG system using RAGAS's
@experiment() decorator pattern with an LLM-based metric.

Example: Evaluate a RAG system that answers questions about Ragas docs
"""

from dotenv import load_dotenv
import asyncio
import pandas as pd
from litellm import completion

from ragas import experiment, Dataset
from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory
from openai import OpenAI

load_dotenv()



class SimpleRAGClient:
    """Simple RAG client that retrieves and generates answers."""

    def __init__(self):
        # Simple document store (in production, use a vector DB)
        self.documents = [
            "Ragas is a library for evaluating LLM applications.",
            "Install Ragas from pip using: pip install ragas[examples]",
            "Ragas is organized around three concepts: experiments, datasets, and metrics.",
            "The @experiment decorator helps you iterate on prompts and track results.",
            "DiscreteMetric allows you to create LLM-based evaluation metrics.",
        ]

    def query(self, query: str) -> dict:
        """Query the RAG system."""
        # Simple retrieval (in production, use embeddings + vector search)
        relevant_docs = [doc for doc in self.documents if any(
            word.lower() in doc.lower()
            for word in query.split()
        )]

        if not relevant_docs:
            relevant_docs = self.documents[:2]

        # Generate answer using retrieved context
        context = "\n".join(relevant_docs)
        response = completion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Answer based on this context:\n{context}",
                },
                {"role": "user", "content": query},
            ],
        )

        return {
            "answer": response.choices[0].message.content,
            "contexts": relevant_docs,
        }


# Initialize RAG client
rag_client = SimpleRAGClient()

# Initialize LLM for metric evaluation
openai_client = OpenAI()
llm = llm_factory("gpt-4o-mini", client=openai_client)

# Define LLM-based metric
my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned in the grading notes and return 'pass' or 'fail'.\nResponse: {response}\nGrading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_experiment(row):
    """Run experiment on a single row."""
    response = rag_client.query(row["query"])

    score = my_metric.score(
        llm=llm,
        response=response.get("answer", ""),
        grading_notes=row["grading_notes"],
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "score": score.value,
    }
    return experiment_view


def create_dataset():
    """Create test dataset with sample queries and grading notes."""
    samples = [
        {
            "query": "What is Ragas?",
            "grading_notes": "- Ragas is a library for evaluating LLM applications.",
        },
        {
            "query": "How to install Ragas?",
            "grading_notes": "- install from pip using ragas[examples]",
        },
        {
            "query": "What are the main features of Ragas?",
            "grading_notes": "- organized around experiments, datasets, and metrics.",
        },
    ]
    df = pd.DataFrame(samples)
    return Dataset.from_pandas(df, name="simple_rag_eval", backend="local/csv", root_dir="./data")


if __name__ == "__main__":
    # Create dataset
    dataset = create_dataset()
    print("Dataset:")
    print(dataset)
    print()

    # Run experiment
    print("Running RAG evaluation experiment...")
    asyncio.run(run_experiment.arun(dataset))
