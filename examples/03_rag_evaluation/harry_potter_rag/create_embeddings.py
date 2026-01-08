from litellm import embedding
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def create_embeddings(texts, model="text-embedding-3-small"):
    """
    Create embeddings for a list of texts using litellm.

    Args:
        texts: List of strings to embed
        model: Model to use for embeddings (default: text-embedding-3-small)

    Returns:
        list: List of embedding vectors
    """
    # Use OpenAI directly for embeddings
    # Note: Portkey requires virtual keys setup which is more complex
    # For simplicity, we use OpenAI API directly
    response = embedding(
        model=model,
        input=texts,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Extract just the embeddings from the response
    embeddings = [item['embedding'] for item in response['data']]

    return embeddings


# Example usage
if __name__ == "__main__":
    # Make sure to set your API key
    # os.environ['OPENAI_API_KEY'] = "your-key-here"

    # Sample texts
    texts = [
        "good morning from litellm",
        "this is another text to embed",
        "embeddings are useful for semantic search"
    ]

    # Create embeddings
    embeddings = create_embeddings(texts)

    print(f"Created {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"\nFirst embedding (first 10 values): {embeddings[0][:10]}")
