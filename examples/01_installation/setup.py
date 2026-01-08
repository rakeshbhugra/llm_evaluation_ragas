"""
RAGAS Installation Verification and Setup

This script verifies that RAGAS is installed correctly and demonstrates
basic LLM configuration for evaluation.

Phase 1: Setup & Basic Evaluation
"""

import os
import sys


def check_installation():
    """Verify RAGAS and dependencies are installed correctly."""
    print("=" * 60)
    print("RAGAS Installation Verification")
    print("=" * 60)

    # Check ragas
    try:
        import ragas

        print(f"✓ ragas installed: version {ragas.__version__}")
    except ImportError as e:
        print(f"✗ ragas not installed: {e}")
        return False

    # Check langchain-openai
    try:
        import langchain_openai

        print(f"✓ langchain-openai installed")
    except ImportError as e:
        print(f"✗ langchain-openai not installed: {e}")
        return False

    # Check langchain-core
    try:
        import langchain_core

        print(f"✓ langchain-core installed")
    except ImportError as e:
        print(f"✗ langchain-core not installed: {e}")
        return False

    # Check openai
    try:
        import openai

        print(f"✓ openai installed: version {openai.__version__}")
    except ImportError as e:
        print(f"✗ openai not installed: {e}")
        return False

    # Check datasets
    try:
        import datasets

        print(f"✓ datasets installed: version {datasets.__version__}")
    except ImportError as e:
        print(f"✗ datasets not installed: {e}")
        return False

    return True


def check_api_keys():
    """Check if required API keys are configured."""
    print("\n" + "=" * 60)
    print("API Key Configuration")
    print("=" * 60)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        masked_key = openai_key[:8] + "..." + openai_key[-4:]
        print(f"✓ OPENAI_API_KEY configured: {masked_key}")
    else:
        print("✗ OPENAI_API_KEY not set")
        print("  Set it with: export OPENAI_API_KEY='your-key-here'")
        return False

    return True


def demo_llm_factory():
    """Demonstrate LLM factory usage for RAGAS."""
    print("\n" + "=" * 60)
    print("LLM Factory Demo")
    print("=" * 60)

    try:
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI

        # Create OpenAI LLM wrapper
        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        print(f"✓ LLM wrapper created successfully")
        print(f"  Model: gpt-4o-mini")

        return llm
    except Exception as e:
        print(f"✗ Failed to create LLM wrapper: {e}")
        return None


def demo_embeddings():
    """Demonstrate embeddings setup for RAGAS."""
    print("\n" + "=" * 60)
    print("Embeddings Setup Demo")
    print("=" * 60)

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import OpenAIEmbeddings

        # Create embeddings wrapper
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        print(f"✓ Embeddings wrapper created successfully")
        print(f"  Model: text-embedding-ada-002 (default)")

        return embeddings
    except Exception as e:
        print(f"✗ Failed to create embeddings wrapper: {e}")
        return None


def list_available_metrics():
    """List available RAGAS metrics."""
    print("\n" + "=" * 60)
    print("Available RAGAS Metrics")
    print("=" * 60)

    try:
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            context_entity_recall,
        )

        metrics = [
            ("faithfulness", "Measures if the answer is grounded in the context"),
            ("answer_relevancy", "Measures if the answer is relevant to the question"),
            ("context_precision", "Measures precision of retrieved contexts"),
            ("context_recall", "Measures recall of relevant contexts"),
            ("context_entity_recall", "Measures entity-level recall in contexts"),
        ]

        print("\nRAG Evaluation Metrics:")
        for name, description in metrics:
            print(f"  • {name}: {description}")

        return True
    except ImportError as e:
        print(f"✗ Failed to import metrics: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "#" * 60)
    print("# RAGAS Setup and Installation Verification")
    print("#" * 60 + "\n")

    # Check installation
    if not check_installation():
        print("\n❌ Installation check failed. Please install dependencies:")
        print("   uv add ragas langchain-openai langchain-core openai datasets")
        sys.exit(1)

    # Check API keys
    api_keys_ok = check_api_keys()

    # List metrics (doesn't require API key)
    list_available_metrics()

    # Demo LLM and embeddings setup (requires API key)
    if api_keys_ok:
        demo_llm_factory()
        demo_embeddings()
    else:
        print("\n⚠️  Skipping LLM/Embeddings demo (API key not configured)")

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    if api_keys_ok:
        print("✓ All checks passed. You're ready to use RAGAS!")
    else:
        print("⚠️  Configure your API key to enable all features.")

    print("\nNext steps:")
    print("  1. Run examples/02_basic_evaluation/simple_eval.py")
    print("  2. Run examples/02_basic_evaluation/dataset_eval.py")


if __name__ == "__main__":
    main()
