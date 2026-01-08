# RAGAS Evaluation Framework

Learning examples for the RAGAS (Retrieval Augmented Generation Assessment) framework.

## Setup

```bash
# Install dependencies
uv sync

# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'
```

## Project Structure

```
llm_evaluation_ragas/
├── examples/
│   ├── 01_installation/
│   │   └── setup.py               # Verify installation
│   ├── 02_basic_evaluation/
│   │   ├── simple_eval.py         # SingleTurnSample evaluation
│   │   └── dataset_eval.py        # Dataset-based evaluation
│   └── 03_rag_evaluation/
│       ├── context_metrics.py     # Context precision/recall
│       ├── faithfulness.py        # Faithfulness deep dive
│       └── full_rag_eval.py       # Complete RAG evaluation
├── data/
│   └── sample_qa_pairs.json       # Sample evaluation data
├── docs/
│   └── plan.md                    # Learning plan
└── pyproject.toml
```

## Quick Start

```bash
# 1. Verify installation
python examples/01_installation/setup.py

# 2. Run basic evaluation
python examples/02_basic_evaluation/simple_eval.py

# 3. Run full RAG evaluation
python examples/03_rag_evaluation/full_rag_eval.py
```

## Implemented Phases

- **Phase 1**: Setup & Basic Evaluation
- **Phase 2**: RAG Metrics Deep Dive
