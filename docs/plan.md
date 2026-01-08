# RAGAS Evaluation Framework - Learning Plan

## Overview

RAGAS (Retrieval Augmented Generation Assessment) is an evaluation framework for testing and validating LLM applications, specifically RAG systems, prompts, AI workflows, and agents.

---

## Documentation URLs & Summaries

### Getting Started

| URL | Summary |
|-----|---------|
| https://docs.ragas.io/en/stable/getstarted/ | Main entry point covering installation, quick start, and tutorials |
| https://docs.ragas.io/en/stable/getstarted/install/ | Installation guide: `pip install ragas`, dev setup, LangChain dependency notes |
| https://docs.ragas.io/en/stable/getstarted/evals/ | Core evaluation workflow: dataset loading, query execution, evaluation, results display |
| https://docs.ragas.io/en/stable/tutorials/ | Four tutorials: prompts, RAG systems, AI workflows, and agents |

### Core Concepts

| URL | Summary |
|-----|---------|
| https://docs.ragas.io/en/stable/concepts/metrics/ | Metrics overview: RAG metrics, NVIDIA metrics, agent metrics, NLP comparison, SQL |
| https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ | Complete list of all available metrics with use cases |
| https://docs.ragas.io/en/stable/concepts/components/ | Core components: Prompt Object, Evaluation Sample, Evaluation Dataset |
| https://docs.ragas.io/en/stable/concepts/test_data_generation/ | Synthetic dataset creation for RAG and agent evaluation |
| https://docs.ragas.io/en/stable/concepts/test_data_generation/rag/ | RAG test data: knowledge graphs, query types, scenario generation |
| https://docs.ragas.io/en/stable/concepts/test_data_generation/agents/ | Agent test data generation (under development) |

### Customizations

| URL | Summary |
|-----|---------|
| https://docs.ragas.io/en/stable/howtos/customizations/ | All customization options: models, caching, metrics, language adaptation |
| https://docs.ragas.io/en/stable/howtos/integrations/ | Integrations with LangChain, LlamaIndex, Bedrock, Arize, LangSmith, etc. |

### API References

| URL | Summary |
|-----|---------|
| https://docs.ragas.io/en/stable/references/ | API docs for LLMs, embeddings, metrics, evaluate(), testset generation |

---

## Available Metrics

### RAG Metrics
- **Context Precision** - Measures precision of retrieved context
- **Context Recall** - Measures recall of relevant context
- **Context Entities Recall** - Entity-level recall measurement
- **Noise Sensitivity** - Robustness to irrelevant context
- **Response Relevancy** - How relevant is the response to the query
- **Faithfulness** - Groundedness of response in context

### Agent/Tool Metrics
- **Topic Adherence** - Staying on topic
- **Tool Call Accuracy** - Correct tool selection
- **Tool Call F1** - F1 score for tool calls
- **Agent Goal Accuracy** - Achievement of stated goals

### NLP Comparison Metrics
- **Factual Correctness** - Accuracy of facts
- **Semantic Similarity** - Meaning-based comparison
- **BLEU, ROUGE, CHRF** - Traditional NLP metrics
- **Exact Match** - String-level matching

### SQL Metrics
- Query equivalence
- Execution-based scoring

### General Purpose
- **Aspect Critic** - Custom aspect evaluation
- **Rubrics-Based Scoring** - Rubric-guided evaluation
- **Simple Criteria Scoring** - Single-criterion evaluation

---

## Code Structure for Learning

```
llm_evaluation_ragas/
├── docs/
│   └── plan.md                    # This file
├── examples/
│   ├── 01_installation/
│   │   └── setup.py               # Basic installation verification
│   ├── 02_basic_evaluation/
│   │   ├── simple_eval.py         # SingleTurnSample evaluation
│   │   └── dataset_eval.py        # Dataset-based evaluation
│   ├── 03_rag_evaluation/
│   │   ├── context_metrics.py     # Context precision/recall
│   │   ├── faithfulness.py        # Faithfulness evaluation
│   │   └── full_rag_eval.py       # Complete RAG evaluation
│   ├── 04_prompt_evaluation/
│   │   └── prompt_eval.py         # Prompt testing
│   ├── 05_testset_generation/
│   │   ├── knowledge_graph.py     # KG creation
│   │   ├── extractors.py          # NER, keyphrase extraction
│   │   └── synthesizers.py        # Query synthesis
│   ├── 06_custom_metrics/
│   │   ├── custom_metric.py       # Creating custom metrics
│   │   └── language_adapt.py      # Language adaptation
│   ├── 07_integrations/
│   │   ├── langchain_integration.py
│   │   ├── llamaindex_integration.py
│   │   └── langsmith_tracing.py
│   └── 08_agent_evaluation/
│       ├── tool_accuracy.py       # Tool call metrics
│       └── goal_accuracy.py       # Agent goal metrics
├── pyproject.toml
└── README.md
```

---

## Implementation Plan

### Phase 1: Setup & Basic Evaluation
1. Install RAGAS with examples: `uv add ragas[examples]`
2. Configure LLM provider (OpenAI/Anthropic/etc.)
3. Create basic SingleTurnSample evaluation
4. Learn evaluate() and aevaluate() functions

### Phase 2: RAG Metrics Deep Dive
1. Implement context precision/recall evaluations
2. Test faithfulness metrics
3. Evaluate response relevancy
4. Create comprehensive RAG evaluation pipeline

### Phase 3: Test Data Generation
1. Learn knowledge graph creation
2. Implement extractors (NER, Keyphrase)
3. Build relationship builders
4. Use QuerySynthesizer for test data

### Phase 4: Custom Metrics & Advanced Features
1. Create custom metrics for specific use cases
2. Adapt metrics for different languages
3. Implement caching strategies
4. Set up observability with Arize/LangSmith

### Phase 5: Integrations
1. LangChain integration
2. LlamaIndex integration
3. Amazon Bedrock integration
4. Tracing setup

### Phase 6: Agent Evaluation
1. Tool call accuracy metrics
2. Agent goal accuracy
3. Topic adherence testing

---

## Key Code Examples

### Basic Evaluation
```python
from ragas.llms import llm_factory
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision

# Initialize LLM
llm = llm_factory("gpt-4o")

# Create evaluation dataset
dataset = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris is the capital of France.",
        "contexts": ["Paris is the capital and largest city of France."],
        "ground_truth": "Paris"
    }
]

# Run evaluation
results = evaluate(
    dataset,
    metrics=[faithfulness, context_precision],
    llm=llm
)
```

### Test Data Generation
```python
from ragas.testset.transforms.extractors import NERExtractor
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.testset.transforms import Parallel, apply_transforms

# Create extractors
ner_extractor = NERExtractor()
rel_builder = JaccardSimilarityBuilder(property_name="entities")

# Apply transforms
transforms = [Parallel(KeyphraseExtractor(), NERExtractor()), rel_builder]
apply_transforms(knowledge_graph, transforms)
```

### Custom Metric
```python
from ragas.metrics import DiscreteMetric

# Define custom evaluation criteria
custom_metric = DiscreteMetric(
    name="technical_accuracy",
    description="Evaluates technical accuracy of the response"
)
```

---

## Dependencies

```toml
[project]
dependencies = [
    "ragas[examples]",
    "langchain-openai>=0.1,<0.2",
    "langchain-core>=0.2,<0.3",
    "openai",
]

[project.optional-dependencies]
integrations = [
    "langsmith",
    "llama-index",
    "arize-phoenix",
]
```

---

## Resources

- **Documentation**: https://docs.ragas.io/en/stable/
- **GitHub**: https://github.com/explodinggradients/ragas
- **Discord**: Community support channel
- **Office Hours**: Available for evaluation setup help
