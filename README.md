# Two-Level Attribution for LLM-based Dialogue Systems

This repository provides the implementation of the **Two-Level Attribution Method** for explaining Large Language Model (LLM) outputs in multi-turn dialogue systems.

## Overview

The two-level attribution method identifies which dialogue evidence contributes to LLM-generated outputs:

- **Level 1: Turn-Level Attribution** - Identifies the most influential dialogue turn using marginal likelihood gain
- **Level 2: Sentence-Level Attribution** - Identifies key sentences within the selected turn

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from two_level_attribution import DialogueAttributor

# Load your model
model_path = "your-model-path"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float16")
model = model.cuda()

# Create attributor
attributor = DialogueAttributor(model, tokenizer)

# Example dialogue data
dialogue_data = [
    {"user": "I have a question.", "assistant": "Sure, what's the issue?"},
    {"user": "Can you help me?", "assistant": "Of course."},
    # ... more turns
]

# Run attribution on a specific target
result = attributor.attribute_single_target(
    dialogue_data,
    target_text="Your target output text",
    target_type="response",
    method="loo"  # Options: 'loo', 'gradnorm', 'sim'
)

# Print results
attributor.print_single_result(result)
```

## Attribution Methods

### Level 1: Turn-Level Attribution

Uses **Marginal Likelihood Gain** to compute the contribution of each dialogue turn:

$$g_i = \psi_i - \psi_{i-1}, \quad \psi_i = \log P_\theta(y | C_i)$$

### Level 2: Sentence-Level Attribution

Three baseline methods are provided:

| Method | Description |
|--------|-------------|
| **LOO** | Leave-One-Out: measures likelihood change when removing a sentence |
| **GradNorm** | Gradient Norm: uses gradient magnitude as importance signal |
| **Sim** | Semantic Similarity: computes embedding similarity with target |

## Evaluation

The evaluation module provides standard ranking metrics:

```python
from evaluation import AttributionEvaluator

evaluator = AttributionEvaluator()

# Evaluate attribution results against ground truth
metrics = evaluator.evaluate(
    predictions=predicted_sentences,
    ground_truth=ground_truth_sentences
)

print(f"Hit@1: {metrics['hit@1']}")
print(f"Hit@3: {metrics['hit@3']}")
print(f"MRR: {metrics['mrr']}")
```

## Project Structure

```
TwoLevelAttribution/
├── two_level_attribution/
│   ├── __init__.py
│   ├── dialogue_attributor.py    # Core two-level attribution
│   ├── tree.py                   # Tree data structure
│   ├── context_ops.py            # Context operations
│   ├── kvcache.py                # KV cache for efficiency
│   └── model_utils.py            # Model utilities
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                # Hit@k, MRR, etc.
├── examples/
│   └── example_usage.py          # Usage examples
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License.
