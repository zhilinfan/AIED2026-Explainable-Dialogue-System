# Digital Appendix

## A0. Theoretical Framework of Problem Behavior Diagnosis

|  | Category | Subcategory |
|--|----------|-------------|
| **Problem Behaviors** | Aggressive Behavior | Physical Aggression, Verbal Aggression, Relational Aggression |
|  | Rule-Breaking Behavior | Non-Disturbing Discipline Violation in Class, Disturbing Discipline Violation in Class, Discipline Violation Outside Class |
|  | Delinquent Behavior | Lying Behavior, Stealing Behavior, Immoral Behavior |
|  | Social Withdrawal | Verbal Withdrawal, Behavioral Withdrawal, Psychological Withdrawal |
|  | Emotional Problem | Depression, Anxiety |
|  | Learning Problem | Learning Ability Problem, Learning Method Problem, Learning Attitude Problem, Attention Problem |
|  | Egocentricity | Self-Bragging, Stubbornness, Selfishness |
|  | Special Problem | Addiction, Puppy Love, Extreme Behavior |
| **Influencing Factors** | Family Background | Family Structure, Parenting Style, Family Atmosphere, Education Background, Medical Condition, Delinquent Behaviors, Socioeconomic Status |
|  | School Environment | Teacher Leadership, Peer Acceptance |
|  | Socioculture | Mass Media, Cultural Custom |
|  | Personality | Gender, Grade, Health Condition, Social Group |
| **Psychological Needs** | Physiological Needs | Difficulty in Emotion Control, Pathologic Problem |
|  | Safety Needs | Lack of Security |
|  | Belongingness and Love Needs | Lack of Friendship Support, Lack of Family Affection, Frustrated Romantic Relationship, Lack of Attention |
|  | Esteem Needs | Frustrated Self-Esteem, Lack of Confidence |
|  | Cognition Needs | Mismatched Cognitive Need, Incorrect Perception, Lack of Proper Guidance |
| **Education Strategies** | School Education | Persuasive Education Method, Exemplary Demonstration Method, Emotional Edification Method, Conduct Guidance Method, Practical Training Method, Character Assessment Method, Appreciative Education Method, Learning Assistance Method |
|  | Family Collaboration | Parenting Assistance Method, Home Learning Method, Parent-school Communication Method, Parent Volunteering Method, Decision-making Involvement Method, Community Collaboration Method |

## A1. Dialogue Example

Below is an example of a multi-turn diagnostic dialogue between a teacher and the expert system:

| Turn | Role | Content |
|------|------|---------|
| 0 | Teacher | I have a question related to student behavior. Can you help me? |
| 0 | Expert | Sure, I'll do my best to help you with your questions. What problem behaviors do students exhibit? |
| 1 | Teacher | My student doesn't want to interact with his classmates. When I try to speak to him, he doesn't respond and just keeps his head down and sits quietly by himself. |
| 1 | Expert | What grade is the student in? |
| 2 | Teacher | Grade five. |
| 2 | Expert | What about the student's family atmosphere? |
| 3 | Teacher | His family members often seem to be at odds. I hear that his mom often argues with his dad, and they seldom pay attention to his studies. |
| 3 | Expert | Can you describe the student's health condition? |
| 4 | Teacher | I'm not sure, but he was born with a harelip. |
| 4 | Expert | What is the level of acceptance by his classmates or peers? |
| 5 | Teacher | He gets along with his peers in general, no major conflicts, but no very close friends either. |
| 5 | Expert | This is primarily **behavioral withdrawal**. The main reason is **lack of confidence**. I suggest the following strategies: **1. Family collaboration:** Communicate with parents to reduce conflict in his family and encourage them to show more care for the student. **2. Persuasive education:** Have conversations with him individually and you can make use of shared books or impactful stories to guide and encourage him. **3. Appreciative education:** Recognize and praise the student's efforts and progress to boost his confidence. |

## A2. Fine-tuning Settings

### Base Model
- **Model**: Qwen2.5-3B-Instruct
- **Source**: `Qwen/Qwen2.5-3B-Instruct` (Hugging Face)

### LoRA Configuration
| Parameter | Value |
|-----------|-------|
| PEFT Type | LoRA |
| Rank (r) | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.0 |
| Bias | none |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Number of Epochs | 3 |
| Batch Size | 2 |
| Total Training Steps | 1,587 |
| Learning Rate Schedule | Cosine with warmup |
| Peak Learning Rate | 5e-5 |

### Training Results
| Metric | Value |
|--------|-------|
| Final Training Loss | 0.263 |
| Training Runtime | ~7,105 seconds (~2 hours) |
| Training Samples/Second | 1.786 |
| Total FLOPs | 6.90e+16 |

The training loss decreased from ~3.6 at the beginning to ~0.26 at the end of training, showing stable convergence over 3 epochs.

## A3. Two-Level Attribution

This section provides the implementation of the **Two-Level Attribution Method** for explaining Large Language Model (LLM) outputs in multi-turn dialogue systems.

### Overview

The two-level attribution method identifies which dialogue evidence contributes to LLM-generated outputs:

- **Level 1: Turn-Level Attribution** - Identifies the most influential dialogue turn using marginal likelihood gain
- **Level 2: Sentence-Level Attribution** - Identifies key sentences within the selected turn

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

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

### Attribution Methods

#### Level 1: Turn-Level Attribution

Uses **Marginal Likelihood Gain** to compute the contribution of each dialogue turn:

$$g_i = \psi_i - \psi_{i-1}, \quad \psi_i = \log P_\theta(y | C_i)$$

#### Level 2: Sentence-Level Attribution

Three baseline methods are provided:

| Method | Formula | Description |
|--------|---------|-------------|
| **LOO** | $s_j = \psi - \psi_{-j}$ | Leave-One-Out: measures likelihood change when removing a sentence |
| **GradNorm** | $s_j = \|\nabla_{e_j} \mathcal{L}\|$ | Gradient Norm: uses gradient magnitude as importance signal |
| **Sim** | $s_j = \cos(e_j, e_y)$ | Semantic Similarity: computes embedding similarity with target |

### Evaluation

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

| Metric | Description |
|--------|-------------|
| Hit@1 | Whether top-1 prediction matches ground truth |
| Hit@3 | Whether ground truth appears in top-3 predictions |
| MRR | Mean Reciprocal Rank of ground truth |

### Project Structure

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

### License

This project is licensed under the MIT License. 
