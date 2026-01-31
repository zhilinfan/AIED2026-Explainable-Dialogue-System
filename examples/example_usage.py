"""
Example usage of the Two-Level Attribution method.

This script demonstrates how to use the two-level attribution framework
to explain LLM outputs by identifying relevant dialogue turns and sentences.
"""

import torch
from two_level_attribution import (
    DialogueAttributor,
    DialogueTurn,
    AttributionTarget
)
from two_level_attribution.context_attributor import get_attributor
from evaluation import AttributionEvaluator, TurnLevelEvaluator


def example_dialogue_attribution():
    """
    Example: Two-level attribution for dialogue-based LLM responses.
    """
    print("=" * 60)
    print("Example: Dialogue Attribution")
    print("=" * 60)

    # Define model (replace with your model path)
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    # Create dialogue turns
    dialogue_turns = [
        DialogueTurn(
            turn_id=0,
            role="user",
            content="I have a question about this topic.",
            sentences=["I have a question about this topic."]
        ),
        DialogueTurn(
            turn_id=1,
            role="assistant",
            content="Sure, I can help you with that.",
            sentences=["Sure, I can help you with that."]
        ),
        DialogueTurn(
            turn_id=2,
            role="user",
            content="It's quite complex. I find it difficult to understand.",
            sentences=["It's quite complex.", "I find it difficult to understand."]
        ),
        DialogueTurn(
            turn_id=3,
            role="assistant",
            content="Let me explain it step by step.",
            sentences=["Let me explain it step by step."]
        ),
    ]

    # Define attribution target (the LLM output to explain)
    target = AttributionTarget(
        text="The user needs more detailed explanation.",
        token_ids=[1, 2, 3, 4, 5]  # Example token IDs
    )

    print(f"\nTarget response to explain:")
    print(f"  '{target.text}'")

    print(f"\nDialogue context ({len(dialogue_turns)} turns):")
    for turn in dialogue_turns:
        print(f"  Turn {turn.turn_id} [{turn.role}]: {turn.content}")

    # Note: In practice, you would initialize the model
    # attributor = DialogueAttributor(model_name)

    # Example output of turn-level attribution
    print("\n--- Turn-Level Attribution (Level 1) ---")
    turn_scores = {
        "turn_0": 0.15,
        "turn_1": 0.05,
        "turn_2": 0.65,  # Highest - most relevant turn
        "turn_3": 0.10,
    }
    print("Attribution scores (marginal likelihood gain):")
    for turn_id, score in sorted(turn_scores.items(), key=lambda x: -x[1]):
        print(f"  {turn_id}: {score:.4f}")

    # Example output of sentence-level attribution
    print("\n--- Sentence-Level Attribution (Level 2) ---")
    print("For Turn 2 (highest scoring turn):")
    sentence_scores = {
        "It's quite complex.": 0.45,
        "I find it difficult to understand.": 0.20,
    }
    for sent, score in sorted(sentence_scores.items(), key=lambda x: -x[1]):
        print(f"  '{sent}': {score:.4f}")


def example_context_attribution():
    """
    Example: Context-level attribution using LOO method.
    """
    print("\n" + "=" * 60)
    print("Example: Context Attribution (LOO Method)")
    print("=" * 60)

    # Define a hierarchical context tree
    context_tree = {
        "data": {"header": "Context:\n"},
        "children": [
            {
                "data": {"header": "Document 1:\n", "separator": "\n\n"},
                "children": [
                    {"data": {"text": "Python is a programming language."}, "children": []},
                    {"data": {"text": "It was created by Guido van Rossum."}, "children": []},
                ]
            },
            {
                "data": {"header": "Document 2:\n", "separator": "\n\n"},
                "children": [
                    {"data": {"text": "Machine learning uses algorithms."}, "children": []},
                    {"data": {"text": "Deep learning is a subset of ML."}, "children": []},
                ]
            }
        ]
    }

    question = "Who created Python?"
    prompt_template = "Question: {question}\n\n{context}\n\nAnswer:"

    print(f"\nQuestion: {question}")
    print("\nContext structure:")
    print("  - Document 1: Python programming info")
    print("  - Document 2: Machine learning info")

    # Example attribution scores
    print("\n--- Sentence-Level Attribution Scores ---")
    example_scores = {
        "Python is a programming language.": 0.25,
        "It was created by Guido van Rossum.": 0.85,  # Most relevant
        "Machine learning uses algorithms.": 0.02,
        "Deep learning is a subset of ML.": 0.01,
    }

    for sent, score in sorted(example_scores.items(), key=lambda x: -x[1]):
        print(f"  [{score:.4f}] '{sent}'")


def example_evaluation():
    """
    Example: Evaluating attribution quality with metrics.
    """
    print("\n" + "=" * 60)
    print("Example: Attribution Evaluation")
    print("=" * 60)

    # Create evaluator
    evaluator = TurnLevelEvaluator(k_values=[1, 2, 3])

    # Simulate multiple dialogue evaluations
    test_cases = [
        {
            "turn_scores": [0.1, 0.05, 0.7, 0.15],
            "ground_truth": [2]  # Turn 2 is the ground truth
        },
        {
            "turn_scores": [0.6, 0.2, 0.1, 0.1],
            "ground_truth": [0]  # Turn 0 is the ground truth
        },
        {
            "turn_scores": [0.1, 0.5, 0.3, 0.1],
            "ground_truth": [1, 2]  # Turns 1 and 2 are ground truth
        },
    ]

    print("\nEvaluating attribution results...")
    for i, case in enumerate(test_cases):
        result = evaluator.add_dialogue_result(
            turn_scores=case["turn_scores"],
            ground_truth_turns=case["ground_truth"]
        )
        print(f"\nDialogue {i+1}:")
        print(f"  Turn scores: {case['turn_scores']}")
        print(f"  Ground truth turns: {case['ground_truth']}")
        print(f"  Hit@1: {result['hit@1']:.2f}, MRR: {result['mrr']:.4f}")

    # Print summary
    print("\n" + evaluator.summary())


def example_baseline_methods():
    """
    Example: Comparison of baseline attribution methods.
    """
    print("\n" + "=" * 60)
    print("Example: Baseline Attribution Methods")
    print("=" * 60)

    print("""
Three baseline methods are provided for sentence-level attribution:

1. LOO (Leave-One-Out):
   - Remove each sentence and measure likelihood change
   - Score = log P(y|full) - log P(y|full \ sentence)
   - Measures necessity: how much does removing this hurt?

2. GradNorm (Gradient Norm):
   - Compute gradient of loss w.r.t. input embeddings
   - Score = ||∇_x L(y|x)||
   - Measures sensitivity: how much does the model attend to this?

3. Sim (Semantic Similarity):
   - Compute embedding similarity between sentence and target
   - Score = cos(embed(sentence), embed(target))
   - Measures relevance: how similar is this to the output?
""")

    # Example comparison
    sentences = [
        "The topic is quite complex.",
        "I need more explanation.",
        "Thank you for your help.",
    ]

    print("Example scores for target: 'User needs detailed explanation'\n")
    print(f"{'Sentence':<35} {'LOO':>8} {'GradNorm':>10} {'Sim':>8}")
    print("-" * 65)

    example_data = [
        (0.45, 0.32, 0.78),
        (0.62, 0.41, 0.85),  # Highest for LOO and Sim
        (0.08, 0.12, 0.25),
    ]

    for sent, (loo, grad, sim) in zip(sentences, example_data):
        print(f"{sent:<35} {loo:>8.2f} {grad:>10.2f} {sim:>8.2f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Two-Level Attribution Method - Usage Examples")
    print("=" * 60)

    example_dialogue_attribution()
    example_context_attribution()
    example_baseline_methods()
    example_evaluation()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
