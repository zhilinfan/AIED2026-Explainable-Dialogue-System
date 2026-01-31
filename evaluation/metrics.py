"""
Evaluation metrics for attribution methods.

Provides metrics for evaluating attribution quality including:
- Hit@k: Whether ground truth appears in top-k predictions
- MRR: Mean Reciprocal Rank
- NDCG@k: Normalized Discounted Cumulative Gain
- Precision@k and Recall@k
"""

from typing import List, Dict, Union, Optional
import numpy as np


def hit_at_k(ranked_items: List[str], ground_truth: List[str], k: int) -> float:
    """
    Compute Hit@k metric.

    Returns 1 if any ground truth item appears in top-k ranked items, 0 otherwise.

    Args:
        ranked_items: List of items ranked by attribution score (highest first)
        ground_truth: List of ground truth relevant items
        k: Number of top items to consider

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if not ground_truth:
        return 0.0

    top_k = set(ranked_items[:k])
    ground_truth_set = set(ground_truth)

    return 1.0 if len(top_k & ground_truth_set) > 0 else 0.0


def mean_reciprocal_rank(ranked_items: List[str], ground_truth: List[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Returns the reciprocal of the rank of the first relevant item.

    Args:
        ranked_items: List of items ranked by attribution score (highest first)
        ground_truth: List of ground truth relevant items

    Returns:
        Reciprocal rank value (0 if no relevant item found)
    """
    if not ground_truth:
        return 0.0

    ground_truth_set = set(ground_truth)

    for i, item in enumerate(ranked_items):
        if item in ground_truth_set:
            return 1.0 / (i + 1)

    return 0.0


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at k.

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Number of top items to consider

    Returns:
        DCG@k value
    """
    relevance_scores = np.array(relevance_scores[:k])
    if len(relevance_scores) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
    return np.sum(relevance_scores / discounts)


def ndcg_at_k(ranked_items: List[str], ground_truth: List[str], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.

    Args:
        ranked_items: List of items ranked by attribution score (highest first)
        ground_truth: List of ground truth relevant items
        k: Number of top items to consider

    Returns:
        NDCG@k value
    """
    if not ground_truth:
        return 0.0

    ground_truth_set = set(ground_truth)

    # Compute relevance scores for ranked items
    relevance = [1.0 if item in ground_truth_set else 0.0 for item in ranked_items[:k]]

    # Compute DCG
    dcg = dcg_at_k(relevance, k)

    # Compute ideal DCG (all relevant items at top)
    ideal_relevance = [1.0] * min(len(ground_truth), k)
    idcg = dcg_at_k(ideal_relevance, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def precision_at_k(ranked_items: List[str], ground_truth: List[str], k: int) -> float:
    """
    Compute Precision@k.

    Args:
        ranked_items: List of items ranked by attribution score (highest first)
        ground_truth: List of ground truth relevant items
        k: Number of top items to consider

    Returns:
        Precision@k value
    """
    if not ground_truth or k == 0:
        return 0.0

    top_k = ranked_items[:k]
    ground_truth_set = set(ground_truth)

    relevant_in_top_k = sum(1 for item in top_k if item in ground_truth_set)

    return relevant_in_top_k / k


def recall_at_k(ranked_items: List[str], ground_truth: List[str], k: int) -> float:
    """
    Compute Recall@k.

    Args:
        ranked_items: List of items ranked by attribution score (highest first)
        ground_truth: List of ground truth relevant items
        k: Number of top items to consider

    Returns:
        Recall@k value
    """
    if not ground_truth:
        return 0.0

    top_k = set(ranked_items[:k])
    ground_truth_set = set(ground_truth)

    relevant_in_top_k = len(top_k & ground_truth_set)

    return relevant_in_top_k / len(ground_truth_set)


def evaluate_attribution(
    attribution_scores: Dict[str, float],
    ground_truth: List[str],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Evaluate attribution results against ground truth.

    Args:
        attribution_scores: Dict mapping item identifiers to attribution scores
        ground_truth: List of ground truth relevant item identifiers
        k_values: List of k values for computing metrics

    Returns:
        Dict containing all computed metrics
    """
    # Sort items by attribution score (descending)
    ranked_items = sorted(attribution_scores.keys(),
                         key=lambda x: attribution_scores[x],
                         reverse=True)

    results = {
        'mrr': mean_reciprocal_rank(ranked_items, ground_truth)
    }

    for k in k_values:
        results[f'hit@{k}'] = hit_at_k(ranked_items, ground_truth, k)
        results[f'ndcg@{k}'] = ndcg_at_k(ranked_items, ground_truth, k)
        results[f'precision@{k}'] = precision_at_k(ranked_items, ground_truth, k)
        results[f'recall@{k}'] = recall_at_k(ranked_items, ground_truth, k)

    return results


class AttributionEvaluator:
    """
    Evaluator class for attribution methods.

    Computes and aggregates metrics across multiple examples.
    """

    def __init__(self, k_values: List[int] = [1, 3, 5]):
        """
        Initialize evaluator.

        Args:
            k_values: List of k values for computing metrics
        """
        self.k_values = k_values
        self.results = []

    def add_result(
        self,
        attribution_scores: Dict[str, float],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        """
        Add a single evaluation result.

        Args:
            attribution_scores: Dict mapping item identifiers to attribution scores
            ground_truth: List of ground truth relevant item identifiers

        Returns:
            Dict containing metrics for this example
        """
        result = evaluate_attribution(attribution_scores, ground_truth, self.k_values)
        self.results.append(result)
        return result

    def get_aggregate_results(self) -> Dict[str, float]:
        """
        Compute aggregate metrics across all examples.

        Returns:
            Dict containing mean values for all metrics
        """
        if not self.results:
            return {}

        aggregate = {}
        metric_keys = self.results[0].keys()

        for key in metric_keys:
            values = [r[key] for r in self.results]
            aggregate[f'mean_{key}'] = np.mean(values)
            aggregate[f'std_{key}'] = np.std(values)

        aggregate['num_examples'] = len(self.results)

        return aggregate

    def reset(self):
        """Reset all stored results."""
        self.results = []

    def summary(self) -> str:
        """
        Generate a summary string of the evaluation results.

        Returns:
            Formatted string with evaluation summary
        """
        agg = self.get_aggregate_results()
        if not agg:
            return "No results to summarize."

        lines = [
            f"Attribution Evaluation Summary ({agg['num_examples']} examples)",
            "=" * 50
        ]

        lines.append(f"MRR: {agg['mean_mrr']:.4f} (±{agg['std_mrr']:.4f})")

        for k in self.k_values:
            lines.append(f"\nK = {k}:")
            lines.append(f"  Hit@{k}:       {agg[f'mean_hit@{k}']:.4f} (±{agg[f'std_hit@{k}']:.4f})")
            lines.append(f"  NDCG@{k}:      {agg[f'mean_ndcg@{k}']:.4f} (±{agg[f'std_ndcg@{k}']:.4f})")
            lines.append(f"  Precision@{k}: {agg[f'mean_precision@{k}']:.4f} (±{agg[f'std_precision@{k}']:.4f})")
            lines.append(f"  Recall@{k}:    {agg[f'mean_recall@{k}']:.4f} (±{agg[f'std_recall@{k}']:.4f})")

        return "\n".join(lines)


class TurnLevelEvaluator(AttributionEvaluator):
    """
    Specialized evaluator for turn-level attribution.

    Evaluates whether the most relevant dialogue turns are correctly identified.
    """

    def add_dialogue_result(
        self,
        turn_scores: List[float],
        ground_truth_turns: List[int]
    ) -> Dict[str, float]:
        """
        Add evaluation result for a dialogue.

        Args:
            turn_scores: Attribution scores for each turn (index = turn number)
            ground_truth_turns: List of turn indices that are ground truth

        Returns:
            Dict containing metrics for this dialogue
        """
        # Convert to string identifiers for compatibility
        attribution_scores = {str(i): score for i, score in enumerate(turn_scores)}
        ground_truth = [str(t) for t in ground_truth_turns]

        return self.add_result(attribution_scores, ground_truth)


class SentenceLevelEvaluator(AttributionEvaluator):
    """
    Specialized evaluator for sentence-level attribution.

    Evaluates whether the most relevant sentences within turns are correctly identified.
    """

    def add_sentence_result(
        self,
        sentence_scores: Dict[str, float],
        ground_truth_sentences: List[str]
    ) -> Dict[str, float]:
        """
        Add evaluation result for sentence-level attribution.

        Args:
            sentence_scores: Dict mapping sentence identifiers to attribution scores
            ground_truth_sentences: List of sentence identifiers that are ground truth

        Returns:
            Dict containing metrics for this example
        """
        return self.add_result(sentence_scores, ground_truth_sentences)
