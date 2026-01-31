"""
Two-Level Attribution for Explainable LLM-based Dialogue Systems

This package provides methods for explaining LLM outputs through:
- Level 1: Turn-level attribution (dialogue turn importance)
- Level 2: Sentence-level attribution (sentence importance within turns)
"""

from .dialogue_attributor import (
    DialogueAttributor,
    DialogueTurn,
    AttributionTarget,
    AttributionChain
)
from .context_attributor import (
    LOOAttributor,
    GradientNormAttributor,
    SimAttributor,
    get_attributor
)
from .tree import Tree, traverse, get_nodes_at_depth
from .context_ops import flatten_context, generate_masked_contexts
from .model_utils import load_model_and_tokenizer, tokenize, detokenize

__version__ = "1.0.0"
__all__ = [
    # Dialogue Attribution
    "DialogueAttributor",
    "DialogueTurn",
    "AttributionTarget",
    "AttributionChain",
    # Context Attribution
    "LOOAttributor",
    "GradientNormAttributor",
    "SimAttributor",
    "get_attributor",
    # Utilities
    "Tree",
    "traverse",
    "get_nodes_at_depth",
    "flatten_context",
    "generate_masked_contexts",
    "load_model_and_tokenizer",
    "tokenize",
    "detokenize",
]
