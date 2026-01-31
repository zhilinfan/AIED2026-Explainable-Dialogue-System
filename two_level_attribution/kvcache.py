"""
KV Cache for efficient likelihood computation.
"""

from typing import Tuple, Optional

import torch
from torch import Tensor


class KVCache:
    """
    A cache to store past_key_values and retrieve past_key_values prefixes.
    Used to speed up repeated likelihood computations with shared prefixes.
    """
    def __init__(self, input_ids: Optional[Tensor] = None, past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None):
        self.input_ids = input_ids
        self.past_key_values = past_key_values

    def insert(self, input_ids: Tensor, past_key_values: Tuple[Tuple[Tensor, Tensor], ...]):
        """Insert the past_key_values for input_ids into the cache."""
        self.input_ids = input_ids
        self.past_key_values = past_key_values

    def get(self, input_ids: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """
        Get cached KV values and uncached input_ids.

        Returns:
            (cached_kv_pairs, remaining_ids): Cached KV pairs and tokens needing computation
        """
        if self.input_ids is None or self.past_key_values is None:
            return None, input_ids

        min_length = min(self.input_ids.size(1), input_ids.size(1))
        comparison = self.input_ids[:, :min_length] == input_ids[:, :min_length]
        differing_indices = torch.nonzero(~comparison, as_tuple=False)

        if differing_indices.size(0) == 0:
            return self._get_past_key_values_prefix(min_length), input_ids[:, min_length:]

        differing_index = differing_indices[0, 1]
        return self._get_past_key_values_prefix(differing_index), input_ids[:, differing_index:]

    def _get_past_key_values_prefix(self, prefix_length: int) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """Return the past_key_values up to the prefix_length."""
        return tuple((layer_key[:, :, :prefix_length, :], layer_value[:, :, :prefix_length, :])
                     for layer_key, layer_value in self.past_key_values)
