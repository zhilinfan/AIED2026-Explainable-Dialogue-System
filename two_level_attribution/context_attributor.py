"""
Context-level attribution methods for LLM outputs.

Provides various attribution methods including:
- LOO (Leave-One-Out)
- Gradient Norm
- Semantic Similarity
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List

import torch
import numpy as np
from tqdm import tqdm

from .tree import Tree, get_nodes_at_depth
from .context_ops import flatten_context, generate_masked_contexts
from .kvcache import KVCache
from . import model_utils


class AttributorBase(ABC):
    """Base class for attribution methods."""

    def __init__(self, model, tokenizer, use_cache: bool = False, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.use_cache = use_cache
        self.generate_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.1,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "return_dict_in_generate": True
        }

    def compute_log_likelihood(
        self,
        model,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        slice_config: Optional[Dict] = None,
        cache: Optional[KVCache] = None,
        update_cache: bool = False
    ) -> float:
        """
        Compute log likelihood of response given prompt.

        Args:
            model: Language model
            prompt_ids: Prompt token IDs [1, seq_len]
            response_ids: Response token IDs [1, response_len]
            slice_config: Optional dict with 'start_idx' and 'end_idx' for partial response
            cache: Optional KV cache for efficiency
            update_cache: Whether to update the cache

        Returns:
            Log likelihood value
        """
        with torch.no_grad():
            if cache is not None:
                past_key_values, remaining_prompt_ids = cache.get(prompt_ids[:, :-1])
                input_ids = torch.cat((remaining_prompt_ids, prompt_ids[:, -1:], response_ids), dim=1)
                response_offset = remaining_prompt_ids.size(1) + prompt_ids[:, -1:].size(1)

                if slice_config is not None:
                    labels = torch.full_like(input_ids, -100)
                    labels[:, response_offset + slice_config['start_idx']:
                           response_offset + slice_config['end_idx']] = \
                        response_ids[:, slice_config['start_idx']:slice_config['end_idx']]
                else:
                    labels = torch.cat((
                        torch.full_like(remaining_prompt_ids, -100),
                        torch.full_like(prompt_ids[:, -1:], -100),
                        response_ids
                    ), dim=1)

                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    past_key_values=past_key_values,
                    use_cache=(cache is not None)
                )
            else:
                past_key_values = None
                input_ids = torch.cat((prompt_ids, response_ids), dim=1)

                if slice_config is not None:
                    labels = torch.full_like(input_ids, -100)
                    response_offset = prompt_ids.size(1)
                    labels[:, response_offset + slice_config['start_idx']:
                           response_offset + slice_config['end_idx']] = \
                        response_ids[:, slice_config['start_idx']:slice_config['end_idx']]
                else:
                    labels = torch.cat((
                        torch.full_like(prompt_ids, -100),
                        response_ids
                    ), dim=1)

                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    past_key_values=None,
                    use_cache=False
                )

            if slice_config is not None:
                token_count = slice_config['end_idx'] - slice_config['start_idx']
            else:
                token_count = response_ids.shape[1]

            log_likelihood = -(output.loss * token_count)

            if update_cache and cache is not None:
                cache.insert(input_ids, output.past_key_values)

            return log_likelihood.detach().cpu().numpy().item()

    @abstractmethod
    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List) -> Dict:
        """Run attribution analysis."""
        pass


class LOOAttributor(AttributorBase):
    """Leave-One-Out attribution method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List,
            response_start_idx: int = None, response_end_idx: int = None) -> Dict:
        """
        Compute context attribution scores using Leave-One-Out method.

        Args:
            question: Input question
            context_tree: Hierarchical context tree
            prompt_template: Prompt template for formatting
            response_ids: Response token IDs
            response_start_idx: Optional start index for response slice
            response_end_idx: Optional end index for response slice

        Returns:
            Context tree with attribution scores
        """
        cache = KVCache() if self.use_cache else None

        response_ids = torch.tensor(response_ids).to(self.model.device)
        if len(response_ids.shape) == 1:
            response_ids = response_ids.unsqueeze(0)

        slice_config = None
        if response_start_idx is not None and response_end_idx is not None:
            slice_config = {
                'start_idx': response_start_idx,
                'end_idx': response_end_idx
            }

        # Compute full context likelihood
        full_context = flatten_context(context_tree)
        prompt = prompt_template.format(question=question, context=full_context)
        prompt_ids = model_utils.tokenize(self.tokenizer, prompt)["input_ids"].to(self.model.device)

        full_context_likelihood = self.compute_log_likelihood(
            self.model, prompt_ids, response_ids,
            slice_config=slice_config, cache=cache, update_cache=True
        )

        context_tree = Tree.from_dict(context_tree)

        # Compute attribution for each context segment
        for context, ablated_subtree in tqdm(
            generate_masked_contexts(context_tree, depth=2),
            desc="Computing attribution scores",
            leave=False
        ):
            partial_prompt = prompt_template.format(question=question, context=context)
            partial_prompt_ids = model_utils.tokenize(self.tokenizer, partial_prompt)["input_ids"].to(self.model.device)

            partial_context_likelihood = self.compute_log_likelihood(
                self.model, partial_prompt_ids, response_ids,
                slice_config=slice_config, cache=cache, update_cache=False
            )

            attribution_score = full_context_likelihood - partial_context_likelihood
            ablated_subtree.data["attribution_score"] = attribution_score

        return context_tree.to_dict()


class GradientNormAttributor(AttributorBase):
    """Gradient norm based attribution method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = None
        self.model.model.embed_tokens.register_forward_hook(self.embedding_hook_fn)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def embedding_hook_fn(self, module, input, output):
        self.embeddings = output
        output.requires_grad_(True)
        return output

    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List,
            response_start_idx: int = None, response_end_idx: int = None) -> Dict:
        """Compute attribution using gradient norms."""
        full_context = flatten_context(context_tree)
        prompt = prompt_template.format(question=question, context=full_context)
        prompt_encoding, prompt_text = model_utils.tokenize(self.tokenizer, prompt, return_text=True)
        prompt_ids = prompt_encoding["input_ids"].to(self.model.device)

        if len(prompt_ids.shape) == 2:
            response_ids = torch.tensor(response_ids).unsqueeze(0).to(self.model.device)
        else:
            response_ids = torch.tensor(response_ids).to(self.model.device)

        input_ids = torch.cat((prompt_ids, response_ids), dim=1)
        labels = torch.full_like(input_ids, -100)
        prompt_length = prompt_ids.shape[1]

        if response_start_idx is not None and response_end_idx is not None:
            labels[:, prompt_length + response_start_idx:prompt_length + response_end_idx] = \
                response_ids[:, response_start_idx:response_end_idx]
        else:
            labels[:, prompt_length:] = response_ids

        output = self.model(input_ids=input_ids, labels=labels)
        valid_token_count = (labels != -100).sum()
        log_likelihood = -(output.loss * valid_token_count)

        embedding_grads = torch.autograd.grad(log_likelihood, self.embeddings)[0]

        context_tree = Tree.from_dict(context_tree)
        ignore_prefix = 0
        token_start_indices = np.array([
            prompt_encoding.token_to_chars(i).start
            for i in range(1, prompt_encoding.input_ids.shape[1] - 1)
        ])

        nodes = get_nodes_at_depth(context_tree, 2)
        for node in nodes:
            sent = node.data['text']
            start_char = prompt_text.find(sent, ignore_prefix)
            if start_char < 0:
                raise ValueError(f"Cannot find sentence '{sent}' in prompt")
            end_char = start_char + len(sent)
            start_token = np.where(token_start_indices <= start_char)[0][-1]
            end_token = np.where(token_start_indices > end_char)[0][0]
            node.data["attribution_score"] = torch.norm(
                embedding_grads[:, start_token:end_token].reshape(-1)
            ).item()
            ignore_prefix = end_char

        return context_tree.to_dict()


class SimAttributor(AttributorBase):
    """Semantic similarity based attribution method."""

    def __init__(self, *args, similarity_model_name: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        import sentence_transformers
        if similarity_model_name:
            self.sent_model = sentence_transformers.SentenceTransformer(similarity_model_name)
        else:
            self.sent_model = sentence_transformers.SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )

    def run(self, question: str, context_tree: Dict, prompt_template: str, response_ids: List,
            response_start_idx: int = None, response_end_idx: int = None) -> Dict:
        """Compute attribution using semantic similarity."""
        context_tree = Tree.from_dict(context_tree)
        nodes = get_nodes_at_depth(context_tree, 2)

        if isinstance(response_ids, list) and len(response_ids) == 1:
            response_ids = [response_ids]
        response_ids = torch.tensor(response_ids)

        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)

        if response_start_idx is not None and response_end_idx is not None:
            response_start_idx = max(0, response_start_idx)
            response_end_idx = min(response_ids.size(1), response_end_idx)
            response = self.tokenizer.decode(response_ids[0][response_start_idx:response_end_idx])
        else:
            response = self.tokenizer.decode(response_ids[0][:-1])

        resp_embed = self.sent_model.encode(response)

        for node in nodes:
            source_embed = self.sent_model.encode(node.data['text'])
            resp_embed_reshaped = resp_embed.reshape(1, -1)
            source_embed_reshaped = source_embed.reshape(1, -1)
            similarity = self.sent_model.similarity(resp_embed_reshaped, source_embed_reshaped).item()
            node.data['attribution_score'] = similarity

        return context_tree.to_dict()


def get_attributor(**kwargs):
    """
    Factory function to create the appropriate attributor.

    Args:
        attribution_method: Method name ('loo', 'gradnorm', 'sim')
        model_name: Model path/name
        dtype: Data type ('float16', 'bfloat16', 'float32')
        **kwargs: Additional arguments

    Returns:
        Attributor instance
    """
    method = kwargs.get("attribution_method", "loo")
    assert method in ["loo", "gradnorm", "sim"], f"Invalid attribution method: {method}"

    model, tokenizer = model_utils.load_model_and_tokenizer(
        kwargs["model_name"],
        kwargs.get("dtype", "float16")
    )

    if method == "loo":
        return LOOAttributor(model, tokenizer, **kwargs)
    elif method == "gradnorm":
        return GradientNormAttributor(model, tokenizer, **kwargs)
    elif method == "sim":
        return SimAttributor(model, tokenizer, **kwargs)
