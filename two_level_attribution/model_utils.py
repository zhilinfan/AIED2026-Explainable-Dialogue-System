"""
Model utilities for loading and using language models.
"""

from typing import Dict, Tuple

import transformers
import torch

from .context_ops import flatten_context


def load_model_and_tokenizer(model_name: str, dtype: str = "float16") -> Tuple:
    """
    Load model and tokenizer.

    Args:
        model_name: Path or name of the model
        dtype: Data type for model weights ('float16', 'bfloat16', 'float32')

    Returns:
        Tuple of (model, tokenizer)
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def tokenize(tokenizer, text: str, apply_chat_template: bool = True, return_text: bool = False) -> Dict[str, torch.Tensor]:
    """
    Tokenize text and optionally apply chat template.

    Args:
        tokenizer: The tokenizer to use
        text: Text to tokenize
        apply_chat_template: Whether to apply chat template
        return_text: Whether to return the processed text as well

    Returns:
        Token encoding, optionally with processed text
    """
    if apply_chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True
        )
    encoding = tokenizer(text, return_tensors="pt")
    return (encoding, text) if return_text else encoding


def detokenize(tokenizer, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
    """Detokenize token ids back to text."""
    return tokenizer.decode(ids.squeeze().tolist(), skip_special_tokens=skip_special_tokens)


def generate_response(model, tokenizer, example: Dict, prompt_template: str, **generate_kwargs) -> Dict[str, torch.Tensor]:
    """
    Generate response for a given example.

    Args:
        model: Language model
        tokenizer: Tokenizer
        example: Dict with 'question' and 'context_tree' keys
        prompt_template: Template string with {question} and {context} placeholders
        **generate_kwargs: Additional generation arguments

    Returns:
        Dict with 'prompt_ids' and 'response_ids'
    """
    default_generate_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.1,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
    }
    generate_kwargs = {**default_generate_kwargs, **generate_kwargs}

    context = flatten_context(example["context_tree"])
    prompt = prompt_template.format(question=example["question"], context=context)
    inputs = tokenize(tokenizer, prompt).to(model.device)
    output = model.generate(**inputs, **generate_kwargs)

    return {
        "prompt_ids": inputs.input_ids,
        "response_ids": output.sequences[:, inputs.input_ids.size(1):],
    }
