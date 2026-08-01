"""Hugging Face loading, generation and benchmarking helpers.

These mirror the boilerplate repeated in the fine-tuning classes (4-bit
quantised loading, a generation helper and the VRAM/latency measurements).
"""

from __future__ import annotations

import time
from typing import Any


def four_bit_config(compute_dtype: Any = None, quant_type: str = "nf4", double_quant: bool = True):
    """The QLoRA 4-bit ``BitsAndBytesConfig`` used throughout the fine-tuning classes."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype or torch.float16,
        bnb_4bit_use_double_quant=double_quant,
    )


def load_tokenizer(model_name: str, **kwargs: Any):
    """Load a tokenizer with a padding token guaranteed to exist."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_name: str, four_bit: bool = True, device_map: str = "auto", for_training: bool = False, **kwargs: Any):
    """Load a causal LM, optionally 4-bit quantised and prepared for k-bit training."""
    import torch
    from transformers import AutoModelForCausalLM

    use_cuda = torch.cuda.is_available()
    if four_bit and use_cuda:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=four_bit_config(),
            device_map=device_map,
            trust_remote_code=True,
            **kwargs,
        )
        if for_training:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            trust_remote_code=True,
            **kwargs,
        )

    model.config.use_cache = not for_training
    return model


def load_model_and_tokenizer(model_name: str, **kwargs: Any):
    """Convenience wrapper returning ``(model, tokenizer)``."""
    return load_causal_lm(model_name, **kwargs), load_tokenizer(model_name)


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    skip_prompt: bool = False,
) -> str:
    """Generate a completion for ``prompt`` with the settings used in class."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in tokenizer(prompt, return_tensors="pt").items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:] if skip_prompt else outputs[0]
    return tokenizer.decode(generated, skip_special_tokens=True)


def benchmark_generation(model, tokenizer, prompt: str, max_new_tokens: int = 120, **generate_kwargs: Any) -> dict:
    """Time a generation call and report tokens/second and peak VRAM."""
    import torch

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.time()
    text = generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens, **generate_kwargs)
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.time() - started

    generated_tokens = len(tokenizer(text)["input_ids"])
    return {
        "text": text,
        "seconds": elapsed,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / elapsed if elapsed else 0.0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if use_cuda else 0.0,
    }
