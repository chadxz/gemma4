#!/usr/bin/env python3

import argparse
import gc
import json
import time

import mlx.core as mx

from mlx_vlm import load
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template


BASE_SYSTEM = (
    "You are a local coding assistant running on a developer laptop. Be concise, "
    "accurate, and practical."
)

BASE_RULES = [
    "Read the user's request carefully before proposing changes.",
    "Prefer concrete answers over generic advice.",
    "Call out tradeoffs when latency, memory, and quality pull in different directions.",
    "Focus on coding and debugging work in repositories, terminals, and local tools.",
    "Preserve existing behavior unless the request explicitly asks for a change.",
    "When a workflow is fragile, describe the safest exact command to run.",
]

BASE_TASK = (
    "We are working in a mono-repo with Python, TypeScript, shell scripts, "
    "JSON settings, and local AI tooling. The agent should inspect files, "
    "reason about performance regressions, compare configuration options, "
    "and produce a short recommendation with exact next steps."
)

FINAL_QUESTION = (
    "Given the repository context above, summarize the three biggest performance "
    "risks you would investigate first and give one concrete tuning suggestion "
    "for each."
)


def build_messages(target_blocks: int):
    repeated = []
    for i in range(target_blocks):
        repeated.append(
            f"Context block {i + 1}: "
            + BASE_TASK
            + " "
            + " ".join(BASE_RULES)
        )
    user = "\n\n".join(repeated) + "\n\n" + FINAL_QUESTION
    return [
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_prompt(processor, model_config, target_tokens: int):
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    blocks = 1
    prompt = ""

    while True:
        messages = build_messages(blocks)
        prompt = apply_chat_template(processor, model_config, messages)
        token_count = len(tokenizer.encode(prompt))
        if token_count >= target_tokens:
            return prompt, token_count, blocks
        blocks += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark one mlx-vlm config.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--kv-bits", type=float, default=None)
    parser.add_argument(
        "--kv-quant-scheme",
        choices=("uniform", "turboquant"),
        default="uniform",
    )
    parser.add_argument("--quantized-kv-start", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()

    mx.clear_cache()
    gc.collect()

    load_start = time.perf_counter()
    model, processor = load(args.model)
    load_seconds = time.perf_counter() - load_start

    prompt, prompt_token_estimate, blocks = build_prompt(
        processor, model.config, args.target_tokens
    )

    run_start = time.perf_counter()
    result = generate(
        model,
        processor,
        prompt,
        max_tokens=args.max_tokens,
        temperature=0.0,
        prefill_step_size=args.prefill_step_size,
        kv_bits=args.kv_bits,
        kv_quant_scheme=args.kv_quant_scheme,
        quantized_kv_start=args.quantized_kv_start,
        verbose=False,
    )
    run_seconds = time.perf_counter() - run_start

    payload = {
        "label": args.label,
        "model": args.model,
        "target_tokens": args.target_tokens,
        "prompt_token_estimate": prompt_token_estimate,
        "blocks": blocks,
        "load_seconds": round(load_seconds, 3),
        "run_seconds": round(run_seconds, 3),
        "prompt_tokens": result.prompt_tokens,
        "generation_tokens": result.generation_tokens,
        "prompt_tps": round(result.prompt_tps, 3),
        "generation_tps": round(result.generation_tps, 3),
        "peak_memory_gb": round(result.peak_memory, 3),
        "prefill_step_size": args.prefill_step_size,
        "kv_bits": args.kv_bits,
        "kv_quant_scheme": args.kv_quant_scheme,
        "quantized_kv_start": args.quantized_kv_start,
        "output_preview": result.text[:240],
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
