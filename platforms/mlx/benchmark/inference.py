"""Baseline inference benchmark.

Measures prefill latency, decode throughput, total latency, and peak memory
across a set of fixed context lengths — no kernel changes, just a
reproducible baseline.
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

# benchmark/inference.py -> mlx -> platforms -> llm-inference-lab -> model
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "model"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONTEXT_LENGTHS = [128, 512, 1024, 2048]
GENERATION_TOKENS = 128
NUM_TRIALS = 3


def run_single(model, tokenizer, vocab_size: int, prompt_tokens: int, generation_tokens: int, prefill_step_size: int):
    """Run one generation and return throughput and memory stats."""
    # Measure peak memory independently for this run. MLX otherwise reports
    # the high-water mark since process start or the previous reset.
    mx.reset_peak_memory()
    prompt = mx.random.randint(0, vocab_size, (prompt_tokens,)).tolist()

    tic = time.perf_counter()
    response = None
    for response in stream_generate(model, tokenizer, prompt, max_tokens=generation_tokens, prefill_step_size=prefill_step_size):
        pass
    toc = time.perf_counter()

    return {
        "prompt_tokens": response.prompt_tokens,
        "prompt_tps": response.prompt_tps,
        "generation_tokens": response.generation_tokens,
        "generation_tps": response.generation_tps,
        "peak_memory_gb": response.peak_memory,
        "active_memory_gb": mx.get_active_memory() / 1e9,
        "allocator_cache_memory_gb": mx.get_cache_memory() / 1e9,
        "total_time_s": toc - tic,
    }


def benchmark_context_length(
    model, tokenizer, vocab_size: int, prompt_tokens: int, generation_tokens: int, num_trials: int = 3, prefill_step_size: int=2048
):
    """Warm up once, then run num_trials timed trials and average them."""
    print(f"[prompt_tokens={prompt_tokens}] warmup...")
    run_single(model, tokenizer, vocab_size, prompt_tokens, generation_tokens, prefill_step_size=prefill_step_size)

    trials = []
    for i in range(num_trials):
        stats = run_single(model, tokenizer, vocab_size, prompt_tokens, generation_tokens, prefill_step_size=prefill_step_size)
        print(f"[prompt_tokens={prompt_tokens}] trial {i + 1}/{num_trials}: {stats}")
        trials.append(stats)

    avg = {key: sum(t[key] for t in trials) / num_trials for key in trials[0]}
    return {"prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens, "trials": trials, "avg": avg}


def collect_environment_metadata() -> dict:
    """Collect reproducibility metadata available from the current platform."""
    return {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "device": str(mx.default_device()),
        "device_info": mx.device_info(),
    }


def main():
    parser = argparse.ArgumentParser(description="Inference benchmark")
    parser.add_argument(
        "--model-path", type=str, default=str(MODEL_PATH), help="Path to the model directory to benchmark"
    )
    parser.add_argument(
        "--label",
        type=str,
        default="baseline",
        help="Short label for this run, used in the output filename (e.g. 'baseline', '4bit')",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
#        help="Short label for this run, used in the output filename (e.g. 'baseline', '4bit')",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=CONTEXT_LENGTHS,
#        help="Short label for this run, used in the output filename (e.g. 'baseline', '4bit')",
    )
    args = parser.parse_args()

    mx.random.seed(0)

    print(f"Loading model from {args.model_path} ...")
    model, tokenizer, config = load(args.model_path, return_config=True)
    print("Model loaded.")

    # Random prompts can accidentally sample an eos token; clear it so
    # generation always runs for the full requested length.
    tokenizer._eos_token_ids = {}
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]

    results = []
    for prompt_tokens in args.context_lengths:
        result = benchmark_context_length(
            model, tokenizer, vocab_size, prompt_tokens, GENERATION_TOKENS, num_trials=NUM_TRIALS, prefill_step_size=args.prefill_step_size
        )
        results.append(result)
        print(f"[prompt_tokens={prompt_tokens}] avg: {result['avg']}")

    output = {
        "metadata": {
            "model_path": args.model_path,
            "label": args.label,
            "mlx_version": version("mlx"),
            "mlx_lm_version": version("mlx-lm"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_trials": NUM_TRIALS,
            "environment": collect_environment_metadata(),
            "prefill_step_size": args.prefill_step_size
        },
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}")

    return results


if __name__ == "__main__":
    main()
