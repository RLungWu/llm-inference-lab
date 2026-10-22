"""Compare two inference benchmark result files (e.g. baseline vs. quantized).

Usage:
    python benchmark/compare.py results/baseline_XXXX.json results/4bit_XXXX.json
"""

import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict:
    """Load a benchmark JSON file produced by inference.py.

    Return the parsed dict — it has "metadata" and "results" keys
    (see inference.py's `output` dict for the exact shape).
    """
    # TODO
    raise NotImplementedError


def index_by_context_length(data: dict) -> dict[int, dict]:
    """Turn data["results"] (a list of per-context-length entries) into a
    dict keyed by prompt_tokens, so callers can do e.g. `by_ctx[512]` to
    get straight to that context length's "avg" stats dict.

    Each value in the returned dict should be the "avg" sub-dict, not the
    whole entry (we don't need the raw per-trial data for this).
    """
    # TODO
    raise NotImplementedError


def diff_by_context_length(baseline: dict, candidate: dict) -> list[dict]:
    """For every prompt_tokens value present in BOTH baseline and candidate,
    compute the relative change and return one row per context length,
    sorted by prompt_tokens ascending:

        {
            "prompt_tokens": 512,
            "prompt_tps_speedup": candidate.prompt_tps / baseline.prompt_tps,
            "generation_tps_speedup": candidate.generation_tps / baseline.generation_tps,
            "peak_memory_ratio": candidate.peak_memory_gb / baseline.peak_memory_gb,
        }

    speedup > 1 means candidate is faster; peak_memory_ratio < 1 means
    candidate uses less memory.
    """
    # TODO — use index_by_context_length() on both baseline and candidate
    raise NotImplementedError


def print_table(rows: list[dict]) -> None:
    """Pretty-print the comparison rows as an aligned table to stdout.

    Columns: prompt_tokens, prompt_tps_speedup, generation_tps_speedup, peak_memory_ratio
    (2 decimal places is plenty, e.g. "1.83x").
    """
    # TODO
    raise NotImplementedError


def main():
    if len(sys.argv) != 3:
        print("Usage: python benchmark/compare.py <baseline.json> <candidate.json>")
        sys.exit(1)

    baseline = load_results(Path(sys.argv[1]))
    candidate = load_results(Path(sys.argv[2]))

    print(f"baseline:  {baseline['metadata']['label']} ({baseline['metadata']['model_path']})")
    print(f"candidate: {candidate['metadata']['label']} ({candidate['metadata']['model_path']})")

    rows = diff_by_context_length(baseline, candidate)
    print_table(rows)


if __name__ == "__main__":
    main()
