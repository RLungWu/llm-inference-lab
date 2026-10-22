import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_averages(result_path: Path) -> list[dict]:
    """Load the average measurements from one inference result file."""
    with result_path.open() as f:
        data = json.load(f)

    return [item["avg"] for item in data["results"]]


def plot_metric(
    contexts: list[int],
    values: list[float],
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Plot one metric against context length and save it as a PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(contexts, values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_xticks(contexts)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MLX inference benchmark results")
    parser.add_argument("result_path", type=Path, help="Path to a benchmark JSON file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "plots" / "mlx",
        help="Directory for generated PNG files",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_averages(args.result_path)
    contexts = [int(row["prompt_tokens"]) for row in rows]
    prompt_tps = [row["prompt_tps"] for row in rows]
    generation_tps = [row["generation_tps"] for row in rows]

    plot_metric(
        contexts,
        prompt_tps,
        "Prompt Throughput vs Context Length",
        "prompt_throughput.png",
        output_dir,
    )
    plot_metric(
        contexts,
        generation_tps,
        "Generation Throughput vs Context Length",
        "generation_throughput.png",
        output_dir,
    )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
