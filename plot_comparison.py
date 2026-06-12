"""
Comparison plots: PHARAOH vs DuoAttention vs PyramidKV

Plots produced:
  1. prompt_strict_vs_ratio — headline all-or-nothing accuracy per prompt
  2. instruction_strict_vs_ratio — per-instruction strict accuracy per method

Usage (from repo root):
    python plot_comparison.py
    python plot_comparison.py --out_dir my/out
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent / "pitfalls-of-kv-cache-compression/kv_cache_compression/experiments/compressed_context_ifeval_outputs"

FIVE82_DIR = BASE / "llama_3.2_1b_instruct_five82_20260611-17-45-14"
DUO_DIR = BASE / "llama_3.2_1b_instruct_duo_attention_on_the_fly_20260611-21-51-44"
PYRAMID_DIR = BASE / "llama_3.2_1b_instruct_pyramid_20260612-08-33-13"

METHODS = {
    "PHARAOH": FIVE82_DIR,
    "DuoAttention": DUO_DIR,
    "PyramidKV": PYRAMID_DIR,
}


def load_results(exp_dir: Path) -> dict[float, dict]:
    """Return {ratio: ifeval_results_dict} sorted by ratio."""
    results = {}
    for ratio_dir in sorted(exp_dir.glob("responses/ratio_*")):
        ratio = float(ratio_dir.name.split("_", 1)[1])
        result_file = ratio_dir / "ifeval_results.json"
        if result_file.exists():
            with open(result_file) as f:
                results[ratio] = json.load(f)
    return dict(sorted(results.items()))


def plot_prompt_strict_vs_ratio(
    all_results: dict[str, dict[float, dict]], out_dir: Path
) -> None:
    """
    Plot 1: prompt_strict (all-or-nothing per prompt) vs compression ratio,
    one line per method.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = {"PHARAOH": "#c99b38", "DuoAttention": "#ff5a5e", "PyramidKV": "#00b0be"}

    for method, results in all_results.items():
        xs = np.array(list(results.keys()))
        ys = np.array([r["scores"]["prompt_strict"] for r in results.values()])
        errs = np.array([r["scores"].get("overall_err_strict", 0.0) for r in results.values()])

        # Plot central line
        ax.plot(
            xs, ys,
            label=method,
            color=colors[method],
            linewidth=2.0,
        )
        # Plot error shading instead of error bars
        ax.fill_between(
            xs, ys - errs, ys + errs,
            color=colors[method],
            alpha=0.15,
        )

    ax.set_xlabel("Compression Ratio (fraction of KV cache evicted)", fontsize=11)
    ax.set_ylabel("Prompt-Level Strict Accuracy", fontsize=11)
    #ax.set_title("Prompt-Strict Accuracy vs Compression Ratio\n(all instructions must pass)", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.02, 0.92)

    # Remove top, left, and right spines
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)

    # Light grey horizontal grid lines
    ax.grid(True, axis='y', color='lightgrey', linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(fontsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prompt_strict_vs_ratio_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=600, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_instruction_strict_vs_ratio(
    all_results: dict[str, dict[float, dict]], out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = {"PHARAOH": "#c99b38", "DuoAttention": "#ff5a5e", "PyramidKV": "#00b0be"}

    for method, results in all_results.items():
        xs = np.array(list(results.keys()))
        ys = np.array([r["scores"]["instruction_strict"] for r in results.values()])
        errs = np.array([r["scores"].get("overall_err_strict", 0.0) for r in results.values()])

        ax.plot(xs, ys, label=method, color=colors[method], linewidth=2.0)
        ax.fill_between(xs, ys - errs, ys + errs, color=colors[method], alpha=0.15)

    ax.set_xlabel("Compression Ratio (fraction of KV cache evicted)", fontsize=11)
    ax.set_ylabel("Instruction-Level Strict Accuracy", fontsize=11)
    #ax.set_title("Instruction-Strict Accuracy vs Compression Ratio", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.02, 0.92)

    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)

    ax.grid(True, axis='y', color='lightgrey', linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(fontsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "instruction_strict_vs_ratio_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=600, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=BASE / "comparison_plots",
        help="Directory to write output PDFs",
    )
    args = parser.parse_args()

    all_results = {method: load_results(path) for method, path in METHODS.items()}

    for method, results in all_results.items():
        if not results:
            raise FileNotFoundError(f"No ifeval_results.json files found under {METHODS[method]}")
        print(f"{method}: loaded {len(results)} ratios ({sorted(results.keys())})")

    plot_prompt_strict_vs_ratio(all_results, args.out_dir)
    plot_instruction_strict_vs_ratio(all_results, args.out_dir)


if __name__ == "__main__":
    main()