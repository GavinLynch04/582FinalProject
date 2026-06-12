"""
Regenerate families_all_strict_vs_ratio.svg for a single run,
keeping all instruction families.

Usage (from repo root):
    python plot_families_selected.py
    python plot_families_selected.py --run_dir path/to/run
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent / "pitfalls-of-kv-cache-compression/kv_cache_compression/experiments/compressed_context_ifeval_outputs"

DEFAULT_RUN = BASE / "llama_3.2_1b_instruct_five82_20260611-22-56-13"

FAMILIES = [
    "detectable_format",
    "keywords",
    "length_constraints",
    "punctuation",
]


def load_results(run_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Returns:
      xs: sorted compression ratios as float array
      fam_strict: {family: np.array of strict accuracy per ratio}
    """
    ratio_dirs = sorted(
        run_dir.glob("responses/ratio_*"),
        key=lambda p: float(p.name.split("_", 1)[1]),
    )
    if not ratio_dirs:
        raise FileNotFoundError(f"No ratio_* directories found under {run_dir}/responses/")

    xs = []
    rows = []
    for d in ratio_dirs:
        result_file = d / "ifeval_results.json"
        if not result_file.exists():
            continue
        xs.append(float(d.name.split("_", 1)[1]))
        with open(result_file) as f:
            rows.append(json.load(f))

    xs = np.array(xs, dtype=float)

    fam_strict = {}
    for fam in FAMILIES:
        vals = []
        for row in rows:
            rec = (row.get("by_family") or {}).get(fam)
            vals.append(rec["strict"] if rec is not None else np.nan)
        fam_strict[fam] = np.array(vals, dtype=float)

    return xs, fam_strict


def plot(xs: np.ndarray, fam_strict: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    color_map = {
        "detectable_format": "#4a2377",  # Purple
        "keywords": "#f55f74",  # Pink
        "length_constraints": "#8cc5e3",  # Blue
        "punctuation": "#0d7d87"  # Teal
    }

    for fam in FAMILIES:
        ys = fam_strict[fam]
        if np.all(np.isnan(ys)):
            continue

        color = color_map.get(fam, "black")

        ax.plot(xs, ys, label=fam, color=color, linewidth=2)

    ax.set_xlabel("Compression Ratio (fraction of KV cache evicted)", fontsize=11)
    ax.set_ylabel("Strict Accuracy", fontsize=11)
    #ax.set_title("All Families (Strict) vs Compression Ratio", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.02, 0.92)

    # Remove top, left, and right spines
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)

    # Light grey horizontal grid lines only
    ax.grid(True, axis='y', color='lightgrey', linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)

    # Legend moved inside the plot axes bounding box at the top right
    ax.legend(title="Family", loc="upper right", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()

    xs, fam_strict = load_results(args.run_dir)
    print(f"Loaded {len(xs)} ratios: {xs.tolist()}")

    # Output tracking adapted to match .svg format modification
    out_path = args.run_dir / "plots" / "families_all_strict_vs_ratio.pdf"
    plot(xs, fam_strict, out_path)


if __name__ == "__main__":
    main()