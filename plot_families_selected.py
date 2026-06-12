"""
Regenerate families_all_strict_vs_ratio.png for a single run,
keeping only a chosen subset of instruction families.

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
    "length_constraints",
    "keywords",
    "punctuation",
    "startend",
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
    plt.figure()
    for fam in FAMILIES:
        ys = fam_strict[fam]
        if np.all(np.isnan(ys)):
            continue
        plt.plot(xs, ys, marker="o", label=fam)
    plt.xlabel("Compression ratio")
    plt.ylabel("Strict")
    plt.title("All Families (Strict) vs Compression Ratio")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()

    xs, fam_strict = load_results(args.run_dir)
    print(f"Loaded {len(xs)} ratios: {xs.tolist()}")

    out_path = args.run_dir / "plots" / "families_all_strict_vs_ratio.png"
    plot(xs, fam_strict, out_path)


if __name__ == "__main__":
    main()
