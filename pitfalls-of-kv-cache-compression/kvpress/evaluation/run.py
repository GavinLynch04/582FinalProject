import os
import subprocess
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. Configuration
# ==========================================
# Add your custom press name to this list once it's in evaluate_registry.py
PRESSES = ["five", "pyramidkv", "duo_attention"]
COMPRESSION_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

DATASET = "ruler"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./results"

# ==========================================
# 2. Run Evaluations
# ==========================================
for press in PRESSES:
    for ratio in COMPRESSION_RATIOS:
        print(f"Running evaluation: {press} at ratio {ratio}")

        # Build the CLI command to trigger evaluate.py
        command = [
            "python", "evaluate.py",
            "--dataset", DATASET,
            "--model", MODEL,
            "--data_dir", "4096",
            "--fraction", "0.01",
            "--press_name", press,
            "--compression_ratio", str(ratio),
            "--output_dir", OUTPUT_DIR
        ]


        # Execute the evaluation script
        subprocess.run(command, check=True)

print("\nAll evaluations complete! Parsing results for plotting...")

# ==========================================
# 3. Parse Results & Extract Metrics
# ==========================================
data_points = []

# Walk through the output directory to collect metrics
results_path = Path(OUTPUT_DIR)
for config_dir in results_path.iterdir():
    if not config_dir.is_dir():
        continue

    config_file = config_dir / "config.yaml"
    metrics_file = config_dir / "metrics.json"

    if config_file.exists() and metrics_file.exists():
        try:
            # Read metadata from config
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)

            # Read score from metrics
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)

            # Dynamic extraction based on RULER or other benchmark structures
            # Adjust the key if your scorer outputs a different primary metric name
            accuracy = metrics_data.get("accuracy", metrics_data.get("avg", None))
            if accuracy is None:
                # Fallback to the first numeric value found if key is non-standard
                numeric_values = [v for v in metrics_data.values() if isinstance(v, (int, float))]
                accuracy = numeric_values[0] if numeric_values else 0.0

            data_points.append({
                "press_name": config_data.get("press_name"),
                "compression_ratio": float(config_data.get("compression_ratio")),
                "accuracy": float(accuracy)
            })
        except Exception as e:
            print(f"Skipping directory {config_dir.name} due to parsing error: {e}")

# Convert parsed data into a DataFrame
df_results = pd.DataFrame(data_points)

# Filter down to the specific benchmark/run we just did
df_results = df_results[df_results["press_name"].isin(PRESSES)]

# ==========================================
# 4. Generate the Compression vs. Accuracy Plot
# ==========================================
plt.figure(figsize=(10, 6))

for press in PRESSES:
    # Filter and sort data points to ensure the line plots correctly
    df_press = df_results[df_results["press_name"] == press].sort_values(by="compression_ratio")

    if not df_press.empty:
        plt.plot(
            df_press["compression_ratio"],
            df_press["accuracy"],
            marker='o',
            linewidth=2,
            label=press
        )

plt.title(f"KV Cache Compression Ratio vs. Accuracy ({DATASET.upper()})", fontsize=14, fontweight='bold')
plt.xlabel("Compression Ratio (Lower = Higher Compression)", fontsize=12)
plt.ylabel("Accuracy / Metric Score", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)  # Adjust if your metric doesn't scale 0-1

# Save the final figure
plot_output = "./compression_vs_accuracy.png"
plt.savefig(plot_output, dpi=300, bbox_inches="tight")
print(f"\nSuccess! Performance curve saved to: {plot_output}")
plt.show()