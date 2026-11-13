"""Experiment: which notion of 'structural role' works best?

Trains Role2Vec three times, once with each feature type, evaluates
each, and prints a results table (also saved to output/comparison.csv).

    python src/compare.py
"""

import sys
import subprocess
import pandas as pd
from evaluate import evaluate, majority_baseline

FEATURES = {
    "degree": "Node degree (number of connections)",
    "wl": "Weisfeiler-Lehman labels (neighbourhood shape)",
    "motif": "Graphlet/motif counts (small-subgraph patterns)",
}

rows = []
for feature, description in FEATURES.items():
    out_path = "output/cora_role2vec_{}.csv".format(feature)
    print("\n=== Training with --features {} ===".format(feature))
    subprocess.run(
        [sys.executable, "src/main.py", "--features", feature, "--output", out_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    acc, f1 = evaluate(out_path)
    print("Accuracy {:.1%}, Macro F1 {:.3f}".format(acc, f1))
    rows.append({"Features": feature, "Description": description,
                 "Accuracy": round(acc, 3), "Macro F1": round(f1, 3)})

rows.append({"Features": "baseline", "Description": "Always guess the most common topic",
             "Accuracy": round(majority_baseline(), 3), "Macro F1": None})

table = pd.DataFrame(rows)
table.to_csv("output/comparison.csv", index=False)
print("\n" + table.to_string(index=False))
print("\nSaved to output/comparison.csv")