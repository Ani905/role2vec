"""How good are the embeddings? Train a simple classifier to predict each
paper's topic from its embedding, and report the accuracy.

Run after src/main.py:
    python src/evaluate.py
    python src/evaluate.py --embedding output/some_other_embedding.csv
"""

import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

TARGET_PATH = "input/cora_target.csv"


def evaluate(embedding_path, target_path=TARGET_PATH):
    """Return (accuracy, macro_f1) from 5-fold cross-validation."""
    emb = pd.read_csv(embedding_path)
    emb["id"] = emb["id"].astype(int)
    data = emb.merge(pd.read_csv(target_path), on="id")
    X = data.drop(columns=["id", "target"]).values
    y = data["target"].values

    # 5-fold cross-validation: split papers into 5 parts, train on 4,
    # test on the 5th, repeat 5 times, and average. Every paper gets
    # tested exactly once, so the score isn't luck of a single split.
    model = LogisticRegression(max_iter=2000)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=folds, scoring="accuracy").mean()
    f1 = cross_val_score(model, X, y, cv=folds, scoring="f1_macro").mean()
    return acc, f1


def majority_baseline(target_path=TARGET_PATH):
    """Accuracy of always guessing the most common topic."""
    y = pd.read_csv(target_path)["target"]
    return y.value_counts(normalize=True).max()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="output/cora_role2vec.csv")
    args = parser.parse_args()

    acc, f1 = evaluate(args.embedding)
    print("Embedding:          " + args.embedding)
    print("Accuracy:           {:.1%}".format(acc))
    print("Macro F1:           {:.3f}".format(f1))
    print("Always-guess-most-common baseline: {:.1%}".format(majority_baseline()))