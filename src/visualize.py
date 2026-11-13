"""Plot Role2Vec embeddings in 2D, coloured by Cora paper topic.

Run after src/main.py:
    python src/visualize.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

EMBEDDING_PATH = "output/cora_role2vec.csv"
TARGET_PATH = "input/cora_target.csv"
PLOT_PATH = "output/cora_role2vec_tsne.png"

# Load the 128-dim embeddings and the topic label of each paper
emb = pd.read_csv(EMBEDDING_PATH)
emb["id"] = emb["id"].astype(int)
labels = pd.read_csv(TARGET_PATH)
data = emb.merge(labels, on="id")

X = data.drop(columns=["id", "target"]).values
y = data["target"].values

# Squash 128 dimensions down to 2 so we can draw them.
# t-SNE keeps points that were close in 128-D close in 2-D.
print("Running t-SNE (takes ~30 seconds)...")
X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)

plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=6, alpha=0.8)
plt.legend(*scatter.legend_elements(), title="Topic", loc="best", fontsize=8)
plt.title("Role2Vec embeddings of Cora papers (t-SNE)")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print("Saved plot to " + PLOT_PATH)
plt.show()