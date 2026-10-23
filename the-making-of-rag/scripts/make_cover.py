import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from embedding_pretrained import EmbeddingModel

groups = {
    "business": ["revenue", "sales", "profit margin", "quarterly earnings", "net income",
                 "stores", "shops", "coffee chain"],
    "football": ["goal", "striker", "stadium", "football match", "goalkeeper", "penalty"],
}
colors = {"business": "#2a78d6", "football": "#eb6834"}
query = "How many shops do they have?"

model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", max_tokens=256)
words = [w for g in groups.values() for w in g]
V = model.embed(words + [query])
# 2D projection with plain SVD (PCA), like in the article
C = V - V.mean(axis=0)
_, _, Vt = np.linalg.svd(C, full_matrices=False)
P = C @ Vt[:2].T
q2, qv = P[-1], V[-1]
sims = V[:-1] @ qv
top = np.argsort(sims)[::-1][:3]
print([(words[i], round(float(sims[i]), 2)) for i in top])

INK, MUTED, BG = "#0b0b0b", "#52514e", "#fcfcfb"
fig = plt.figure(figsize=(14, 7.875), dpi=100, facecolor=BG)
# left: text
fig.text(0.06, 0.80, "THE MAKING OF RAG  ·  PART 3", fontsize=15, color=MUTED, family="DejaVu Sans", weight="bold")
fig.text(0.06, 0.62, "Embedding", fontsize=64, color=INK, family="DejaVu Serif", weight="bold")
fig.text(0.06, 0.53, "the geometry of meaning", fontsize=26, color=MUTED, family="DejaVu Serif", style="italic")
rev = V[words.index("revenue")][:4]; sal = V[words.index("sales")][:4]
fmt = lambda v: "[" + ", ".join(f"{x:+.3f}" for x in v) + ", …]"
fig.text(0.06, 0.33, f'"revenue" → {fmt(rev)}', fontsize=13, color=INK, family="DejaVu Sans Mono")
fig.text(0.06, 0.28, f'"sales"   → {fmt(sal)}', fontsize=13, color=INK, family="DejaVu Sans Mono")
fig.text(0.06, 0.20, f"cosine(revenue, sales) = {float(V[words.index('revenue')] @ V[words.index('sales')]):.2f}",
         fontsize=13, color=MUTED, family="DejaVu Sans Mono")

# right: the map
OFFSETS = {"goal": (-9, 0, "right"), "goalkeeper": (9, 2, "left"), "football match": (-9, 0, "right"),
           "striker": (9, -4, "left"), "profit margin": (9, 4, "left"), "quarterly earnings": (9, -10, "left"),
           "net income": (-9, 0, "right"), "stores": (9, 6, "left"), "shops": (9, -8, "left")}
ax = fig.add_axes([0.53, 0.10, 0.40, 0.80], facecolor=BG)
for i in top:
    ax.plot([q2[0], P[i, 0]], [q2[1], P[i, 1]], color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
for g, ws in groups.items():
    for w in ws:
        i = words.index(w)
        ax.scatter(*P[i], s=90, color=colors[g], edgecolor=BG, linewidth=2, zorder=3)
        dx, dy, ha = OFFSETS.get(w, (8, 4, "left"))
        ax.annotate(w, P[i], xytext=(dx, dy), textcoords="offset points", fontsize=13, color=INK,
                    family="DejaVu Sans", ha=ha, va="center")
ax.scatter(*q2, s=170, marker="D", color=INK, edgecolor=BG, linewidth=2, zorder=4)
ax.annotate(f'"{query}"', q2, xytext=(-14, 0), textcoords="offset points", fontsize=12,
            color=INK, family="DejaVu Sans", style="italic", ha="right", va="center")
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.margins(0.18)
fig.text(0.96, 0.03, "384-dim all-MiniLM-L6-v2 vectors, projected to 2D with SVD", fontsize=10, color=MUTED, ha="right", family="DejaVu Sans")
fig.savefig("../embeddings_cover.png", facecolor=BG)
