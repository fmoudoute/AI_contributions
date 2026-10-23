# Cover image for Part 4, drawn from the real evaluation results of retrieval_eval.py
import contextlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import retrieval_eval as r

methods = {
    "Keyword (BM25)": r.bm25.scores,
    "Semantic": r.semantic_scores,
    "Hybrid (RRF)": lambda q: r.rrf(r.bm25.scores(q), r.semantic_scores(q)),
    "Hybrid + re-rank": lambda q: r.rerank(q, r.rrf(r.bm25.scores(q), r.semantic_scores(q))),
}
with contextlib.redirect_stdout(io.StringIO()):
    ranks = {name: r.evaluate(name, fn) for name, fn in methods.items()}
mrr = {name: sum(1 / x for x in rk) / len(rk) for name, rk in ranks.items()}
print({k: round(v, 2) for k, v in mrr.items()})

INK, MUTED, BG, ACCENT, GREY = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#c9c8c3"
fig = plt.figure(figsize=(14, 7.875), dpi=100, facecolor=BG)
fig.text(0.06, 0.80, "THE MAKING OF RAG  ·  PART 4", fontsize=15, color=MUTED, family="DejaVu Sans", weight="bold")
fig.text(0.06, 0.62, "Retrieval", fontsize=64, color=INK, family="DejaVu Serif", weight="bold")
fig.text(0.06, 0.53, "finding the needle", fontsize=26, color=MUTED, family="DejaVu Serif", style="italic")
fig.text(0.06, 0.33, "RRF(chunk) = Σ 1 / (60 + rank)", fontsize=15, color=INK, family="DejaVu Sans Mono")
fig.text(0.06, 0.27, "keyword + semantic, fused by rank", fontsize=13, color=MUTED, family="DejaVu Sans Mono")

ax = fig.add_axes([0.56, 0.22, 0.36, 0.56], facecolor=BG)
names = list(mrr)[::-1]
values = [mrr[n] for n in names]
colors = [ACCENT if "Hybrid" in n else GREY for n in names]
ax.barh(names, values, height=0.5, color=colors)
for i, v in enumerate(values):
    ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=15, color=INK, family="DejaVu Sans")
ax.set_xlim(0, 1.05)
ax.tick_params(axis="y", length=0, labelsize=15, labelcolor=INK)
ax.set_xticks([])
for s in ax.spines.values():
    s.set_visible(False)
fig.text(0.56, 0.82, "Mean Reciprocal Rank on 12 real questions (1.0 = always first)",
         fontsize=12, color=MUTED, family="DejaVu Sans")
fig.text(0.96, 0.03, "Luckin Coffee Q1 2026 earnings call · 107 chunks", fontsize=10, color=MUTED,
         ha="right", family="DejaVu Sans")
fig.savefig("../retrieval_cover.png", facecolor=BG)
