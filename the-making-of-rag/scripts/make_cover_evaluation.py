# Cover image for Part 6: failure attribution, copied from the output of eval_end_to_end.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = {  # model: (OK, generation failed, retrieval failed), on the 12 answerable questions
    "LFM2.5 1.2B": (7, 4, 1),
    "llama3.1 8B": (10, 1, 1),
    "qwen2.5 7B": (11, 0, 1),
}
SEGMENTS = [("correct", "#2a78d6"), ("generation failed", "#eb6834"), ("retrieval failed", "#1baf7a")]
INK, MUTED, BG = "#0b0b0b", "#52514e", "#fcfcfb"

fig = plt.figure(figsize=(14, 7.875), dpi=100, facecolor=BG)
fig.text(0.06, 0.80, "THE MAKING OF RAG  ·  PART 6", fontsize=15, color=MUTED, family="DejaVu Sans", weight="bold")
fig.text(0.06, 0.62, "Evaluation", fontsize=64, color=INK, family="DejaVu Serif", weight="bold")
fig.text(0.06, 0.53, "who judges the judge?", fontsize=26, color=MUTED, family="DejaVu Serif", style="italic")
fig.text(0.06, 0.33, "string match vs LLM judge:", fontsize=14, color=INK, family="DejaVu Sans Mono")
fig.text(0.06, 0.28, "4 disagreements in 48 answers", fontsize=14, color=INK, family="DejaVu Sans Mono")
fig.text(0.06, 0.23, "the judge was right once", fontsize=14, color=MUTED, family="DejaVu Sans Mono")

ax = fig.add_axes([0.60, 0.30, 0.34, 0.42], facecolor=BG)
names = list(RESULTS)
for row, name in enumerate(names):
    left = 0
    for (label, color), value in zip(SEGMENTS, RESULTS[name]):
        if value:
            ax.barh(row, value - 0.08, left=left + 0.04, height=0.55, color=color)  # small gap between segments
            ax.text(left + value / 2, row, str(value), ha="center", va="center", fontsize=14,
                    color="white" if label == "correct" else INK, family="DejaVu Sans", weight="bold")
        left += value
ax.set_yticks(range(len(names)), names, fontsize=15, color=INK)
ax.tick_params(axis="y", length=0)
ax.set_xticks([])
ax.set_xlim(0, 12)
for spine in ax.spines.values():
    spine.set_visible(False)

fig.text(0.60, 0.80, "Where the failures come from (12 questions)", fontsize=12, color=MUTED, family="DejaVu Sans")
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in SEGMENTS]
fig.legend(handles, [label for label, _ in SEGMENTS], loc="lower left", bbox_to_anchor=(0.595, 0.19), ncol=3,
           frameon=False, fontsize=12, handlelength=1, handleheight=1, columnspacing=1.6)
fig.text(0.96, 0.03, "Luckin Coffee Q1 2026 earnings call · hybrid search + re-rank, K=3 · local models via Ollama",
         fontsize=10, color=MUTED, ha="right", family="DejaVu Sans")
fig.savefig("../evaluation_cover.png", facecolor=BG)
