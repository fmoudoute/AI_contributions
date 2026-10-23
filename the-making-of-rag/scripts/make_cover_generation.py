# Cover image for Part 5: the "faithful but wrong" example, quoted from generation_demo.py output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, BG, CARD, ACCENT, WARN = "#0b0b0b", "#52514e", "#fcfcfb", "#f1f0ec", "#2a78d6", "#eb6834"
fig = plt.figure(figsize=(14, 7.875), dpi=100, facecolor=BG)
fig.text(0.06, 0.80, "THE MAKING OF RAG  ·  PART 5", fontsize=15, color=MUTED, family="DejaVu Sans", weight="bold")
fig.text(0.06, 0.62, "Generation", fontsize=64, color=INK, family="DejaVu Serif", weight="bold")
fig.text(0.06, 0.53, "the last mile", fontsize=26, color=MUTED, family="DejaVu Serif", style="italic")
fig.text(0.06, 0.30, "qwen2.5:7b  ·  running locally with Ollama", fontsize=13, color=MUTED, family="DejaVu Sans Mono")


def card(y, height, label, label_color):
    fig.patches.append(FancyBboxPatch((0.55, y), 0.40, height, boxstyle="round,pad=0.006,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=CARD, edgecolor="none"))
    fig.text(0.57, y + height - 0.045, label, fontsize=12, color=label_color, family="DejaVu Sans", weight="bold")


MONO = dict(fontsize=14, color=INK, family="DejaVu Sans Mono")
CHAR = 14 * 0.6 / (14 * 72)  # width of one monospace character, in figure coordinates
card(0.52, 0.30, "CONTEXT  ·  chunk 20", MUTED)
fig.text(0.57, 0.705, "...", **MONO)
fig.text(0.57 + 4 * CHAR, 0.705, "In China", fontsize=14, color=WARN, family="DejaVu Sans Mono", weight="bold")
fig.text(0.57 + 12 * CHAR, 0.705, ", we added 2,531", **MONO)
fig.text(0.57, 0.655, "net new stores during the quarter,", **MONO)
fig.text(0.57, 0.605, "bringing our total store count", **MONO)
fig.text(0.57, 0.555, "to 33,419.", **MONO)

card(0.20, 0.27, "ANSWER", MUTED)
fig.text(0.57, 0.355, "The company operates a total of", **MONO)
fig.text(0.57, 0.305, "33,419 stores [...] [chunk 20]", **MONO)
fig.text(0.57, 0.235, "cited ✓   grounded ✓", fontsize=13, color=MUTED, family="DejaVu Sans Mono", weight="bold")
fig.text(0.57 + 23 * CHAR * 13 / 14, 0.235, "correct ✗", fontsize=13, color=WARN, family="DejaVu Sans Mono", weight="bold")

fig.text(0.96, 0.03, "Luckin Coffee Q1 2026 earnings call · the real total is 33,596", fontsize=10, color=MUTED,
         ha="right", family="DejaVu Sans")
fig.savefig("../generation_cover.png", facecolor=BG)
