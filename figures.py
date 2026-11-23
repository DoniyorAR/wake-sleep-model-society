"""Paper figures (print, single text column = 80 mm wide).

usage: python figures.py results/<model_tag> out_dir
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

# categorical slots 1-3 of the validated palette + neutral for the reference line
C_NAIVE, C_DOC, C_GATE, C_ORACLE = "#eb6834", "#2a78d6", "#1baf7a", "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W_IN = 3.25


def _font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"):
        if cand in names:
            return cand
    return "serif"


def _style():
    plt.rcParams.update({"font.family": _font(), "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "savefig.dpi": 400, "figure.dpi": 150})


def _sans():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Lato", "Carlito", "Calibri", "Liberation Sans", "Arial", "Nimbus Sans", "DejaVu Sans"):
        if cand in names:
            return cand
    return "sans-serif"


# soft panel fill and dark chip colour per role in Fig. 1
P_GATE = ("#cfe7db", "#1a7a57")
P_BLOCK = ("#deddd9", "#8d8c88")
P_CLAIM = ("#ddd3ee", "#7650b0")
P_FACT = ("#cbdff4", "#2867ae")
P_CORE = ("#f4f3ef", "#55534f")
P_REH = ("#f1d2cc", "#b85a52")


def schematic(out):
    """Fig. 1: (a) the two-model write gate during the day, (b) the trust layers written at night."""
    _style()
    font = _sans()
    x_lo, x_hi, y_lo, y_hi = 0.0, 100.0, 1.2, 89.6
    fig = plt.figure(figsize=(W_IN, W_IN * (y_hi - y_lo) / (x_hi - x_lo)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.axis("off")

    def panel(x, y, w, h, color, r=3.0, z=1, edge="none", lw=0.0):
        ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}",
                                    fc=color, ec=edge, lw=lw, zorder=z))

    def label(x, y, text, size=6.0, weight="normal", ha="center", color=INK, z=7):
        ax.text(x, y, text, ha=ha, va="center", fontsize=size, weight=weight, color=color,
                family=font, linespacing=1.08, zorder=z)

    def chip(x, y, w, text, color, h=4.8, size=6.3):
        panel(x, y, w, h, color, r=1.2, z=3)
        label(x + w / 2, y + h / 2 - 0.15, text, size=size, weight="bold", color="white")

    def box(x, y, w, h, color, title, lines):
        panel(x, y, w, h, color, r=1.6, z=2)
        top = y + h / 2 + 1.5 * len(lines)
        label(x + w / 2, top, title, size=6.3, weight="bold")
        for i, line in enumerate(lines):
            label(x + w / 2, top - 3.5 - 2.95 * i, line, size=5.8)

    def thin_arrow(p, q, rad=0.0, style="-|>"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=5.5, color=INK, lw=0.8,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=6))

    def line(xs, ys):
        ax.plot(xs, ys, color=INK, lw=0.8, solid_capstyle="round", zorder=6)

    def block_arrow(p, q, color, w=4.0, hw=8.4, hl=4.6):
        (x0, y0), (x1, y1) = p, q
        d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        nx, ny = -uy, ux
        bx, by = x1 - hl * ux, y1 - hl * uy
        pts = [(x0 + nx * w / 2, y0 + ny * w / 2), (bx + nx * w / 2, by + ny * w / 2),
               (bx + nx * hw / 2, by + ny * hw / 2), (x1, y1), (bx - nx * hw / 2, by - ny * hw / 2),
               (bx - nx * w / 2, by - ny * w / 2), (x0 - nx * w / 2, y0 - ny * w / 2)]
        ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=2))

    def doc_icon(cx, cy, w=5.6, h=7.0, c=1.8):
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        ax.add_patch(Polygon([(x0, y0), (x1, y0), (x1, y1 - c), (x1 - c, y1), (x0, y1)], closed=True,
                             fc="white", ec=INK, lw=0.9, joinstyle="round", zorder=5))
        ax.add_patch(Polygon([(x1 - c, y1), (x1 - c, y1 - c), (x1, y1 - c)], closed=False, fc="none",
                             ec=INK, lw=0.9, joinstyle="round", zorder=6))
        for k in range(4):
            yy = y0 + 1.25 + k * 1.2
            ax.plot([x0 + 1.05, x1 - 1.05 - (1.5 if k == 3 else 0.0)], [yy, yy], color=INK, lw=0.7,
                    solid_capstyle="round", zorder=6)

    def bubble_icon(cx, cy, w=7.0, h=4.6):
        x0, y0 = cx - w / 2, cy - h / 2 + 0.9
        panel(x0, y0, w, h, "white", r=1.3, z=5, edge=INK, lw=0.9)
        tip = (x0 + 1.1, y0 - 2.0)
        ax.add_patch(Polygon([(x0 + 1.5, y0 + 0.3), tip, (x0 + 3.5, y0 + 0.3)], closed=True, fc="white",
                             ec="none", zorder=6))
        ax.plot([x0 + 1.5, tip[0], x0 + 3.5], [y0, tip[1], y0], color=INK, lw=0.9,
                solid_capstyle="round", solid_joinstyle="round", zorder=7)
        for k in (-1, 0, 1):
            ax.add_patch(Circle((cx + 1.55 * k, y0 + h / 2), 0.42, fc=INK, ec="none", zorder=7))

    def net_icon(cx, cy, layers, dx=2.9, dy=1.95, r=0.62):
        pos = []
        for i, n in enumerate(layers):
            xx = cx + (i - (len(layers) - 1) / 2) * dx
            pos.append([(xx, cy + (j - (n - 1) / 2) * dy) for j in range(n)])
        for la, lb in zip(pos[:-1], pos[1:]):
            for pa in la:
                for pb in lb:
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=INK, lw=0.35, zorder=5)
        for layer in pos:
            for pt in layer:
                ax.add_patch(Circle(pt, r, fc=INK, ec="none", zorder=6))

    def moon_icon(cx, cy, r=2.2):
        ax.add_patch(Circle((cx, cy), r, fc=INK, ec="none", zorder=5))
        ax.add_patch(Circle((cx + 0.95, cy + 0.7), r * 0.84, fc="white", ec="none", zorder=6))

    # ---------------------------------------------------------------- a: wake
    label(0.2, 87.0, "a", size=9.5, weight="bold", ha="left")
    label(5.0, 86.8, "Wake: the model society gates every write", size=6.8, weight="bold", ha="left")
    doc_icon(8.0, 74.6)
    label(8.0, 68.8, "document")
    bubble_icon(8.0, 59.4)
    label(8.0, 53.6, "user remark")
    line([12.4, 17.0, 17.0, 13.0], [74.6, 74.6, 60.2, 60.2])
    thin_arrow((17.0, 67.4), (21.5, 67.4))

    panel(22.0, 50.0, 44.0, 33.0, P_GATE[0])
    chip(24.5, 76.8, 23.5, "sleeping model", P_GATE[1], h=4.6)
    chip(24.5, 51.6, 20.5, "helper model", P_GATE[1], h=4.6)
    rows = ((71.1, (3, 4, 3)), (61.9, (4, 3, 4)))
    for yy, layers in rows:
        net_icon(29.3, yy, layers)
        thin_arrow((34.2, yy), (38.6, yy))
        panel(39.4, yy - 2.2, 12.0, 4.4, "white", r=1.2, z=3, edge=P_GATE[1], lw=0.7)
        label(45.4, yy - 0.1, "support", size=6.0)
    thin_arrow((52.6, rows[0][0]), (52.6, rows[1][0]), rad=-0.55, style="<|-|>")
    label(60.1, 66.5, "both\nneeded", size=6.0)

    block_arrow((67.4, 66.5), (76.8, 66.5), P_BLOCK[1])
    label(72.0, 59.2, "not\nwritten", size=5.8)
    box(78.0, 67.25, 22.0, 15.75, P_BLOCK[0], "rejected", ["not supported", "by both models"])
    box(78.0, 50.0, 22.0, 15.75, P_BLOCK[0], "quarantine", ["conflict with", "known facts"])

    # ---------------------------------------------------------------- b: sleep
    label(0.2, 44.3, "b", size=9.5, weight="bold", ha="left")
    label(5.0, 44.1, "Sleep: memory in trust layers", size=6.8, weight="bold", ha="left")
    block_arrow((57.0, 49.3), (57.0, 39.3), "#3d9f7b")
    moon_icon(65.3, 44.6)
    label(69.2, 46.0, "sleep", size=6.3, weight="bold", ha="left")
    label(69.2, 42.9, "LoRA update", size=5.8, ha="left")

    panel(0.0, 2.0, 65.0, 36.5, P_CLAIM[0])
    chip(2.5, 32.3, 19.0, "user claims", P_CLAIM[1], h=4.6)
    label(23.3, 34.5, "source tag kept", size=5.8, ha="left")
    thin_arrow((60.0, 36.2), (60.0, 26.9))
    label(58.2, 35.5, "confirmed by", size=5.8, ha="right")
    label(58.2, 32.5, "a document", size=5.8, ha="right")
    panel(3.5, 4.3, 58.5, 25.0, P_FACT[0])
    chip(6.0, 23.1, 11.5, "facts", P_FACT[1], h=4.6)
    label(19.3, 25.3, "stated by a document", size=5.8, ha="left")
    panel(7.0, 6.6, 51.5, 14.0, P_CORE[0])
    chip(16.5, 14.3, 32.5, "pretrained knowledge", P_CORE[1], h=4.6)
    label(32.75, 10.2, "held in place by rehearsal", size=5.8)

    block_arrow((76.8, 20.25), (67.4, 20.25), P_REH[1])
    label(72.0, 13.5, "rehearsal", size=5.8)
    box(78.0, 27.4, 22.0, 11.1, P_REH[0], "replay", ["of earlier", "memories"])
    box(78.0, 14.7, 22.0, 11.1, P_REH[0], "refusals", ["for unknown", "people"])
    box(78.0, 2.0, 22.0, 11.1, P_REH[0], "anchors", ["the model's", "own answers"])
    fig.savefig(out)
    plt.close(fig)


SERIES = [("naive", "Naive sleep", C_NAIVE, "o"), ("doc_only", "Document-only", C_DOC, "s"),
          ("wsms", "WSMS (ours)", C_GATE, "^"), ("oracle", "Oracle", C_ORACLE, "D")]
PANELS = [("recall", "Recall of true facts (%)"), ("false_assert", "False assertions (%)")]


def curves(models, out):
    """models: list of (title, summary). Rows = metrics, columns = models."""
    _style()
    ncol = len(models)
    fig, axes = plt.subplots(2, ncol, figsize=(W_IN, 2.6), sharex=True, sharey="row", squeeze=False)
    for ci, (title, summary) in enumerate(models):
        conds = summary["conditions"]
        for ri, (key, ylabel) in enumerate(PANELS):
            ax = axes[ri][ci]
            for name, label, color, marker in SERIES:
                if name not in conds:
                    continue
                cur = conds[name]["curve"][key]
                x = conds[name]["curve_nights"]
                mean = [100 * v[0] for v in cur]
                sd = [100 * v[1] for v in cur]
                ax.fill_between(x, [m - d for m, d in zip(mean, sd)], [m + d for m, d in zip(mean, sd)],
                                color=color, alpha=0.15, lw=0)
                ax.plot(x, mean, color=color, lw=1.4, marker=marker, ms=3.4, mec="white", mew=0.6,
                        label=label, solid_capstyle="round", ls="-" if name != "oracle" else (0, (4, 2)))
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=7.5)
            if ri == 0:
                ax.set_title(title, fontsize=7.5, pad=3)
            if ri == 1:
                ax.set_xlabel("Night", fontsize=7.5)
                ax.set_xlim(1.5, 31)
                ax.set_xticks([6, 12, 18, 24, 30])
            ax.tick_params(labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=6.6, handlelength=1.8,
               columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.54, 1.005))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.885, bottom=0.115, hspace=0.14, wspace=0.10)
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    schematic(os.path.join(out_dir, "fig1_wsms.png"))
    models = []
    for tag, title in (("qwen2.5-1.5b", "Qwen2.5-1.5B"), ("qwen3-4b", "Qwen3-4B")):
        path = os.path.join(root, tag, "summary.json")
        if os.path.exists(path):
            models.append((title, json.load(open(path))))
    if models:
        curves(models, os.path.join(out_dir, "fig2_curves.png"))
    print("figures written to", out_dir, "| font:", _font())
