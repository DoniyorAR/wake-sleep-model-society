"""Shared figure style: fonts, colours and sizes (single text column, print)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# categorical slots 1-3 of the validated palette + neutral for the reference line
C_NAIVE, C_DOC, C_GATE, C_ORACLE = "#eb6834", "#2a78d6", "#1baf7a", "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W_IN = 3.25


def serif_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"):
        if cand in names:
            return cand
    return "serif"


def apply_style():
    plt.rcParams.update({"font.family": serif_font(), "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "savefig.dpi": 400, "figure.dpi": 150})


def sans_font():
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
