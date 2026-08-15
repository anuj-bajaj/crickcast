"""
Shared matplotlib styling for every chart the pipeline generates —
Phase 3's feature importance, Phase 4's calibration curve, Phase 5's
event-impact chart. Same palette as the frontend
(frontend/src/index.css's --color-brand-* custom properties), so the
diagnostics section of the site doesn't look like three generic
matplotlib exports sitting next to a designed product.

Import and call apply_brand_style() once, before creating a figure —
it's global rcParams state, cheap to call repeatedly, no return value.
"""

import matplotlib as mpl

PINE = "#1b3a2b"
PINE_LIGHT = "#2a4f3a"
PINE_DARK = "#12261c"
PARCHMENT = "#f4efe1"
PARCHMENT_DIM = "#e8e1cd"
CARD = "#fbf8f0"
WICKET = "#8c2f39"
MOSS = "#3f6b4d"
AMBER = "#d97706"
YELLOW = "#f2c230"


def apply_brand_style():
    mpl.rcParams.update({
        "figure.facecolor": CARD,
        "axes.facecolor": CARD,
        "savefig.facecolor": CARD,
        "axes.edgecolor": PINE,
        "axes.linewidth": 1.1,
        "axes.labelcolor": PINE,
        "text.color": PINE,
        "xtick.color": PINE,
        "ytick.color": PINE,
        "axes.grid": True,
        "grid.color": PINE,
        "grid.alpha": 0.10,
        "grid.linewidth": 0.8,
        "font.family": "serif",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })
