"""
Shared helpers for the scripts in analysis/.

Importing this module also puts the repository root on ``sys.path`` so the
scripts can import project packages 
when launched as ``python analysis/<script>.py`` from any directory.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; every script saves to disk
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FEATURES_DIR = "slot_features"

# label encoding used throughout: 1 = positive, 0 = uncertain, -1 = negative
CLASS_STYLE = {
    0.0:  dict(name="Uncertain", color="#c8c8c8", s=12, alpha=0.30, edgecolors="none", linewidths=0.0),
    1.0:  dict(name="Positive",  color="#e63946", s=22, alpha=0.75, edgecolors="white", linewidths=0.3),
    -1.0: dict(name="Negative",  color="#1d6fa4", s=22, alpha=0.75, edgecolors="white", linewidths=0.3),
}


# ── I/O ──────────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_slot_features(features_dir, load_labels: bool = True):
    """Load the output of save_slots.py.

    Returns (slots [N,T,D], labels [N,T] | None, disease_names list[str]).
    """
    d = Path(features_dir)
    slots = np.load(d / "slot_features.npy")
    labels = np.load(d / "labels.npy") if load_labels else None
    names = load_json(d / "disease_names.json")
    if slots.shape[1] != len(names):
        raise ValueError(f"{slots.shape[1]} slots but {len(names)} disease names in {d}")
    print(f"Loaded slots {slots.shape} from {d}")
    return slots, labels, names


def add_features_dir_arg(parser):
    parser.add_argument("--features_dir", default=DEFAULT_FEATURES_DIR,
                        help="Directory written by save_slots.py "
                             "(slot_features.npy, labels.npy, disease_names.json).")


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── device ───────────────────────────────────────────────────────────────────
def get_device(requested: str | None = None):
    """'auto'/None -> cuda, else hpu (if Habana is installed), else cpu."""
    import torch

    if requested in (None, "", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import habana_frameworks.torch.core  # noqa: F401
            return torch.device("hpu")
        except ImportError:
            return torch.device("cpu")
    if requested.startswith("hpu"):
        import habana_frameworks.torch.core  # noqa: F401
    return torch.device(requested)


# ── pathology selection ──────────────────────────────────────────────────────
def select_top_pathologies(labels: np.ndarray, top_n: int) -> list[int]:
    """Indices of the top_n pathologies ranked by their smallest class count
    (pos / uncertain / neg) — i.e. the most balanced ones first."""
    pos = (labels == 1.0).sum(0)
    unc = (labels == 0.0).sum(0)
    neg = (labels == -1.0).sum(0)
    min_count = np.minimum(np.minimum(pos, unc), neg)
    return np.argsort(-min_count, kind="stable")[:top_n].tolist()


# ── t-SNE class-separation panels ────────────────────────────────────────────
def resolve_perplexity(value, n_samples: int) -> float:
    """None -> sqrt(N); always clamped below n_samples (sklearn requirement)."""
    p = int(np.sqrt(n_samples)) if value is None else value
    return float(max(2, min(p, n_samples - 1)))


def plot_class_scatter(ax, feat, lbl, title, perplexity, seed=42,
                       show_ylabel=True, title_size=13, count_size=8):
    """t-SNE of one slot's features, coloured by positive/negative/uncertain."""
    from sklearn.manifold import TSNE

    ax.set_facecolor("#fafafa")
    tsne = TSNE(n_components=2, perplexity=resolve_perplexity(perplexity, len(feat)),
                init="pca", learning_rate="auto", random_state=seed, n_jobs=-1)
    xy = tsne.fit_transform(feat)

    for val in (0.0, 1.0, -1.0):  # uncertain first (background)
        st = dict(CLASS_STYLE[val])
        st.pop("name")
        m = lbl == val
        ax.scatter(xy[m, 0], xy[m, 1], **st)

    ax.text(0.03, 0.97,
            f"pos={int((lbl == 1).sum())}  neg={int((lbl == -1).sum())}  unc={int((lbl == 0).sum())}",
            transform=ax.transAxes, fontsize=count_size, va="top", color="#444444",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7, ec="none"))
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10, labelpad=6)
    ax.set_ylabel("t-SNE Dimension 2" if show_ylabel else "", fontsize=10, labelpad=6)
    ax.tick_params(axis="both", which="both", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4, color="grey")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
        sp.set_linewidth(0.8)
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=10)


def make_grid(n_plots: int, ncols: int, cell_w: float, cell_h: float, dpi: int):
    """Grid just big enough for n_plots; spare axes are hidden."""
    ncols = max(1, min(ncols, n_plots))
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, cell_h * nrows),
                             dpi=dpi, squeeze=False)
    fig.patch.set_facecolor("white")
    flat = axes.flatten()
    for ax in flat[n_plots:]:
        ax.set_visible(False)
    return fig, flat, ncols


def add_class_legend(fig, fontsize=13, y=0.01):
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=9,
                      markerfacecolor=CLASS_STYLE[v]["color"], markeredgecolor="none",
                      label=CLASS_STYLE[v]["name"]) for v in (0.0, 1.0, -1.0)]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, y),
               fontsize=fontsize, frameon=True, edgecolor="#aaaaaa")