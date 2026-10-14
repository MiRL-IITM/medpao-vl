"""
Global t-SNE of ALL slot vectors (N samples x T slots), coloured by ABCDEF
category.

Pipeline: flatten -> PCA -> class-balanced subsample -> t-SNE (cosine).

    python analysis/only_tsne_slots.py \
        --features_dir slot_features \
        --concepts_dict concepts_dict.json \
        --output 00_slot_tsne.png

`concepts_dict.json` maps a disease name to its category letter, e.g.
{"pleural effusion": "B", "pacemaker": "F", ...}. Unmapped names -> "Other".
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common import add_features_dir_arg, ensure_parent, load_json, load_slot_features

PALETTE = {"A": "#e41a1c", "B": "#377eb8", "C": "#ff7f00", "D": "#4daf4a",
           "E": "#984ea3", "F": "#a65628", "Other": "#bbbbbb"}
CAT_LABELS = {"A": "A: Airways", "B": "B: Breathing", "C": "C: Circulation",
              "D": "D: Diaphragm", "E": "E: External", "F": "F: Foreign", "Other": "Other"}
# drawn bottom -> top; rare classes last so they stay visible
DRAW_ORDER = ["Other", "B", "C", "E", "F", "D", "A"]
SIZE = {"A": 40, "B": 8, "C": 14, "D": 18, "E": 14, "F": 14, "Other": 6}
ALPHA = {"A": 1.0, "B": 0.25, "C": 0.5, "D": 0.6, "E": 0.5, "F": 0.55, "Other": 0.15}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_features_dir_arg(p)
    p.add_argument("--concepts_dict", required=True, help="JSON: disease name -> ABCDEF letter.")
    p.add_argument("--output", default="00_slot_tsne.png")
    p.add_argument("--pca_dims", type=int, default=200)
    p.add_argument("--max_per_class", type=int, default=4000,
                   help="Cap points per category so a large class (B) doesn't dominate.")
    p.add_argument("--perplexity", type=float, default=160)
    p.add_argument("--max_iter", type=int, default=800)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--annotate_centroids", action="store_true",
                   help="Write the category letter at each category centroid.")
    p.add_argument("--title", default=None)
    return p.parse_args()


def run_tsne(x, args):
    kw = dict(n_components=2, perplexity=args.perplexity, metric=args.metric,
              init="pca", random_state=args.seed, n_jobs=-1)
    try:
        return TSNE(max_iter=args.max_iter, **kw).fit_transform(x)
    except TypeError:                       # scikit-learn < 1.5
        return TSNE(n_iter=args.max_iter, **kw).fit_transform(x)


def main():
    args = parse_args()
    slots, _, names = load_slot_features(args.features_dir, load_labels=False)
    categories = load_json(args.concepts_dict)

    N, T, D = slots.shape
    vectors = slots.reshape(-1, D)
    slot_ids = np.tile(np.arange(T), N)
    slot_names = [names[i] for i in slot_ids]
    slot_cats = np.array([categories.get(n, "Other") for n in slot_names])

    reduced = PCA(n_components=min(args.pca_dims, D, len(vectors)),
                  random_state=args.seed).fit_transform(vectors)

    # class-balanced subsample
    rng = np.random.default_rng(args.seed)
    keep = np.concatenate([
        rng.choice(np.flatnonzero(slot_cats == c),
                   size=min(args.max_per_class, int((slot_cats == c).sum())), replace=False)
        for c in np.unique(slot_cats)
    ])
    reduced, cats = reduced[keep], slot_cats[keep]
    print(f"Points: {len(reduced)} | {pd.Series(cats).value_counts().to_dict()}")

    print("Running t-SNE...")
    xy = run_tsne(reduced, args)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=args.dpi)
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")

    for cat in DRAW_ORDER:
        m = cats == cat
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], c=PALETTE[cat], s=SIZE[cat], alpha=ALPHA[cat],
                   linewidth=0, label=CAT_LABELS[cat], rasterized=True)
        if args.annotate_centroids and cat != "Other":
            ax.annotate(cat, xy=(xy[m, 0].mean(), xy[m, 1].mean()), fontsize=11,
                        fontweight="bold", color=PALETTE[cat], ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE[cat],
                                  alpha=0.85, linewidth=1.2))

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12, labelpad=8)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12, labelpad=8)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
        sp.set_linewidth(0.8)
    ax.legend(title="ABCDEF Category", title_fontsize=10, fontsize=9, loc="upper left",
              framealpha=0.9, edgecolor="#cccccc", markerscale=2.0)
    if args.title:
        ax.set_title(args.title, fontsize=14, fontweight="bold", pad=14)

    fig.tight_layout()
    out = ensure_parent(args.output)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()