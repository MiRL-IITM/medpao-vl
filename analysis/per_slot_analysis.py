"""
Single-figure t-SNE class-separation grid for the most balanced pathologies.

For each of the top-N pathologies (ranked by smallest class count) the slot
features of every sample are embedded with t-SNE and coloured by
positive / negative / uncertain.

    python analysis/per_slot_analysis.py --features_dir slot_features \
        --top_n 10 --ncols 5 --output 03_pathology_separation_final.png

For N > 10 pathologies split over several images use getting_all_tsne_plots.py.
"""
import argparse

import matplotlib.pyplot as plt

from common import (add_class_legend, add_features_dir_arg, ensure_parent,
                    load_slot_features, make_grid, plot_class_scatter,
                    select_top_pathologies)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_features_dir_arg(p)
    p.add_argument("--output", default="03_pathology_separation_final.png")
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--ncols", type=int, default=5)
    p.add_argument("--perplexity", type=float, default=None,
                   help="t-SNE perplexity. Default: sqrt(N samples).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--title", default="Latent Manifold Analysis: Pathology-Specific Class Separation")
    p.add_argument("--no_title", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    slots, labels, names = load_slot_features(args.features_dir)

    selected = select_top_pathologies(labels, args.top_n)
    print(f"Plotting {len(selected)} pathologies: {[names[i] for i in selected]}")

    fig, axes, ncols = make_grid(len(selected), args.ncols, 6, 7, args.dpi)
    for i, t in enumerate(selected):
        plot_class_scatter(axes[i], slots[:, t, :], labels[:, t], names[t],
                           args.perplexity, args.seed, show_ylabel=(i % ncols == 0))

    add_class_legend(fig)
    if not args.no_title:
        fig.suptitle(args.title, fontsize=20, y=1.01, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])

    out = ensure_parent(args.output)
    fig.savefig(out, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()