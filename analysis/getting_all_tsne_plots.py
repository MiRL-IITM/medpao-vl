"""
t-SNE class-separation plots for the top-N pathologies, split over several
images (BATCH_SIZE plots per image).

    python analysis/getting_all_tsne_plots.py --features_dir slot_features \
        --top_n 30 --batch_size 10 --output_dir figures/tsne_batches
"""
import argparse
import math

import matplotlib.pyplot as plt
from tqdm import tqdm

from common import (add_class_legend, add_features_dir_arg, ensure_dir,
                    load_slot_features, make_grid, plot_class_scatter,
                    select_top_pathologies)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_features_dir_arg(p)
    p.add_argument("--output_dir", default=".")
    p.add_argument("--prefix", default="03_pathology_separation_batch")
    p.add_argument("--top_n", type=int, default=10, help="Number of pathologies to visualise.")
    p.add_argument("--batch_size", type=int, default=10, help="Plots per image.")
    p.add_argument("--ncols", type=int, default=5)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    slots, labels, names = load_slot_features(args.features_dir)
    out_dir = ensure_dir(args.output_dir)

    selected = select_top_pathologies(labels, args.top_n)
    print(f"Total pathologies selected: {len(selected)}")
    n_batches = math.ceil(len(selected) / args.batch_size)

    for b in tqdm(range(n_batches), desc="Batches"):
        idxs = selected[b * args.batch_size:(b + 1) * args.batch_size]
        fig, axes, ncols = make_grid(len(idxs), args.ncols, 6, 7, args.dpi)

        for i, t in enumerate(idxs):
            plot_class_scatter(axes[i], slots[:, t, :], labels[:, t], names[t],
                               args.perplexity, args.seed, show_ylabel=(i % ncols == 0),
                               count_size=7.5)

        add_class_legend(fig)
        fig.suptitle("Latent Manifold Analysis: Pathology-Specific Class Separation "
                     f"(Batch {b + 1}/{n_batches})", fontsize=20, y=1.01, fontweight="bold")
        fig.tight_layout(rect=[0, 0.06, 1, 0.98])

        out = out_dir / f"{args.prefix}{b + 1:02d}.png"
        fig.savefig(out, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)  # free memory before the next batch
        print(f"  Saved: {out}  ({len(idxs)} plots)")

    print("Done.")


if __name__ == "__main__":
    main()