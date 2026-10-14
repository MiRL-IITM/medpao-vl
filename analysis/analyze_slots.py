"""
Inter-slot cosine-similarity heatmap (quick disentanglement check).

Averages the slot features over all samples to get one prototype per
pathology, then plots the T x T cosine-similarity matrix.

    python analysis/analyze_slots.py --features_dir slot_features \
        --output figures/01_slot_similarity_heatmap.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common import add_features_dir_arg, ensure_parent, load_slot_features


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_features_dir_arg(p)
    p.add_argument("--output", default="01_slot_similarity_heatmap.png")
    p.add_argument("--cmap", default="magma")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    slots, _, names = load_slot_features(args.features_dir, load_labels=False)

    prototypes = slots.mean(axis=0)                                   # [T, D]
    unit = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    sim = unit @ unit.T

    plt.figure(figsize=(12, 10))
    sns.heatmap(sim, cmap=args.cmap, xticklabels=False, yticklabels=False)
    plt.title(f"Inter-Slot Cosine Similarity ({len(names)} slots)")
    out = ensure_parent(args.output)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()