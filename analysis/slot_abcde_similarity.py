"""
Slot-prototype cosine similarity, grouped by ABCDEF category.

Produces two figures:
  * a full T x T heatmap, slots sorted by (category, name) with category
    dividers  ->  <output_dir>/01_refined_structured_heatmap.png
  * a zoomed, annotated heatmap of a slice of one category
    ->  <output_dir>/01_clinical_zoom_<category>.png

    python analysis/slot_abcde_similarity.py \
        --features_dir slot_features --concepts_dict concepts_dict.json \
        --output_dir figures --zoom_category B --zoom_start 2 --zoom_end 12

Prototypes are the per-slot mean over samples, zero-centred across slots
(removes the shared "global background"); disable with --no_center.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import add_features_dir_arg, ensure_dir, load_json, load_slot_features


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_features_dir_arg(p)
    p.add_argument("--concepts_dict", required=True, help="JSON: disease name -> ABCDEF letter.")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--no_center", action="store_true", help="Skip zero-centring the prototypes.")
    p.add_argument("--vmin", type=float, default=-0.4)
    p.add_argument("--vmax", type=float, default=0.8)
    p.add_argument("--zoom_category", default="B", help="Category shown in the zoomed heatmap.")
    p.add_argument("--zoom_start", type=int, default=2, help="First slot (within the category).")
    p.add_argument("--zoom_end", type=int, default=12, help="End slot (exclusive).")
    p.add_argument("--no_zoom", action="store_true")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def main():
    args = parse_args()
    slots, _, names = load_slot_features(args.features_dir, load_labels=False)
    categories = load_json(args.concepts_dict)
    out_dir = ensure_dir(args.output_dir)

    # 1. sort slots by (category, name); unmapped names -> "Other"
    order = pd.DataFrame({"name": names, "index": range(len(names))})
    order["cat"] = order["name"].map(categories).fillna("Other")
    order = order.sort_values(by=["cat", "name"]).reset_index(drop=True)
    ordered_idx = order["index"].values

    # 2. prototypes + cosine similarity (rows/cols are in *sorted* order)
    protos = slots.mean(axis=0)[ordered_idx]
    if not args.no_center:
        protos = protos - protos.mean(axis=0)
    unit = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    sim = unit @ unit.T
    cbar = "Cosine Similarity" + ("" if args.no_center else " (Zero-Centered)")

    # 3. zoomed category view — positions refer to the SORTED order, matching `sim`
    if not args.no_zoom:
        pos = np.flatnonzero(order["cat"].values == args.zoom_category)[args.zoom_start:args.zoom_end]
        if len(pos) == 0:
            print(f"Warning: no slots for category '{args.zoom_category}' in slice "
                  f"[{args.zoom_start}:{args.zoom_end}] — skipping zoom plot.")
        else:
            sub_names = order["name"].values[pos]
            plt.figure(figsize=(10, 8), dpi=args.dpi)
            sns.heatmap(sim[np.ix_(pos, pos)], cmap="RdBu_r", vmin=args.vmin, vmax=args.vmax,
                        center=0, xticklabels=sub_names, yticklabels=sub_names,
                        annot=True, fmt=".2f", cbar_kws={"label": "Cosine Similarity"})
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            out = out_dir / f"01_clinical_zoom_{args.zoom_category}.png"
            plt.savefig(out, dpi=args.dpi)
            plt.close()
            print(f"Saved -> {out}")

    # 4. full structured heatmap with category dividers
    plt.figure(figsize=(15, 12), dpi=args.dpi)
    sns.heatmap(sim, cmap="RdBu_r", vmin=args.vmin, vmax=args.vmax, center=0,
                xticklabels=False, yticklabels=False, cbar_kws={"label": cbar})
    start = 0
    for cat in sorted(order["cat"].unique()):
        count = int((order["cat"] == cat).sum())
        plt.text(start + count / 2, -0.012 * len(order), f"Cat {cat}",
                 ha="center", fontweight="bold", fontsize=16)
        plt.axvline(start + count, color="black", linewidth=1.5, alpha=0.6)
        plt.axhline(start + count, color="black", linewidth=1.5, alpha=0.6)
        start += count

    out = out_dir / "01_refined_structured_heatmap.png"
    plt.savefig(out, bbox_inches="tight", dpi=args.dpi)
    plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()