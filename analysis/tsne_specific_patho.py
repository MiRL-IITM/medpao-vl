"""
t-SNE class-separation grid for a hand-picked list of pathologies
(e.g. the clinically interesting ones used in the paper figure).

    python analysis/tsne_specific_patho.py --features_dir slot_features \
        --pathologies "pulmonary edema" "bilateral pleural effusions" "low lung volumes" \
        --rename "right chest tube in place without pneumothorax=chest tube" \
        --ncols 4 --output clinical_pathology_separation.png

Names are matched exactly against disease_names.json; if there is no exact
match a unique case-insensitive substring match is used.
"""
import argparse

import matplotlib.pyplot as plt

from common import (add_class_legend, add_features_dir_arg, ensure_parent,
                    load_slot_features, make_grid, plot_class_scatter)

DEFAULT_PATHOLOGIES = [
    "pulmonary edema",
    "unchanged PICC line",
    "right chest tube in place without pneumothorax",
    "dual lead left-sided pacer device",
    "calcified aortic knob",
    "bilateral pleural effusions",
    "status post surgery",
    "low lung volumes",
]
DEFAULT_RENAMES = ["right chest tube in place without pneumothorax=chest tube"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_features_dir_arg(p)
    p.add_argument("--output", default="clinical_pathology_separation.png")
    p.add_argument("--pathologies", nargs="+", default=DEFAULT_PATHOLOGIES,
                   help="Pathology names to plot.")
    p.add_argument("--rename", nargs="*", default=DEFAULT_RENAMES, metavar="OLD=NEW",
                   help="Display-name overrides for subplot titles.")
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--perplexity", type=float, default=70.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--title", default=None, help="Optional figure title.")
    return p.parse_args()


def resolve_index(query, names):
    if query in names:
        return names.index(query)
    hits = [i for i, n in enumerate(names) if query.lower() in n.lower()]
    if len(hits) == 1:
        print(f"  '{query}' -> '{names[hits[0]]}' (substring match)")
        return hits[0]
    print(f"Warning: '{query}' "
          f"{'is ambiguous (' + str(len(hits)) + ' matches)' if hits else 'not found'} — skipped")
    return None


def main():
    args = parse_args()
    slots, labels, names = load_slot_features(args.features_dir)
    renames = dict(r.split("=", 1) for r in args.rename)

    selected = [i for i in (resolve_index(q, names) for q in args.pathologies) if i is not None]
    if not selected:
        raise SystemExit("None of the requested pathologies were found.")
    print(f"Total pathologies found and selected: {len(selected)}")

    fig, axes, ncols = make_grid(len(selected), args.ncols, 7, 8, args.dpi)
    for i, t in enumerate(selected):
        plot_class_scatter(axes[i], slots[:, t, :], labels[:, t],
                           renames.get(names[t], names[t]), args.perplexity, args.seed,
                           show_ylabel=(i % ncols == 0), title_size=14)

    add_class_legend(fig, fontsize=14)
    if args.title:
        fig.suptitle(args.title, fontsize=22, y=1.02, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])

    out = ensure_parent(args.output)
    fig.savefig(out, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()