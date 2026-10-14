"""
Polar slot-similarity plot
==========================
Shows the angular relationship between one *reference* pathology slot and
several comparison slots for a given sample. Vectors are projected onto the
plane where the reference lies on the +x axis and unit-normalised.

  cool colours (purple/blue)  : angle < 90°  (positively aligned)
  warm colours (orange/red)   : angle > 90°  (opposed)

Labels are spread around the circle so that no two are closer than
--min_sep degrees.

    python analysis/plot_slot_vectors.py --features_dir slot_features \
        --reference "pleural effusion" --compare "pneumonia" "atelectasis" "edema" \
        --sample_idx 0 --output polar_sim.png

    # one figure per sample for the first 20 test samples
    python analysis/plot_slot_vectors.py --reference ... --compare ... \
        --batch_n 20 --output_dir polar_plots/

Names are matched by case-insensitive substring against disease_names.json.
"""
import argparse
import textwrap

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

from common import (add_features_dir_arg, ensure_dir, ensure_parent,
                    load_slot_features)

COOL = ["#7B2FBE", "#1565C0"]   # purple, deep blue
WARM = ["#F2994A", "#C62828"]   # orange, deep red
RADIAN_LABELS = {0: "0", 30: "π/6", 60: "π/3", 90: "π/2", 120: "2π/3", 150: "5π/6",
                 180: "π", 210: "7π/6", 240: "4π/3", 270: "3π/2", 300: "5π/3", 330: "11π/6"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_features_dir_arg(p)
    p.add_argument("--reference", required=True, help="Reference pathology (substring match).")
    p.add_argument("--compare", nargs="+", required=True, help="Comparison pathologies.")
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--batch_n", type=int, default=None,
                   help="If set, plot samples 0..batch_n-1 into --output_dir.")
    p.add_argument("--output", default="polar_sim.png", help="Output file (single-sample mode).")
    p.add_argument("--output_dir", default=".", help="Output directory (--batch_n mode).")
    p.add_argument("--min_sep", type=float, default=8.0, help="Min. angular gap between labels (deg).")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# ── maths ────────────────────────────────────────────────────────────────────
def find_idx(names, query):
    q = query.lower()
    for i, n in enumerate(names):
        if q in n.lower():
            return i
    raise ValueError(f"'{query}' not found. First 10 names: {names[:10]}")


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(cos_sim(a, b), -1, 1))))


def project_2d(ref, others):
    """Project onto the plane spanned by `ref` and the first non-collinear
    comparison vector; ref -> +x. Returns unit 2-D vectors (angles preserved)."""
    u = ref / (np.linalg.norm(ref) + 1e-8)
    ortho = None
    for v in others:
        vn = v / (np.linalg.norm(v) + 1e-8)
        perp = vn - np.dot(vn, u) * u
        if np.linalg.norm(perp) > 1e-6:
            ortho = perp / np.linalg.norm(perp)
            break
    if ortho is None:
        ortho = np.zeros_like(u)
        ortho[1] = 1.0

    def to2d(v):
        vn = v / (np.linalg.norm(v) + 1e-8)
        x, y = float(np.dot(vn, u)), float(np.dot(vn, ortho))
        m = np.hypot(x, y)
        return np.array([x / (m + 1e-8), y / (m + 1e-8)])

    return to2d(ref), [to2d(v) for v in others]


def pick_color(adeg, cool_used, warm_used):
    if adeg < 90:
        return COOL[cool_used % len(COOL)], cool_used + 1, warm_used
    return WARM[warm_used % len(WARM)], cool_used, warm_used + 1


# ── label placement ──────────────────────────────────────────────────────────
def _angular_dist(a, b):
    d = abs(a % 360 - b % 360)
    return min(d, 360 - d)


def resolve_overlaps(angles, min_sep, max_iters=200):
    """Push label angles apart until every pair is >= min_sep degrees apart."""
    angles = [a % 360 for a in angles]
    step = min_sep * 0.5
    for _ in range(max_iters):
        moved = False
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                if _angular_dist(angles[i], angles[j]) < min_sep - 1e-6:
                    ccw = (angles[j] - angles[i]) % 360 <= 180
                    s = 1 if ccw else -1
                    angles[i] = (angles[i] - s * step) % 360
                    angles[j] = (angles[j] + s * step) % 360
                    moved = True
        if not moved:
            break
    return angles


def ha_va(angle):
    cx, cy = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    ha = "center" if abs(cx) < 0.25 else ("left" if cx > 0 else "right")
    va = "center" if abs(cy) < 0.25 else ("bottom" if cy > 0 else "top")
    return ha, va


def wrap(t, w=16):
    return "\n".join(textwrap.wrap(t, w))


# ── drawing ──────────────────────────────────────────────────────────────────
def draw_polar_circle(ax, R=1.0):
    theta = np.linspace(0, 2 * np.pi, 720)
    ax.plot(np.cos(theta) * R, np.sin(theta) * R, color="#999999", lw=1.5, zorder=2)
    for deg in range(0, 360, 30):
        rad = np.deg2rad(deg)
        cx, cy = np.cos(rad), np.sin(rad)
        ax.plot([0, cx * R], [0, cy * R], color="#dddddd", lw=0.7, ls="--", zorder=1)
        ax.plot([cx * R * 0.94, cx * R], [cy * R * 0.94, cy * R], color="#666666", lw=1.2, zorder=3)
        ha = "center" if abs(cx) < 0.15 else ("left" if cx > 0 else "right")
        va = "center" if abs(cy) < 0.15 else ("bottom" if cy > 0 else "top")
        ax.text(cx * 1.10, cy * 1.10, f"{deg}°", color="#111111", fontsize=12,
                ha=ha, va=va, fontweight="bold", zorder=5)
        ax.text(cx * 1.21, cy * 1.21, f"({RADIAN_LABELS[deg]})", color="#555555",
                fontsize=10, ha=ha, va=va, style="italic", zorder=5)


class DualPatchHandler(HandlerBase):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1, self.c2 = c1, c2

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        w2, gap = width * 0.42, width * 0.08
        return [Rectangle([xdescent, ydescent], w2, height, transform=trans,
                          facecolor=self.c1, edgecolor="none"),
                Rectangle([xdescent + w2 + gap, ydescent], w2, height, transform=trans,
                          facecolor=self.c2, edgecolor="none")]


def build_figure(slots_1d, names, ref_idx, cmp_idx, out_path, min_sep, dpi):
    ref = slots_1d[ref_idx]
    adegs = [angle_deg(ref, slots_1d[i]) for i in cmp_idx]
    ref_2d, others_2d = project_2d(ref, [slots_1d[i] for i in cmp_idx])

    colours, cu, wu = [], 0, 0
    for a in adegs:
        c, cu, wu = pick_color(a, cu, wu)
        colours.append(c)

    raw = [float(np.degrees(np.arctan2(ref_2d[1], ref_2d[0])))] + \
          [float(np.degrees(np.arctan2(v[1], v[0]))) for v in others_2d]
    label_degs = resolve_overlaps(raw, min_sep)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif"]
    fig, ax = plt.subplots(figsize=(13, 12), facecolor="white", dpi=dpi)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    draw_polar_circle(ax)

    R_LABEL = 1.33
    ref_arrow = dict(arrowstyle="-|>,head_width=0.18,head_length=0.18", color="#111111",
                     lw=4.0, mutation_scale=28, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=tuple(ref_2d), xytext=(0, 0), arrowprops=ref_arrow, zorder=8)

    def put_label(text, deg, color):
        r = np.deg2rad(deg)
        ha, va = ha_va(deg)
        ax.text(R_LABEL * np.cos(r), R_LABEL * np.sin(r), wrap(text), color=color,
                fontsize=14, ha=ha, va=va, fontweight="bold", linespacing=1.3, zorder=10)

    put_label(names[ref_idx], label_degs[0], "#111111")
    for k, (v, idx, c) in enumerate(zip(others_2d, cmp_idx, colours)):
        arrow = dict(arrowstyle="-|>,head_width=0.14,head_length=0.16", color=c, lw=3.2,
                     mutation_scale=26, shrinkA=0, shrinkB=0)
        ax.annotate("", xy=tuple(v), xytext=(0, 0), arrowprops=arrow, zorder=7)
        put_label(names[idx], label_degs[k + 1], c)

    ax.plot(0, 0, "o", color="#333333", ms=7, zorder=12)
    ax.text(0.0, -0.08, "Latent Origin", color="#777777", fontsize=10,
            ha="center", va="top", zorder=10)

    pad = 0.62
    ax.set_xlim(-1 - pad, 1 + pad)
    ax.set_ylim(-1 - pad, 1 + pad)
    ax.set_aspect("equal")
    ax.set_title(f"Reference: {names[ref_idx].title()}   "
                 "(unit-normalised · ref aligned to +x axis)",
                 color="#111111", fontsize=13, fontweight="bold", pad=0.5)

    ref_patch = mpatches.Patch(color="#111111", label="Reference")
    cool_proxy = mpatches.Patch(color=COOL[0], label="Positive Semantic Alignment  (Angle < 90°)")
    warm_proxy = mpatches.Patch(color=WARM[0], label="Negative Semantic Alignment  (Angle > 90°)")
    leg = ax.legend(handles=[ref_patch, cool_proxy, warm_proxy],
                    handler_map={cool_proxy: DualPatchHandler(*COOL),
                                 warm_proxy: DualPatchHandler(*WARM)},
                    loc="lower center", ncol=3, fontsize=13, frameon=True,
                    facecolor="#f7f7f7", edgecolor="#bbbbbb", labelcolor="#111111",
                    handlelength=2.4, handleheight=1.0, borderpad=0.7, columnspacing=1,
                    bbox_to_anchor=(0.5, 0.0))
    for t in leg.get_texts():
        t.set_fontweight("bold")

    fig.tight_layout(rect=[0.0, 0.06, 1.0, 1.0], pad=0.3)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    args = parse_args()
    slots_all, _, names = load_slot_features(args.features_dir, load_labels=False)
    N = len(slots_all)

    ref_idx = find_idx(names, args.reference)
    cmp_idx = []
    for q in args.compare:
        i = find_idx(names, q)
        if i != ref_idx and i not in cmp_idx:
            cmp_idx.append(i)
    if not cmp_idx:
        raise SystemExit("No valid comparison pathologies.")
    print(f"Reference: {names[ref_idx]} | compare: {[names[i] for i in cmp_idx]}")

    if args.batch_n:
        out_dir = ensure_dir(args.output_dir)
        for i in range(min(args.batch_n, N)):
            try:
                build_figure(slots_all[i], names, ref_idx, cmp_idx,
                             out_dir / f"polar_sim{i}.png", args.min_sep, args.dpi)
            except Exception as e:  # keep going on a bad sample
                print(f"ERROR sample {i}: {e}")
    else:
        idx = min(args.sample_idx, N - 1)
        build_figure(slots_all[idx], names, ref_idx, cmp_idx,
                     ensure_parent(args.output), args.min_sep, args.dpi)


if __name__ == "__main__":
    main()