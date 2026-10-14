"""
Slot-similarity visualisation for ONE chest X-ray.

  1. Run the model, capture slots [T, D] and attentions [T, P]
  2. Cosine similarity of every slot to a reference pathology
  3. Pick the --top_k most- and least-aligned pathologies
  4. Project those slot vectors to 2-D (reference on +x, unit-normalised)
  5. Figure: LEFT  = CXR with colour-matched attention overlays
             RIGHT = 2-D vector plot of the angular relationships

    python analysis/get_attn_slot_features.py \
        --checkpoint checkpoints/slot_attn_final/best_model.pt \
        --csv_file   data/cxr_cbm_exps/concepts_df.csv \
        --root_dir   data/mimic_jpg_cbm \
        --row_idx 0 --reference "bilateral pleural effusion" \
        --output slot_similarity_viz.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import to_rgba
from PIL import Image
from transformers import AutoProcessor

from common import ensure_parent, get_device  # also adds the repo root to sys.path
from vision_concept_module.model import MedSIGLIPSlotAttention


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    B = argparse.BooleanOptionalAction
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv_file", required=True)
    p.add_argument("--root_dir", required=True, help="Directory containing the CXR jpgs.")
    p.add_argument("--row_idx", type=int, default=0, help="CSV row of the CXR to visualise.")
    p.add_argument("--reference", default="bilateral pleural effusion")
    p.add_argument("--output", default="slot_similarity_viz.png")
    p.add_argument("--device", default=None, help="cpu | cuda | hpu | ... (auto if omitted)")
    p.add_argument("--processor_id", default="google/medsiglip-448")
    p.add_argument("--num_tasks", type=int, default=236)
    p.add_argument("--num_classes_per_task", type=int, default=3)
    p.add_argument("--reduced_dim", type=int, default=768)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--slot_iters", type=int, default=3)
    p.add_argument("--embedding_path", default=None)
    p.add_argument("--freeze_backbone", action=B, default=True)
    p.add_argument("--top_k", type=int, default=2, help="Most/least similar slots to show each.")
    p.add_argument("--attn_alpha", type=float, default=0.65, help="Overlay colour strength (0-1).")
    p.add_argument("--base_weight", type=float, default=0.30, help="CXR brightness (0-1).")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# ── model wrapper ────────────────────────────────────────────────────────────
class SlotAttentionCapture(torch.nn.Module):
    def __init__(self, base: MedSIGLIPSlotAttention):
        super().__init__()
        self.base = base
        self.slots = None        # [B, T, D]
        self.attentions = None   # [B, T, P]

    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        feats = self.base.medsiglip.vision_model(pixel_values).last_hidden_state
        slots, attn = self.base.slot_attention(self.base.dim_reduction(feats))
        self.slots, self.attentions = slots.detach().cpu(), attn.detach().cpu()
        logits = self.base.classifier(slots.reshape(b * self.base.num_tasks, -1)).reshape(
            b, self.base.num_tasks, self.base.num_classes_per_task)
        return {"logits": logits, "attentions": attn}


def load_model(args, device) -> SlotAttentionCapture:
    base = MedSIGLIPSlotAttention(
        num_tasks=args.num_tasks, num_classes_per_task=args.num_classes_per_task,
        hidden_dim=args.hidden_dim, slot_iters=args.slot_iters,
        freeze_backbone=args.freeze_backbone, lambda_attn_loss=0.0,
        reduced_dim=args.reduced_dim, disease_embeddings_path=args.embedding_path)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    base.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    return SlotAttentionCapture(base).to(device).eval()


# ── helpers ──────────────────────────────────────────────────────────────────
def build_image_path(row, root_dir):
    pid, sid, fname = row["pa_view_path"].split("/")[-3:]
    return os.path.join(root_dir, f"{pid}_{sid}_{fname}.jpg")


def resize_attention(attn_flat, target_hw):
    g = int(np.sqrt(len(attn_flat)))
    img = Image.fromarray((attn_flat.reshape(g, g) * 255).astype(np.uint8))
    out = np.array(img.resize((target_hw[1], target_hw[0]), Image.BILINEAR), dtype=np.float32) / 255.0
    return (out - out.min()) / (out.max() - out.min() + 1e-8)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def project_and_normalise(ref_vec, other_vecs):
    """Project onto the plane where ref = +x; return unit 2-D vectors."""
    u = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)
    ortho = None
    for v in other_vecs:
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

    return to2d(ref_vec), [to2d(v) for v in other_vecs]


def find_pathology_index(names, query):
    q = query.lower()
    for i, n in enumerate(names):
        if q in n.lower():
            return i
    raise ValueError(f"Reference '{query}' not found. First 20: {names[:20]}")


# ── figure ───────────────────────────────────────────────────────────────────
def build_figure(img_np, attn_maps, slots, names, ref_idx, top_k,
                 attn_alpha, base_weight, out_path, dpi):
    T = len(names)
    H, W = img_np.shape[:2]

    ref_slot = slots[ref_idx]
    sims = np.array([cosine_sim(ref_slot, slots[i]) for i in range(T)])
    cand = sorted(((i, float(sims[i])) for i in range(T) if i != ref_idx), key=lambda x: x[1])
    bot = [i for i, _ in cand[:top_k]]
    top = [i for i, _ in reversed(cand[-top_k:])]

    selected = [ref_idx] + top + bot
    labels = ["reference"] + [f"high-sim #{k + 1}" for k in range(top_k)] + \
             [f"low-sim #{k + 1}" for k in range(top_k)]
    sel_names = [names[i] for i in selected]

    print("\n── Similarity report ─────────────────────────────")
    print(f"Reference : {names[ref_idx]}")
    for i in top:
        print(f"  [HIGH]  {names[i]:<50s} cos={sims[i]:.4f}")
    for i in bot:
        print(f"  [LOW ]  {names[i]:<50s} cos={sims[i]:.4f}")

    palette = ["#D97800", "#C62828", "#E64A19", "#0277BD", "#6A1B9A", "#2E7D32", "#AD1457"]
    colors = [to_rgba(palette[k % len(palette)]) for k in range(len(selected))]

    ref_2d, others_2d = project_and_normalise(ref_slot, [slots[i] for i in selected[1:]])
    vecs_2d = [ref_2d] + others_2d

    overlay = img_np.astype(np.float32) / 255.0 * base_weight
    for k, idx in enumerate(selected):
        a = resize_attention(attn_maps[idx], (H, W))
        r, g, b, _ = colors[k]
        overlay += np.stack([a * r, a * g, a * b], axis=-1) * attn_alpha
    overlay = np.clip(overlay, 0, 1)

    fig = plt.figure(figsize=(22, 10), facecolor="white", dpi=dpi)
    ax_img = fig.add_axes([0.01, 0.04, 0.45, 0.90])
    ax_vec = fig.add_axes([0.52, 0.07, 0.45, 0.82])

    # left: CXR + overlay
    ax_img.imshow(overlay)
    ax_img.axis("off")
    ax_img.legend(handles=[mpatches.Patch(color=colors[k], label=f"{labels[k]}: {sel_names[k]}"
                                          + ("" if k == 0 else f"   cos={sims[selected[k]]:.3f}"))
                           for k in range(len(selected))],
                  loc="lower left", fontsize=8, framealpha=0.9, facecolor="white",
                  labelcolor="#111", edgecolor="#bbb")
    ax_img.set_title("CXR + Attention Overlay", color="#111", fontsize=14, pad=10)

    # right: vector plot
    ax_vec.set_facecolor("white")
    for sp in ax_vec.spines.values():
        sp.set_color("#bbb")
    ax_vec.tick_params(colors="#444", labelsize=8)
    ax_vec.axhline(0, color="#ccc", lw=1.0, zorder=0)
    ax_vec.axvline(0, color="#ccc", lw=1.0, zorder=0)
    ax_vec.grid(True, color="#ebebeb", lw=0.7, zorder=0)
    th = np.linspace(0, 2 * np.pi, 360)
    ax_vec.plot(np.cos(th), np.sin(th), color="#ddd", lw=1.2, zorder=1)

    for k, (vec, name, color) in enumerate(zip(vecs_2d, sel_names, colors)):
        x, y = float(vec[0]), float(vec[1])
        lw, hw, hl = (5.0, 0.10, 0.12) if k == 0 else (3.5, 0.07, 0.09)
        ax_vec.annotate("", xy=(x, y), xytext=(0, 0), zorder=5 + k,
                        arrowprops=dict(arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                                        color=color, lw=lw, mutation_scale=22,
                                        shrinkA=0, shrinkB=0))
        ax_vec.text(x * 1.22, y * 1.22, name, color=color, fontsize=8, ha="center", va="center",
                    fontweight="bold", zorder=15,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.95, lw=1.2))

    ARC_R, angle_degs = 0.28, {}
    for k, (vec, color) in enumerate(zip(vecs_2d[1:], colors[1:]), start=1):
        ang = float(np.degrees(np.arctan2(vec[1], vec[0])))
        angle_degs[selected[k]] = ang
        ax_vec.add_patch(mpatches.Arc((0, 0), 2 * ARC_R, 2 * ARC_R, angle=0,
                                      theta1=min(0.0, ang), theta2=max(0.0, ang),
                                      color=color, lw=1.8, linestyle="--", alpha=0.9, zorder=4))

    pad = 0.60
    ax_vec.set_xlim(-1 - pad, 1 + pad)
    ax_vec.set_ylim(-1 - pad, 1 + pad)
    ax_vec.set_aspect("equal")
    ax_vec.set_title("Slot Vector Alignment\n(unit-normalised, reference = +x axis)",
                     color="#111", fontsize=13, pad=10)
    ax_vec.set_xlabel("component along reference", fontsize=9, color="#444")
    ax_vec.set_ylabel("orthogonal component", fontsize=9, color="#444")

    handles = [mpatches.Patch(color=colors[k],
                              label=f"{labels[k]}  |  {names[selected[k]]}\n"
                                    f"         cos={sims[selected[k]]:+.4f}   "
                                    f"angle={abs(angle_degs.get(selected[k], 0.0)):.1f}°")
               for k in range(1, len(selected))]
    leg = ax_vec.legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.95,
                        facecolor="white", labelcolor="#111", edgecolor="#bbb",
                        title="Pathology   |   cos-sim   angle vs ref", title_fontsize=7,
                        handlelength=1.2, borderpad=0.8)
    leg.get_title().set_color("#555")
    fig.suptitle(f"Slot Similarity  —  Reference: {names[ref_idx].title()}",
                 color="#111", fontsize=16, y=0.98)

    out_path = str(ensure_parent(out_path))
    if not out_path.lower().endswith((".png", ".jpg", ".jpeg")):
        out_path += ".png"
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved -> {out_path}")


def main():
    args = parse_args()
    device = get_device(args.device)

    df = pd.read_csv(args.csv_file)
    img_path = build_image_path(df.iloc[args.row_idx], args.root_dir)
    print(f"Image: {img_path}")

    names = [c[:-4] for c in df.columns if c.endswith("_bin")]
    print(f"Diseases in CSV: {len(names)}")
    if len(names) != args.num_tasks:
        print(f"  Note: --num_tasks={args.num_tasks} differs from CSV ({len(names)}); using CLI value.")
        names = (names + [f"task_{i}" for i in range(len(names), args.num_tasks)])[:args.num_tasks]

    ref_idx = find_pathology_index(names, args.reference)
    print(f"Reference: '{names[ref_idx]}' (index {ref_idx})")

    model = load_model(args, device)
    processor = AutoProcessor.from_pretrained(args.processor_id)
    pil = Image.open(img_path).convert("RGB")
    pixel_values = processor(images=pil, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        model(pixel_values)

    slots, attn = model.slots[0].numpy(), model.attentions[0].numpy()
    print(f"Slots {slots.shape} | attentions {attn.shape}")

    build_figure(np.array(pil), attn, slots, names, ref_idx, args.top_k,
                 args.attn_alpha, args.base_weight, args.output, args.dpi)


if __name__ == "__main__":
    main()