"""
Per-pathology attention-map overlays on chest X-rays.

For every pathology that is "active" for an image (label present in the CSV
and, unless --include_uncertain, not equal to --uncertain_value) the slot
attention map is upsampled and overlaid on the CXR.

Two modes
---------
  single image : --image <file.jpg>            (absolute, or relative to --root_dir)
  CSV batch    : omit --image; rows --start_idx ... --start_idx+--max_images

Outputs
-------
  <output_dir>/individual/<image_stem>/<pathology>.png   (--save_individual)
  <output_dir>/composite/<image_stem>_composite.png      (--save_composite)

Examples
--------
python analysis/get_model_attn_maps.py \
    --checkpoint checkpoints/slot_attn_final/best_model.pt \
    --csv_file   data/cxr_cbm_exps/concepts_df.csv \
    --root_dir   data/mimic_jpg_cbm \
    --output_dir attention_viz \
    --image p11967920_s54345772_575f3689-....dcm.jpg

python analysis/get_model_attn_maps.py --checkpoint ... --csv_file ... --root_dir ... \
    --start_idx 400 --max_images 20 --save_composite
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from common import get_device  # also adds the repo root to sys.path
from vision_concept_module.model import MedSIGLIPSlotAttention


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    B = argparse.BooleanOptionalAction

    g = p.add_argument_group("inputs")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--csv_file", required=True,
                   help="CSV with a `pa_view_path` column and `*_bin` label columns.")
    g.add_argument("--root_dir", required=True, help="Directory containing the CXR jpgs.")
    g.add_argument("--output_dir", default="attention_visualizations")
    g.add_argument("--image", default=None,
                   help="Single-image mode: jpg named <pid>_<sid>_<file>.jpg (absolute or "
                        "relative to --root_dir). Omit for CSV batch mode.")
    g.add_argument("--start_idx", type=int, default=0, help="Batch mode: first CSV row.")
    g.add_argument("--max_images", type=int, default=None, help="Batch mode: number of rows.")

    g = p.add_argument_group("model")
    g.add_argument("--processor_id", default="google/medsiglip-448")
    g.add_argument("--num_tasks", type=int, default=236)
    g.add_argument("--num_classes_per_task", type=int, default=3)
    g.add_argument("--hidden_dim", type=int, default=128)
    g.add_argument("--slot_iters", type=int, default=3)
    g.add_argument("--reduced_dim", type=int, default=768)
    g.add_argument("--embedding_path", default=None)
    g.add_argument("--freeze_backbone", action=B, default=True)
    g.add_argument("--device", default=None, help="cpu | cuda | hpu | ... (auto if omitted)")

    g = p.add_argument_group("which pathologies")
    g.add_argument("--uncertain_value", type=float, default=0,
                   help="Label value meaning 'uncertain' in the CSV.")
    g.add_argument("--include_uncertain", action="store_true",
                   help="Also visualise pathologies labelled uncertain.")

    g = p.add_argument_group("visualisation")
    g.add_argument("--save_individual", action=B, default=True)
    g.add_argument("--save_composite", action=B, default=False)
    g.add_argument("--alpha", type=float, default=0.85, help="Overlay strength.")
    g.add_argument("--colormap", default="turbo", help="OpenCV colormap: turbo, jet, hot, inferno, ...")
    g.add_argument("--gamma", type=float, default=0.35, help="<1 brightens hotspots.")
    g.add_argument("--threshold", type=float, default=0.25, help="Zero-out weaker activations.")
    g.add_argument("--intensity", type=float, default=2.5)
    g.add_argument("--blur_sigma", type=float, default=1.0)
    return p.parse_args()


def get_cv2_colormap(name: str) -> int:
    attr = f"COLORMAP_{name.upper()}"
    if not hasattr(cv2, attr):
        raise ValueError(f"Unknown OpenCV colormap '{name}'")
    return getattr(cv2, attr)


class AttentionMapVisualizer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.root_dir = Path(args.root_dir)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colormap = get_cv2_colormap(args.colormap)

        self.model = self._load_model()
        self.processor = AutoProcessor.from_pretrained(args.processor_id)

        self.df = pd.read_csv(args.csv_file)
        self.pathology_columns = [c for c in self.df.columns if c.endswith("_bin")]
        print(f"CSV rows: {len(self.df)} | pathology columns: {len(self.pathology_columns)}")
        self._stem_to_row = {self._csv_path_to_stem(p): i
                             for i, p in enumerate(self.df["pa_view_path"])}

    # ── setup ────────────────────────────────────────────────────────────────
    def _load_model(self):
        a = self.args
        print(f"Loading model from {a.checkpoint}")
        model = MedSIGLIPSlotAttention(
            num_tasks=a.num_tasks, num_classes_per_task=a.num_classes_per_task,
            hidden_dim=a.hidden_dim, slot_iters=a.slot_iters,
            freeze_backbone=a.freeze_backbone, lambda_attn_loss=0.0,
            reduced_dim=a.reduced_dim, disease_embeddings_path=a.embedding_path)
        ckpt = torch.load(a.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
        return model.to(self.device).eval()

    @staticmethod
    def _csv_path_to_stem(raw_path):
        """.../p10000032/s53189527/abc.dcm  ->  p10000032_s53189527_abc.dcm"""
        pid, sid, fid = str(raw_path).rstrip("/").split("/")[-3:]
        return f"{pid}_{sid}_{fid}"

    def _active_pathologies(self, row):
        names, indices = [], []
        for i, col in enumerate(self.pathology_columns):
            v = row[col]
            if pd.isna(v):
                continue
            if not self.args.include_uncertain and v == self.args.uncertain_value:
                continue
            names.append(f"{col[:-4]}_{int(v)}")
            indices.append(i)
        return names, indices

    # ── inference + image ops ────────────────────────────────────────────────
    def _infer(self, image_path):
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            out = self.model(pixel_values)
        probs = F.softmax(out["logits"], dim=-1)
        return image, out["attentions"], torch.argmax(out["logits"], dim=-1), probs

    def resize_attention_map(self, attention_map, target_size):
        """Percentile-normalise -> gamma -> threshold -> blur -> amplify -> cubic resize."""
        a = self.args
        g = int(np.sqrt(len(attention_map)))
        attn = attention_map.reshape(g, g).astype(np.float32)
        attn = np.clip(attn, np.percentile(attn, 5), np.percentile(attn, 99))
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        attn = attn ** a.gamma
        attn[attn < a.threshold] = 0
        attn = cv2.GaussianBlur(attn, (0, 0), a.blur_sigma)
        attn = np.clip(attn * a.intensity, 0, 1)
        return cv2.resize(attn, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)

    def create_overlay(self, original_image, attention_map):
        img = np.array(original_image) if isinstance(original_image, Image.Image) else original_image
        if img.ndim == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype(np.float32)
        if img.max() > 1:
            img /= 255.0

        heat = cv2.applyColorMap(np.uint8(attention_map * 255), self.colormap)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        alpha = attention_map[..., None] * self.args.alpha       # strong attention -> stronger overlay
        overlay = img * (1 - alpha) + heat * alpha
        glow = cv2.GaussianBlur(overlay, (0, 0), sigmaX=3)
        return np.clip(overlay + 0.25 * glow * alpha, 0, 1)

    def create_composite(self, original_image, maps, preds, probs):
        n = len(maps)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
        axes = axes.flatten()
        for ax, (name, m) in zip(axes, maps.items()):
            ax.imshow(self.create_overlay(original_image, m))
            ax.axis("off")
            ax.set_title(f"{name}\nPred: {preds[name]} (p={probs[name]:.3f})", fontsize=10)
        for ax in axes[n:]:
            ax.axis("off")
        plt.tight_layout()
        return fig

    # ── core worker ──────────────────────────────────────────────────────────
    def _run(self, image_path, names, indices):
        image_path = Path(image_path)
        stem = image_path.stem
        image, attn, preds, probs = self._infer(image_path)
        original = np.array(image)
        size = original.shape[:2]

        maps, pred_d, prob_d = {}, {}, {}
        for name, t in zip(names, indices):
            maps[name] = self.resize_attention_map(attn[0, t].cpu().numpy(), size)
            cls = preds[0, t].item()
            pred_d[name], prob_d[name] = cls, probs[0, t, cls].item()

        if self.args.save_individual:
            d = self.output_dir / "individual" / stem
            d.mkdir(parents=True, exist_ok=True)
            for name, m in maps.items():
                overlay = Image.fromarray((self.create_overlay(original, m) * 255).astype(np.uint8))
                overlay.save(d / f"{name.replace('/', '_')}.png")
            print(f"  Saved {len(maps)} overlays -> {d}")

        if self.args.save_composite:
            d = self.output_dir / "composite"
            d.mkdir(parents=True, exist_ok=True)
            fig = self.create_composite(original, maps, pred_d, prob_d)
            fig.savefig(d / f"{stem}_composite.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved composite -> {d / (stem + '_composite.png')}")

    # ── public entry points ──────────────────────────────────────────────────
    def visualize_image(self, image):
        path = Path(image)
        if not path.is_absolute():
            path = self.root_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        stem = path.stem
        if stem not in self._stem_to_row:
            raise KeyError(f"'{stem}' not in CSV. Files must be named <pid>_<sid>_<file>.jpg "
                           "matching the `pa_view_path` column.")
        row = self.df.iloc[self._stem_to_row[stem]]
        names, indices = self._active_pathologies(row)
        if not names:
            print(f"No active pathologies for {stem}; skipping.")
            return
        print(f"\n{stem}\n  active: {names}")
        self._run(path, names, indices)

    def visualize_row(self, row_idx):
        row = self.df.iloc[row_idx]
        stem = self._csv_path_to_stem(row["pa_view_path"])
        names, indices = self._active_pathologies(row)
        if not names:
            print(f"No active pathologies for {stem}; skipping.")
            return
        print(f"\n{stem}\n  active: {names}")
        self._run(self.root_dir / f"{stem}.jpg", names, indices)

    def visualize_batch(self, start_idx=0, max_images=None):
        end = len(self.df) if max_images is None else min(start_idx + max_images, len(self.df))
        print(f"Processing rows {start_idx}..{end - 1}")
        for idx in tqdm(range(start_idx, end)):
            try:
                self.visualize_row(idx)
            except Exception as e:  # keep going on a bad row
                print(f"Error on row {idx}: {e}")
        print(f"\nDone. Output in {self.output_dir}")


def main():
    args = parse_args()
    viz = AttentionMapVisualizer(args, get_device(args.device))
    if args.image:
        viz.visualize_image(args.image)
    else:
        viz.visualize_batch(args.start_idx, args.max_images)


if __name__ == "__main__":
    main()