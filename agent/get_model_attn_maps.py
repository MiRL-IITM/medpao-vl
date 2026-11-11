import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import os
from tqdm import tqdm
from transformers import AutoProcessor
import cv2
import habana_frameworks.torch.core as htcore

from model_ablations.medsiglip_slot_attn_shared_cls import MedSIGLIPSlotAttention


class AttentionMapVisualizer:
    """Visualizer for attention maps with CXR overlay"""
    
    def __init__(
        self, 
        model_path,
        csv_path,                   # now always required
        output_dir="./attention_visualizations",
        model_config=None,
        device='cuda',
        uncertain_value=2,          # ground-truth value that means "uncertain"
    ):
        """
        Args:
            model_path      : Path to saved model checkpoint (.pt file)
            csv_path        : Path to CSV with CXR paths and pathology columns
            output_dir      : Directory to save visualizations
            model_config    : Dict with model configuration
            device          : Device to run inference on
            uncertain_value : GT label value that represents "uncertain" (default 2)
        """
        self.model_path      = model_path
        self.csv_path        = csv_path
        self.output_dir      = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device          = device
        self.root_dir        = '/home/user/cxr_cbm/data/mimic_jpg_cbm'
        self.uncertain_value = uncertain_value

        if model_config is None:
            model_config = {
                'num_tasks': 236,
                'num_classes_per_task': 3,
                'freeze_backbone': True,
                'reduced_dim': 768,
                'disease_embeddings_path': None
            }
        self.model_config = model_config

        self.model     = self._load_model()
        self.processor = AutoProcessor.from_pretrained("google/medsiglip-448")

        # Load CSV and build a lookup index keyed by reconstructed image stem
        self.df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(self.df)} rows")

        self.pathology_columns = [col for col in self.df.columns if col.endswith('_bin')]
        print(f"Found {len(self.pathology_columns)} pathology columns")

        # Build {image_stem -> row_index} for fast single-image lookups
        self._build_stem_index()

    # ------------------------------------------------------------------ #
    #  Setup helpers                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        print(f"Loading model from {self.model_path}")
        model = MedSIGLIPSlotAttention(**self.model_config)
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        print("Model loaded successfully")
        return model

    def _csv_row_to_stem(self, raw_path):
        """
        Convert the raw CSV path value (pa_view_path) into the image file stem
        that corresponds to the physical file on disk.

        CSV value example : /some/prefix/p10000032/s53189527/2a2277a9-...dcm
        Physical file     : <root_dir>/p10000032_s53189527_2a2277a9-...dcm.jpg
        Stem we store     : p10000032_s53189527_2a2277a9-...dcm
        """
        parts = str(raw_path).rstrip('/').split('/')
        # last three meaningful segments: pid, sid, id
        pid, sid, fid = parts[-3], parts[-2], parts[-1]
        return f"{pid}_{sid}_{fid}"

    def _build_stem_index(self):
        """Pre-compute {image_stem -> DataFrame row index} for O(1) lookup."""
        self._stem_to_row = {}
        for i, raw_path in enumerate(self.df['pa_view_path']):
            stem = self._csv_row_to_stem(raw_path)
            self._stem_to_row[stem] = i
        print(f"Built stem index for {len(self._stem_to_row)} images")

    def _get_csv_row_for_image(self, image_path):
        """
        Given a physical image path, locate its row in the CSV.

        Returns (row_index, pd.Series) or raises KeyError if not found.
        """
        stem = Path(image_path).stem          # e.g. p10000032_s53189527_2a2277a9-...dcm
        if stem not in self._stem_to_row:
            raise KeyError(
                f"Image stem '{stem}' not found in CSV index.\n"
                f"Make sure the file is named as <pid>_<sid>_<filename> "
                f"matching the pa_view_path column."
            )
        row_idx = self._stem_to_row[stem]
        return row_idx, self.df.iloc[row_idx]

    def _get_active_pathologies(self, row, exclude_uncertain=True):
        """
        Return (active_pathology_names, active_task_indices) for one CSV row.

        A pathology is active when:
          - its GT value is not NaN
          - its GT value != 0  (negative / absent)
          - if exclude_uncertain: its GT value != self.uncertain_value
        """
        active_pathologies = []
        active_indices     = []

        for task_idx, pathology in enumerate(self.pathology_columns):
            value = row[pathology]
            if pd.isna(value):
                continue
            # if value == 0:
            #     continue
            if exclude_uncertain and value == self.uncertain_value:
                continue
            active_pathologies.append(f"{pathology}_{int(value)}")
            active_indices.append(task_idx)

        return active_pathologies, active_indices

    # ------------------------------------------------------------------ #
    #  Core image helpers                                                  #
    # ------------------------------------------------------------------ #

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        return image, pixel_values

    def get_attention_maps(self, pixel_values):
        with torch.no_grad():
            outputs       = self.model(pixel_values)
            attention_maps = outputs['attentions']          # [B, num_tasks, num_patches]
            predictions    = torch.argmax(outputs['logits'], dim=-1)
            probabilities  = F.softmax(outputs['logits'], dim=-1)
        return attention_maps, predictions, probabilities

    def resize_attention_map(
        self,
        attention_map,
        target_size,
        gamma=0.35,
        threshold=0.25,
        intensity_scale=2.5,
        blur_sigma=1.0,
    ):
        """
        Improved attention visualization pipeline.

        Techniques:
        - percentile normalization
        - gamma enhancement
        - weak activation suppression
        - intensity amplification
        - gaussian smoothing
        """

        num_patches = len(attention_map)
        grid_size = int(np.sqrt(num_patches))

        attn_2d = attention_map.reshape(grid_size, grid_size)

        # ----------------------------------------------------
        # Percentile normalization (better than min-max)
        # ----------------------------------------------------
        p_low = np.percentile(attn_2d, 5)
        p_high = np.percentile(attn_2d, 99)

        attn_2d = np.clip(attn_2d, p_low, p_high)

        attn_2d = (attn_2d - attn_2d.min()) / (
            attn_2d.max() - attn_2d.min() + 1e-8
        )

        # ----------------------------------------------------
        # Gamma sharpening
        # Smaller gamma => brighter hotspots
        # ----------------------------------------------------
        attn_2d = attn_2d ** gamma

        # ----------------------------------------------------
        # Remove weak activations
        # ----------------------------------------------------
        attn_2d[attn_2d < threshold] = 0

        # ----------------------------------------------------
        # Smooth blobs slightly
        # ----------------------------------------------------
        attn_2d = cv2.GaussianBlur(attn_2d, (0, 0), blur_sigma)

        # ----------------------------------------------------
        # Amplify intensity
        # ----------------------------------------------------
        attn_2d = np.clip(attn_2d * intensity_scale, 0, 1)

        # ----------------------------------------------------
        # Resize with cubic interpolation
        # better than PIL bilinear
        # ----------------------------------------------------
        attn_resized = cv2.resize(
            attn_2d,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_CUBIC
        )

        return attn_resized


    def create_overlay(
        self,
        original_image,
        attention_map,
        alpha=0.75,
        colormap=cv2.COLORMAP_TURBO,
    ):
        """
        High-quality medical attention overlay.
        """

        # ----------------------------------------------------
        # Prepare original image
        # ----------------------------------------------------
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        elif original_image.shape[2] == 1:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        original_image = original_image.astype(np.float32)

        if original_image.max() > 1:
            original_image /= 255.0

        # ----------------------------------------------------
        # Convert attention to heatmap
        # ----------------------------------------------------
        heatmap_uint8 = np.uint8(attention_map * 255)

        heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0

        # ----------------------------------------------------
        # Adaptive alpha
        # Strong attention => stronger overlay
        # ----------------------------------------------------
        adaptive_alpha = attention_map[..., None] * alpha

        # ----------------------------------------------------
        # Blend
        # ----------------------------------------------------
        overlay = (
            original_image * (1 - adaptive_alpha)
            + heatmap * adaptive_alpha
        )

        # ----------------------------------------------------
        # Extra glow enhancement
        # ----------------------------------------------------
        glow = cv2.GaussianBlur(overlay, (0, 0), sigmaX=3)

        overlay = np.clip(
            overlay + 0.25 * glow * adaptive_alpha,
            0,
            1
        )

        return overlay

    def create_composite_visualization(
        self, original_image, attention_maps_dict,
        predictions_dict, probabilities_dict, alpha=0.5
    ):
        num_pathologies = len(attention_maps_dict)
        cols = min(3, num_pathologies)
        rows = (num_pathologies + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if num_pathologies == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes

        for idx, (pathology, attn_map) in enumerate(attention_maps_dict.items()):
            overlay = self.create_overlay(original_image, attn_map, alpha=alpha)
            axes[idx].imshow(overlay)
            axes[idx].axis('off')
            pred_class = predictions_dict[pathology]
            prob       = probabilities_dict[pathology]
            axes[idx].set_title(f"{pathology}\nPred: {pred_class} (p={prob:.3f})", fontsize=10)

        for idx in range(num_pathologies, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    #  Shared inner worker (used by BOTH single-path and CSV batch modes) #
    # ------------------------------------------------------------------ #

    def _run_visualization(
        self,
        image_path,
        active_pathologies,
        active_indices,
        save_composite=True,
        save_individual=True,
        alpha=0.85,
colormap=cv2.COLORMAP_TURBO,
    ):
        """
        Core routine: given a resolved image path + active (name, index) pairs,
        run inference, build overlays, and save outputs.

        Returns result dict.
        """
        image_name = Path(image_path).stem

        original_image, pixel_values = self.preprocess_image(image_path)
        attention_maps, predictions, probabilities = self.get_attention_maps(pixel_values)

        original_np = np.array(original_image)
        img_size    = original_np.shape[:2]

        attention_maps_dict = {}
        predictions_dict    = {}
        probabilities_dict  = {}

        for name, task_idx in zip(active_pathologies, active_indices):
            attn         = attention_maps[0, task_idx].cpu().numpy()
            attn_resized = self.resize_attention_map(attn, img_size)

            attention_maps_dict[name] = attn_resized
            predictions_dict[name]   = predictions[0, task_idx].item()
            pred_cls                  = predictions[0, task_idx].item()
            probabilities_dict[name]  = probabilities[0, task_idx, pred_cls].item()

        # Individual overlays
        if save_individual:
            ind_dir = self.output_dir / 'individual' / image_name
            ind_dir.mkdir(parents=True, exist_ok=True)
            for name, attn_map in attention_maps_dict.items():
                overlay     = self.create_overlay(original_np, attn_map, alpha=alpha, colormap=colormap)
                overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
                name = name.replace('/', '_')  # sanitize for filesystem
                save_path   = ind_dir / f"{name}.png"
              
                overlay_img.save(save_path)
                print(f"  Saved: {save_path}")

        # Composite figure
        composite_fig = None
        if save_composite:
            comp_dir = self.output_dir / 'composite'
            comp_dir.mkdir(parents=True, exist_ok=True)
            composite_fig = self.create_composite_visualization(
                original_np, attention_maps_dict,
                predictions_dict, probabilities_dict, alpha=alpha
            )
            save_path = comp_dir / f"{image_name}_composite.png"
            composite_fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(composite_fig)
            print(f"  Saved composite: {save_path}")

        return {
            'attention_maps': attention_maps_dict,
            'predictions':    predictions_dict,
            'probabilities':  probabilities_dict,
            'composite_fig':  composite_fig,
        }

    # ------------------------------------------------------------------ #
    #  PUBLIC: single image path entry point                               #
    # ------------------------------------------------------------------ #

    def visualize_single_image_path(
        self,
        image_path,
        save_composite=True,
        save_individual=True,
        alpha=0.6,
        colormap='jet',
    ):
        """
        Generate attention maps for one image by looking it up in the CSV.

        - Resolves the CSV row from the image filename stem
        - Keeps only pathologies whose GT label is NOT 0 (absent) and
          NOT self.uncertain_value (uncertain)
        - Saves individual overlays and/or a composite figure

        Args:
            image_path     : Absolute path to the CXR .jpg file.
                             The filename stem must follow the pattern
                             <pid>_<sid>_<filename> so it can be matched
                             against the pa_view_path column in the CSV.
            save_composite : Save a composite PNG of all active tasks.
            save_individual: Save one PNG per active task.
            alpha          : Overlay transparency.
            colormap       : Matplotlib colormap name.

        Returns:
            dict with 'attention_maps', 'predictions', 'probabilities',
            'composite_fig'.
        """
        image_path = os.path.join(self.root_dir, image_path)
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # ── Look up CSV row ─────────────────────────────────────────────
        row_idx, row = self._get_csv_row_for_image(image_path)
        print(f"\nFound CSV row {row_idx} for: {image_path.name}")

        # ── Filter pathologies: active & not uncertain ──────────────────
        active_pathologies, active_indices = self._get_active_pathologies(
            row, exclude_uncertain=False
        )

        if not active_pathologies:
            print("No active (non-zero, non-uncertain) pathologies found — skipping.")
            return {}

        print(f"Active pathologies (excluding uncertain): {active_pathologies}")

        return self._run_visualization(
            image_path=image_path,
            active_pathologies=active_pathologies,
            active_indices=active_indices,
            save_composite=save_composite,
            save_individual=save_individual,
            alpha=alpha,
            colormap=colormap,
        )

    # ------------------------------------------------------------------ #
    #  PUBLIC: CSV batch mode (original behaviour, now uses shared worker) #
    # ------------------------------------------------------------------ #

    def process_single_image(self, row_idx, save_composite=True, save_individual=True):
        """Process one row from the loaded CSV DataFrame."""
        row      = self.df.iloc[row_idx]
        raw_path = row['pa_view_path']

        # Reconstruct the physical file path (same logic as before)
        stem         = self._csv_row_to_stem(raw_path)

        stem = raw_path.split('/')[-3] + '_' + raw_path.split('/')[-2] + '_' + raw_path.split('/')[-1]+'.jpg'
        correct_path = Path(self.root_dir) / f"{stem}"
        print(f"correct_path is {correct_path}")

        # Filter pathologies: active & not uncertain
        active_pathologies, active_indices = self._get_active_pathologies(
            row, exclude_uncertain=True
        )

        if not active_pathologies:
            print(f"No active (non-zero, non-uncertain) pathologies for {stem}, skipping...")
            return

        print(f"\nProcessing {stem}")
        print(f"Active pathologies (excluding uncertain): {active_pathologies}")

        self._run_visualization(
            image_path=correct_path,
            active_pathologies=active_pathologies,
            active_indices=active_indices,
            save_composite=save_composite,
            save_individual=save_individual,
        )

        print(f"Saved visualizations for {stem}")

    def process_all_images(self, max_images=None, save_composite=True, save_individual=True):
        """Process all (or up to max_images) rows in the CSV."""
        num_images = len(self.df) if max_images is None else min(max_images, len(self.df))
        print(f"Processing {num_images} images...")

        for idx in tqdm(range(num_images)):
            try:
                self.process_single_image(
                    idx+400,
                    save_composite=save_composite,
                    save_individual=save_individual,
                )
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        print(f"\nCompleted! Visualizations saved to {self.output_dir}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    MODEL_PATH   = "/home/user/cxr_cbm/data/cxr_cbm_exps/checkpoints/slot_attn_final/best_model.pt"
    CSV_PATH     = "/home/user/cxr_cbm/data/cxr_cbm_exps/concepts_df.csv"
    OUTPUT_DIR   = "/home/user/cxr_cbm/data/cxr_cbm_exps/attention_viz_temp"
    DEVICE       = 'hpu'

    MODEL_CONFIG = {
        'num_tasks': 236,
        'num_classes_per_task': 3,
        'freeze_backbone': True,
        'reduced_dim': 768,
        'disease_embeddings_path': None,
    }

    # ── Set to a file path to use single-image mode; None for batch mode ─
    SINGLE_IMAGE_PATH = "/home/user/cxr_cbm/data/mimic_jpg_cbm/p11967920_s54345772_575f3689-e868315f-2fc798b9-b2b91931-ed2ade5d.dcm.jpg"   # e.g. "/home/user/.../p10000032_s53189527_abc.jpg"
    SINGLE_IMAGE_PATH = None # <-- Set to None to run in batch mode on the whole CSV
    
    # CSV is always required (needed for GT label lookup in both modes)
    visualizer = AttentionMapVisualizer(
        model_path=MODEL_PATH,
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        model_config=MODEL_CONFIG,
        device=DEVICE,
        uncertain_value=0,      # adjust if your uncertain label differs
    )

    if SINGLE_IMAGE_PATH is not None:
        # ── Single image mode ───────────────────────────────────────────
        results = visualizer.visualize_single_image_path(
            image_path=SINGLE_IMAGE_PATH,
            save_composite=False,
            save_individual=True,
            alpha=0.85,
            colormap=cv2.COLORMAP_TURBO,
                    )
    else:
        # ── Batch CSV mode ──────────────────────────────────────────────
        visualizer.process_all_images(
            max_images=21,
            save_composite=False,
            save_individual=True,
        )


if __name__ == "__main__":
    main()