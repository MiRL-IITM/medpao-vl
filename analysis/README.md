# `analysis/` — slot analysis & visualisation

Scripts for analysing the slot-attention CXR concept model (`MedSIGLIPSlotAttention`):
extracting slot vectors, plotting how they cluster / relate, and overlaying
attention maps on chest X-rays.

Nothing is hard-coded: every path, checkpoint, device and plot setting is a command-line flag.
Run any script with `--help` for its full options.

## Contents

| Script | Needs model? | What it does |
|---|:-:|---|
| `save_slots.py` | ✅ | Runs the model over a test set and saves slot vectors, labels and disease names. **Run this first.** |
| `get_model_attn_maps.py` | ✅ | Per-pathology attention-map overlays on CXRs (single image or CSV batch). |
| `get_attn_slot_features.py` | ✅ | One CXR: attention overlay + 2-D slot-vector alignment plot vs. a reference pathology. |
| `analyze_slots.py` | – | Quick inter-slot cosine-similarity heatmap. |
| `slot_abcde_similarity.py` | – | Slot-prototype similarity grouped by ABCDEF category (full heatmap + zoom on one category). |
| `only_tsne_slots.py` | – | Global t-SNE of all slot vectors coloured by ABCDEF category. |
| `per_slot_analysis.py` | – | One-figure grid of per-pathology t-SNE (pos / neg / uncertain) for the top-N balanced pathologies. |
| `getting_all_tsne_plots.py` | – | Same as above, split over several images (for large N). |
| `tsne_specific_patho.py` | – | Same t-SNE grid for a hand-picked list of pathologies. |
| `plot_slot_vectors.py` | – | Polar plot of angles between a reference slot and comparison slots for one sample. |
| `common.py` | – | Shared helpers (loading, device, t-SNE panels). Not run directly. |

## Setup

Python 3.10+ and the same environment you use for the main project, plus:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm pillow
# only for the three model-based scripts:
pip install torch transformers opencv-python
```

* scikit-learn ≥ 1.5 is preferred (`only_tsne_slots.py` falls back to the old `n_iter` argument on older versions).
* Habana HPUs: `--device hpu` imports `habana_frameworks` automatically. With no `--device`, the order is CUDA → HPU → CPU.
* The model scripts import `MedSIGLIPSlotAttention` and `CXRDataset` from `vision_concept_module/` at the **repo root**. `common.py` adds the repo root to `sys.path`, so run from anywhere:

```bash
python analysis/save_slots.py ...
```

## Workflow

```
 test CSV + images + checkpoint
            │
            ▼
      save_slots.py ───────────►  slot_features/
            │                       ├─ slot_features.npy   [N, T, D]
            │                       ├─ labels.npy          [N, T]   (1 / 0 / -1)
            │                       ├─ disease_names.json  [T]
            │                       ├─ sample_ids.npy, metadata.json
            ▼
 analyze_slots · slot_abcde_similarity · only_tsne_slots
 per_slot_analysis · getting_all_tsne_plots · tsne_specific_patho
 plot_slot_vectors                      (all read --features_dir slot_features)

 get_model_attn_maps · get_attn_slot_features   (run the model directly, no slot_features needed)
```

All plotting scripts share `--features_dir` (default `slot_features`, which matches the default `--output_dir` of `save_slots.py`).

## Inputs

* **CSV** (`--csv_file`): one row per CXR, a `pa_view_path` column (e.g. `.../p10000032/s53189527/<id>.dcm`) and one `<pathology>_bin` column per task.
* **Images** (`--root_dir`): jpgs named `<pid>_<sid>_<id>.dcm.jpg`, i.e. the last three components of `pa_view_path` joined with `_`.
* **Label encoding** in `labels.npy`: `1` = positive, `0` = uncertain, `-1` = negative.
* **`concepts_dict.json`** (only `only_tsne_slots.py`, `slot_abcde_similarity.py`): `{"<disease name>": "A".."F"}`. Names that are missing are grouped as `Other`.

## Usage

### 1. Extract slots

```bash
python analysis/save_slots.py \
    --checkpoint checkpoints/slot_attn_final/best_model.pt \
    --csv_file   data/cxr_cbm_exps/concepts_df.csv \
    --root_dir   data/mimic_jpg_cbm \
    --output_dir slot_features \
    --stratify_column "clear lungs_bin" --seed 42
```

* By default it recreates the stratified split and extracts the **test fold**. Use the same `--test_split`, `--stratify_column` and `--seed` as training. Use `--no_split` to process every row of the CSV.
* Model flags (`--hidden_dim`, `--slot_iters`, `--reduced_dim`, `--embedding_path`) must match the checkpoint.

### 2. Slot-space plots (no model needed)

```bash
# similarity heatmap
python analysis/analyze_slots.py --features_dir slot_features --output figures/01_slot_similarity_heatmap.png

# similarity by ABCDEF category + zoom on slots 2–12 of category B
python analysis/slot_abcde_similarity.py --features_dir slot_features \
    --concepts_dict concepts_dict.json --output_dir figures \
    --zoom_category B --zoom_start 2 --zoom_end 12

# global t-SNE coloured by category
python analysis/only_tsne_slots.py --features_dir slot_features \
    --concepts_dict concepts_dict.json --output figures/00_slot_tsne.png

# per-pathology class separation, top 10 most balanced
python analysis/per_slot_analysis.py --features_dir slot_features --top_n 10 --output figures/separation.png

# ... or 30 pathologies, 10 per image
python analysis/getting_all_tsne_plots.py --features_dir slot_features \
    --top_n 30 --batch_size 10 --output_dir figures/tsne_batches

# ... or a hand-picked set
python analysis/tsne_specific_patho.py --features_dir slot_features \
    --pathologies "pulmonary edema" "low lung volumes" "bilateral pleural effusions" \
    --ncols 3 --output figures/clinical.png

# polar angle plot (names matched by case-insensitive substring)
python analysis/plot_slot_vectors.py --features_dir slot_features \
    --reference "pleural effusion" --compare pneumonia atelectasis edema \
    --sample_idx 0 --output figures/polar_sim.png
#   one figure per sample:   --batch_n 20 --output_dir figures/polar/
```

Useful knobs

| Flag | Scripts | Meaning |
|---|---|---|
| `--perplexity` | per-pathology t-SNE scripts, `only_tsne_slots.py` | t-SNE perplexity (clamped below the number of samples; `per_slot_analysis.py` defaults to √N). |
| `--top_n` | `per_slot_analysis.py`, `getting_all_tsne_plots.py` | Pathologies ranked by their smallest class count (pos/unc/neg), so the most balanced come first. |
| `--max_per_class`, `--pca_dims` | `only_tsne_slots.py` | Class-balanced subsampling and PCA size before t-SNE. |
| `--no_center` | `slot_abcde_similarity.py` | Skip zero-centring of prototypes. |
| `--min_sep` | `plot_slot_vectors.py` | Minimum angle (°) between label positions. |
| `--dpi`, `--seed` | all | Resolution and reproducibility. |

### 3. Attention maps on CXRs (model needed)

```bash
# one image (absolute path, or relative to --root_dir)
python analysis/get_model_attn_maps.py \
    --checkpoint checkpoints/slot_attn_final/best_model.pt \
    --csv_file data/cxr_cbm_exps/concepts_df.csv --root_dir data/mimic_jpg_cbm \
    --output_dir attention_viz --image p11967920_s54345772_575f3689-....dcm.jpg

# 20 CSV rows starting at row 400, plus composite figures
python analysis/get_model_attn_maps.py --checkpoint ... --csv_file ... --root_dir ... \
    --start_idx 400 --max_images 20 --save_composite
```

* Only pathologies whose label ≠ `--uncertain_value` (default `0`) are drawn; add `--include_uncertain` to draw all labelled ones.
* Output: `attention_viz/individual/<image>/<pathology>_<label>.png` and (optional) `attention_viz/composite/<image>_composite.png`.
* Appearance: `--colormap`, `--alpha`, `--gamma`, `--threshold`, `--intensity`, `--blur_sigma`.

```bash
# attention overlay + slot alignment for a single CXR
python analysis/get_attn_slot_features.py \
    --checkpoint checkpoints/slot_attn_final/best_model.pt \
    --csv_file data/cxr_cbm_exps/concepts_df.csv --root_dir data/mimic_jpg_cbm \
    --row_idx 0 --reference "bilateral pleural effusion" --top_k 2 \
    --output figures/slot_similarity_viz.png
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: vision_concept_module` | Keep `analysis/` inside the repo, next to the `vision_concept_module/` folder. |
| `ModuleNotFoundError: habana_frameworks` (model-based scripts) | The model was trained on Intel Gaudi 2 and `vision_concept_module/model.py` imports the Habana bindings, so `save_slots.py`, `get_model_attn_maps.py` and `get_attn_slot_features.py` need the Intel Gaudi software stack installed. The other scripts only read saved `slot_features/` and do not need it. |
| `FileNotFoundError: slot_features.npy` | Run `save_slots.py` first, or point `--features_dir` at its `--output_dir`. |
| `'<name>' not found` | Names are matched against `disease_names.json`; use a unique substring. |
| Wrong number of tasks / state-dict mismatch | `--num_tasks`, `--hidden_dim`, `--slot_iters`, `--reduced_dim` must match the checkpoint. |
| t-SNE is slow | Lower `--top_n`, `--max_per_class` or `--pca_dims`; n_jobs is already `-1`. |
| Image "not in CSV" | The jpg stem must equal `<pid>_<sid>_<file>` built from `pa_view_path`. |
| Slow / huge figures | Lower `--dpi` (600 for `tsne_specific_patho.py` and `only_tsne_slots.py` is publication quality). |

All plots are written to disk (Agg backend), so no display is required.