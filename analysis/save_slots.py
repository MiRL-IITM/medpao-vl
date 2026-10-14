"""
Extract the `slots` tensor [B, T, D] from MedSIGLIPSlotAttention over a test set.

`slots` is the output of the slot-attention block (GRU + MLP refinement),
i.e. the tensor fed to the classifier:

    slots, attn = self.slot_attention(vision_features_reduced)   # [B, T, D]

Outputs (written to --output_dir) — consumed by the other analysis scripts
-------------------------------------------------------------------------
    slot_features.npy   float32  [N, T, D]
    labels.npy          float32  [N, T]     (1 = pos, 0 = uncertain, -1 = neg)
    sample_ids.npy      int64    [N]
    disease_names.json  list of T names
    metadata.json       provenance (checkpoint, shapes, split, args)

Usage
-----
# re-create the stratified split used in training and extract the test fold
python analysis/save_slots.py \
    --checkpoint checkpoints/slot_attn_final/best_model.pt \
    --csv_file   data/cxr_cbm_exps/concepts_df.csv \
    --root_dir   data/mimic_jpg_cbm \
    --output_dir slot_features \
    --stratify_column "clear lungs_bin" --seed 42

# use every row of the CSV (no split)
python analysis/save_slots.py --checkpoint ... --csv_file data/my_test_set.csv \
    --root_dir ... --no_split
"""
import argparse
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from common import get_device  # also adds the repo root to sys.path
from vision_concept_module.dataloader import CXRDataset
from vision_concept_module.model import MedSIGLIPSlotAttention

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Extract slot features from MedSIGLIPSlotAttention",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    B = argparse.BooleanOptionalAction

    g = p.add_argument_group("data")
    g.add_argument("--csv_file", required=True, help="CSV with image paths + *_bin label columns.")
    g.add_argument("--root_dir", required=True, help="Directory containing the CXR jpgs.")
    g.add_argument("--output_dir", default="slot_features")

    g = p.add_argument_group("split")
    g.add_argument("--no_split", action="store_true", help="Use the entire CSV (no splitting).")
    g.add_argument("--train_split", type=float, default=0.8)
    g.add_argument("--val_split", type=float, default=0.1)
    g.add_argument("--test_split", type=float, default=0.1)
    g.add_argument("--stratify_column", default="clear lungs_bin")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("model")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--processor_id", default="google/medsiglip-448")
    g.add_argument("--num_classes_per_task", type=int, default=3)
    g.add_argument("--hidden_dim", type=int, default=128)
    g.add_argument("--slot_iters", type=int, default=3)
    g.add_argument("--reduced_dim", type=int, default=768)
    g.add_argument("--embedding_path", default=None)
    g.add_argument("--freeze_backbone", action=B, default=True)

    g = p.add_argument_group("runtime")
    g.add_argument("--batch_size", type=int, default=32)
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--device", default=None, help="cpu | cuda | hpu | ... (auto-detected if omitted)")
    return p.parse_args()


# ── data helpers (same split logic as the training script) ───────────────────
def create_stratified_test_split(df, test_split, stratify_column, seed):
    indices = np.arange(len(df))
    stratify = df[stratify_column].values if stratify_column in df.columns else None
    _, test_idx = train_test_split(indices, test_size=test_split, stratify=stratify,
                                   random_state=seed)
    return test_idx


def create_collate_fn(processor):
    def collate_fn(batch):
        processed = processor(images=[b["image"] for b in batch], return_tensors="pt")
        return {"image": processed.pixel_values,
                "labels": torch.stack([b["labels"] for b in batch])}
    return collate_fn


# ── capturing wrapper ────────────────────────────────────────────────────────
class SlotsCapturingModel(nn.Module):
    """Runs the exact MedSIGLIPSlotAttention.forward logic and stores `slots`
    (moved to CPU) after every call. The wrapped model is not modified."""

    def __init__(self, base_model: MedSIGLIPSlotAttention):
        super().__init__()
        self.base = base_model
        self.captured_slots: Optional[torch.Tensor] = None  # [B, T, D]

    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        vision_features = self.base.medsiglip.vision_model(pixel_values).last_hidden_state
        reduced = self.base.dim_reduction(vision_features)
        slots, attn = self.base.slot_attention(reduced)           # [B, T, D]
        self.captured_slots = slots.detach().cpu()

        logits = self.base.classifier(slots.reshape(b * self.base.num_tasks, -1)).reshape(
            b, self.base.num_tasks, self.base.num_classes_per_task)
        return {"logits": logits, "attentions": attn}


@torch.no_grad()
def extract_slots(model, loader, device, num_tasks):
    model.eval()
    slot_list, label_list, id_list, n_seen = [], [], [], 0
    for batch in tqdm(loader, desc="Extracting slots"):
        images, labels = batch["image"].to(device), batch["labels"]
        b = images.shape[0]
        model(images)
        slots = model.captured_slots
        assert slots is not None and slots.shape[0] == b and slots.shape[1] == num_tasks, \
            f"unexpected slot shape {None if slots is None else tuple(slots.shape)}"
        slot_list.append(slots.numpy().astype(np.float32))
        label_list.append(labels.numpy().astype(np.float32))
        id_list.append(np.arange(n_seen, n_seen + b, dtype=np.int64))
        n_seen += b
    return (np.concatenate(slot_list), np.concatenate(label_list), np.concatenate(id_list))


def main():
    args = parse_args()
    device = get_device(args.device)
    logger.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_file).dropna().reset_index(drop=True)
    if args.no_split:
        logger.info("--no_split: using entire CSV.")
        test_df = df
    else:
        logger.info("Recreating stratified split; extracting the test fold.")
        idx = create_stratified_test_split(df, args.test_split, args.stratify_column, args.seed)
        test_df = df.iloc[idx].reset_index(drop=True)
    logger.info(f"Test set size: {len(test_df)}")

    dataset = CXRDataset(df=test_df, root_dir=args.root_dir, transform=None, augment=False)
    disease_names = dataset.label_cols
    num_tasks = len(disease_names)
    logger.info(f"Tasks: {num_tasks} | first 5: {disease_names[:5]}")

    processor = AutoProcessor.from_pretrained(args.processor_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,  # keep order!
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=create_collate_fn(processor))

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    base_model = MedSIGLIPSlotAttention(
        num_tasks=num_tasks, num_classes_per_task=args.num_classes_per_task,
        hidden_dim=args.hidden_dim, slot_iters=args.slot_iters,
        freeze_backbone=args.freeze_backbone, lambda_attn_loss=0.0,
        reduced_dim=args.reduced_dim, disease_embeddings_path=args.embedding_path)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}  # strip DDP prefix
    base_model.load_state_dict(state)
    epoch = ckpt.get("epoch", "?")
    logger.info(f"Checkpoint loaded (epoch {epoch})")

    model = SlotsCapturingModel(base_model).to(device).eval()
    slot_features, labels, sample_ids = extract_slots(model, loader, device, num_tasks)
    N, T, D = slot_features.shape
    logger.info(f"Extracted slots {slot_features.shape} (N={N}, T={T}, D={D})")

    out = lambda f: os.path.join(args.output_dir, f)  # noqa: E731
    np.save(out("slot_features.npy"), slot_features)
    np.save(out("labels.npy"), labels)
    np.save(out("sample_ids.npy"), sample_ids)
    with open(out("disease_names.json"), "w") as f:
        json.dump(disease_names, f, indent=2)
    with open(out("metadata.json"), "w") as f:
        json.dump({
            "checkpoint": args.checkpoint, "epoch": epoch,
            "captured_variable": "slots — output of slot_attention(vision_features_reduced), "
                                 "[B, num_tasks, reduced_dim], before classifier",
            "slot_features_shape": [N, T, D], "labels_shape": [N, T],
            "label_encoding": {"1.0": "positive", "0.0": "uncertain", "-1.0": "negative"},
            "split": {"mode": "no_split" if args.no_split else "stratified_test_fold",
                      "train_split": args.train_split, "val_split": args.val_split,
                      "test_split": args.test_split, "stratify_column": args.stratify_column,
                      "seed": args.seed},
            "model": {"hidden_dim": args.hidden_dim, "slot_iters": args.slot_iters,
                      "reduced_dim": args.reduced_dim, "embedding_path": args.embedding_path},
        }, f, indent=2)
    logger.info(f"Saved slot_features.npy, labels.npy, sample_ids.npy, disease_names.json, "
                f"metadata.json -> {args.output_dir}")

    logger.info(f"Slot stats: mean={slot_features.mean():.4f} std={slot_features.std():.4f} "
                f"min={slot_features.min():.4f} max={slot_features.max():.4f}")
    flat = labels.flatten()
    for val, name in [(1.0, "positive"), (0.0, "uncertain"), (-1.0, "negative")]:
        cnt = int((flat == val).sum())
        logger.info(f"  {name:<10s}: {cnt:>8d} ({100 * cnt / len(flat):.1f}%)")


if __name__ == "__main__":
    main()