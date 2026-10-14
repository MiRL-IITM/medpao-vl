#!/usr/bin/env python
"""
MedGemma-4B chest X-ray report generation: zero-shot, LoRA fine-tuning, evaluation.

Modes (--mode)
--------------
  zeroshot    Evaluate the base model, no training.
  train       QLoRA fine-tune only (saves the adapter to <output-dir>/final).
  eval        Evaluate a trained adapter (--adapter-path).
  train_eval  Train, then evaluate the resulting adapter on the test set.

Concept conditioning (--use-concepts / --no-use-concepts, default: off)
  Off -> prompt contains only the image.
  On  -> prompt also lists the observed "Present / Absent" clinical concepts.
  The same switch applies to training, zero-shot and eval, so train and test
  prompts always match.
----------------

"""
import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any


# HF_HOME must be set before transformers / huggingface_hub are imported.
def _early_env() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--hf-home", default=None)
    known, _ = pre.parse_known_args()
    if known.hf_home:
        os.environ["HF_HOME"] = known.hf_home


_early_env()

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset, DatasetDict, Image as HFImage  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MedGemma report generation: zero-shot / fine-tune / evaluate.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    B = argparse.BooleanOptionalAction

    g = p.add_argument_group("General")
    g.add_argument("--mode", required=True,
                   choices=["zeroshot", "train", "eval", "train_eval"])
    g.add_argument("--use-concepts", action=B, default=False,
                   help="Add observed concepts to the prompt (train + test).")
    g.add_argument("--sections", choices=["findings", "findings_impression"],
                   default="findings", help="What the prompt asks the model to write.")
    g.add_argument("--model-id", default="google/medgemma-4b-it")
    g.add_argument("--hf-home", default=None, help="HF cache directory.")
    g.add_argument("--adapter-path", default=None,
                   help="LoRA adapter / checkpoint dir (required for --mode eval).")
    g.add_argument("--load-in-4bit", action=B, default=None,
                   help="4-bit NF4 quantisation. Default: on for training, off for eval.")
    g.add_argument("--run-name", default=None,
                   help="Name used for output files / W&B. Default is auto-generated.")
    g.add_argument("--seed", type=int, default=42)

    d = p.add_argument_group("Training data (train / train_eval)")
    d.add_argument("--train-csv"); d.add_argument("--val-csv")
    d.add_argument("--train-image-dir",
                   help="Dir of pre-converted jpgs, named <a>_<b>_<file>.jpg from the "
                        "last 3 components of the CSV image path.")
    d.add_argument("--train-image-col", default="pa_view_path")
    d.add_argument("--train-text-col", default="mimic_findings")
    d.add_argument("--train-concept-col", default="standardized_concepts")
    d.add_argument("--exclude-csv", default=None,
                   help="CSV whose --test-image-col paths are removed from train/val "
                        "(prevents train/test leakage).")
    d.add_argument("--stratify-col", default="is_long_report",
                   help="Column used for stratified subsampling (ignored if absent).")
    d.add_argument("--max-train-samples", type=int, default=None)
    d.add_argument("--val-size", type=int, default=500, help="Validation set size.")
    d.add_argument("--eval-subset", type=int, default=300,
                   help="Validation samples used for in-training evaluation.")

    t = p.add_argument_group("Test data (zeroshot / eval / train_eval)")
    t.add_argument("--test-csv")
    t.add_argument("--test-image-dir")
    t.add_argument("--test-image-col", default="image_path")
    t.add_argument("--test-text-col", default="finding")
    t.add_argument("--test-concept-col", default="standardized_concepts")
    t.add_argument("--test-name", default=None,
                   help="Tag used in result filenames. Default: test CSV stem.")
    t.add_argument("--max-test-samples", type=int, default=None)

    o = p.add_argument_group("Training hyper-parameters")
    o.add_argument("--output-dir", default=None,
                   help="Default: runs/<run-name>")
    o.add_argument("--epochs", type=int, default=5)
    o.add_argument("--lr", type=float, default=2e-4)
    o.add_argument("--train-batch-size", type=int, default=4)
    o.add_argument("--eval-batch-size", type=int, default=4)
    o.add_argument("--grad-accum", type=int, default=64)
    o.add_argument("--logging-steps", type=int, default=20)
    o.add_argument("--eval-steps", type=int, default=20)
    o.add_argument("--warmup-ratio", type=float, default=0.03)
    o.add_argument("--max-grad-norm", type=float, default=0.3)
    o.add_argument("--lora-r", type=int, default=64)
    o.add_argument("--lora-alpha", type=int, default=128)
    o.add_argument("--lora-dropout", type=float, default=0.05)
    o.add_argument("--report-to", default="wandb", help="wandb | tensorboard | none")
    o.add_argument("--wandb-project", default="huggingface")

    e = p.add_argument_group("Evaluation / generation")
    e.add_argument("--gen-batch-size", type=int, default=8)
    e.add_argument("--max-new-tokens", type=int, default=300)
    e.add_argument("--results-dir", default="results")

    a = p.parse_args()

    # ---- validation -------------------------------------------------------
    if a.mode in ("train", "train_eval"):
        for f in ("train_csv", "val_csv", "train_image_dir"):
            if not getattr(a, f):
                p.error(f"--{f.replace('_', '-')} is required for --mode {a.mode}")
    if a.mode in ("zeroshot", "eval", "train_eval"):
        for f in ("test_csv", "test_image_dir"):
            if not getattr(a, f):
                p.error(f"--{f.replace('_', '-')} is required for --mode {a.mode}")
    if a.mode == "eval" and not a.adapter_path:
        p.error("--adapter-path is required for --mode eval")

    tag = "concepts" if a.use_concepts else "noconcepts"
    a.run_name = a.run_name or f"medgemma4b_{a.mode}_{tag}"
    a.output_dir = a.output_dir or f"runs/medgemma4b_lora_{tag}"
    if a.test_csv:
        a.test_name = a.test_name or Path(a.test_csv).stem
    return a


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
_CONCEPT_RE = re.compile(r"([^(),]+)\((-?\d+(?:\.\d+)?)\)")


def format_concepts(concepts_str: Any) -> str:
    """'a(1), b(-1.0), c(1.0)' -> 'Present: a, c\\nAbsent: b'.
    Handles both '(1)' and '(1.0)' styles; other values (e.g. 0 = uncertain) are ignored."""
    if not isinstance(concepts_str, str) or not concepts_str.strip():
        return "No specific concepts identified"
    present, absent = [], []
    for name, val in _CONCEPT_RE.findall(concepts_str):
        name, val = name.strip(), float(val)
        if val == 1:
            present.append(name)
        elif val == -1:
            absent.append(name)
    lines = []
    if present:
        lines.append("Present: " + ", ".join(present))
    if absent:
        lines.append("Absent: " + ", ".join(absent))
    return "\n".join(lines) or "No specific concepts identified"


def build_prompt(concepts_str: Any, use_concepts: bool, sections: str) -> str:
    target = "findings" if sections == "findings" else "findings and impression"
    if use_concepts:
        return (
            "You are an expert radiologist analyzing a chest X-ray. Based on the image "
            "and the observed clinical concepts, generate a comprehensive radiology "
            f"report containing only the {target}.\n\n"
            f"Observed Concepts:\n{format_concepts(concepts_str)}\n\n"
            "Findings:"
        )
    return (
        "You are an expert radiologist analyzing a chest X-ray. Based on the image "
        f"features, generate a comprehensive radiology report containing only the {target}."
    )


def build_messages(concepts_str, finding, use_concepts, sections, with_answer):
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": build_prompt(concepts_str, use_concepts, sections)},
        ],
    }]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": finding}]})
    return msgs


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _to_hf_dataset(images, findings, concepts) -> Dataset:
    ds = Dataset.from_pandas(pd.DataFrame(
        {"image": images, "finding": findings, "concepts_": concepts}))
    return ds.cast_column("image", HFImage())


def _subsample(df: pd.DataFrame, n: int | None, strat_col: str, seed: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    strat = df[strat_col] if strat_col in df.columns else None
    sub, _ = train_test_split(df, train_size=n, stratify=strat, random_state=seed)
    return sub.reset_index(drop=True)


def _train_image_path(rel: str, image_dir: str) -> str:
    parts = rel.split("/")
    return os.path.join(image_dir, f"{parts[-3]}_{parts[-2]}_{parts[-1]}.jpg")


def build_train_val(a) -> DatasetDict:
    train_df, val_df = pd.read_csv(a.train_csv), pd.read_csv(a.val_csv)

    if a.exclude_csv:
        excluded = set(pd.read_csv(a.exclude_csv)[a.test_image_col])
        keep = lambda df: ~df[a.train_image_col].apply(  # noqa: E731
            lambda x: x.split("files/")[-1]).isin(excluded)
        n0 = len(train_df) + len(val_df)
        train_df, val_df = train_df[keep(train_df)], val_df[keep(val_df)]
        print(f"Excluded {n0 - len(train_df) - len(val_df)} test-overlapping rows.")

    val_df = _subsample(val_df, a.val_size, a.stratify_col, a.seed)
    train_df = _subsample(train_df, a.max_train_samples, a.stratify_col, a.seed)

    def make(df):
        concepts = (df[a.train_concept_col].fillna("")
                    if a.train_concept_col in df.columns else [""] * len(df))
        return _to_hf_dataset(
            [_train_image_path(x, a.train_image_dir) for x in df[a.train_image_col]],
            df[a.train_text_col].tolist(), list(concepts))

    data = DatasetDict({"train": make(train_df), "validation": make(val_df)})
    print(data)
    return data


def build_test_dataset(a) -> Dataset:
    df = pd.read_csv(a.test_csv)
    if a.max_test_samples:
        df = df.sample(n=min(a.max_test_samples, len(df)), random_state=a.seed)
    df = df.reset_index(drop=True)
    concepts = (df[a.test_concept_col].fillna("")
                if a.test_concept_col in df.columns else [""] * len(df))
    if a.use_concepts and a.test_concept_col not in df.columns:
        raise ValueError(f"--use-concepts set but '{a.test_concept_col}' not in {a.test_csv}")
    return _to_hf_dataset(
        [os.path.join(a.test_image_dir, os.path.basename(x)) for x in df[a.test_image_col]],
        df[a.test_text_col].tolist(), list(concepts))


# ---------------------------------------------------------------------------
# Model / processor
# ---------------------------------------------------------------------------
def check_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("GPU does not support bfloat16; use an Ampere+ GPU.")


def load_processor(a, padding_side: str):
    processor = AutoProcessor.from_pretrained(a.model_id)
    processor.tokenizer.padding_side = padding_side
    return processor


def load_model(a, quantize: bool, adapter_path: str | None = None):
    check_gpu()
    kwargs = dict(attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="auto")
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    model = AutoModelForImageTextToText.from_pretrained(a.model_id, **kwargs)
    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainCollator:
    """Builds chat prompts on the fly, tokenises with images, and masks
    padding / image tokens out of the loss."""

    def __init__(self, processor, use_concepts: bool, sections: str):
        self.processor, self.use_concepts, self.sections = processor, use_concepts, sections
        tok = processor.tokenizer
        boi = getattr(tok, "boi_token", None) or tok.special_tokens_map.get("boi_token")
        self.mask_ids = {
            tok.pad_token_id,
            getattr(processor, "image_token_id", 262144),  # Gemma-3 <image_soft_token>
        }
        if boi:
            self.mask_ids.add(tok.convert_tokens_to_ids(boi))
        self.mask_ids.discard(None)

    def __call__(self, examples: list[dict[str, Any]]):
        texts, images = [], []
        for ex in examples:
            msgs = build_messages(ex["concepts_"], ex["finding"],
                                  self.use_concepts, self.sections, with_answer=True)
            texts.append(self.processor.apply_chat_template(
                msgs, add_generation_prompt=False, tokenize=False).strip())
            images.append([ex["image"].convert("RGB")])
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        for tid in self.mask_ids:
            labels[labels == tid] = -100
        batch["labels"] = labels
        return batch


def run_training(a) -> str:
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    if a.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = a.wandb_project
        os.environ["WANDB_RUN_NAME"] = a.run_name

    data = build_train_val(a)
    processor = load_processor(a, padding_side="right")  # right padding for training
    model = load_model(a, quantize=True if a.load_in_4bit is None else a.load_in_4bit)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    peft_config = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        bias="none", target_modules="all-linear", task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )
    cfg = SFTConfig(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.train_batch_size,
        per_device_eval_batch_size=a.eval_batch_size,
        gradient_accumulation_steps=a.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=a.lr,
        lr_scheduler_type="linear",
        warmup_ratio=a.warmup_ratio,
        max_grad_norm=a.max_grad_norm,
        bf16=True,
        logging_steps=a.logging_steps,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=a.eval_steps,
        push_to_hub=False,
        report_to=a.report_to if a.report_to != "none" else [],
        run_name=a.run_name,
        seed=a.seed,
        dataset_kwargs={"skip_prepare_dataset": True},  # collator does the preprocessing
        remove_unused_columns=False,                    # collator needs raw columns
        label_names=["labels"],
    )
    val = data["validation"].shuffle(seed=a.seed)
    val = val.select(range(min(a.eval_subset, len(val))))

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=data["train"], eval_dataset=val,
        peft_config=peft_config, processing_class=processor,
        data_collator=TrainCollator(processor, a.use_concepts, a.sections),
    )
    trainer.train()

    final_dir = os.path.join(a.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Saved final adapter to {final_dir}")

    del trainer, model
    free_gpu()
    return final_dir


# ---------------------------------------------------------------------------
# Generation + metrics
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate_reports(model, processor, dataset: Dataset, a) -> list[str]:
    model.eval()
    preds = []
    bs = a.gen_batch_size
    for i in tqdm(range(0, len(dataset), bs), desc="Generating"):
        rows = dataset[i:i + bs]
        texts = [
            processor.apply_chat_template(
                build_messages(c, None, a.use_concepts, a.sections, with_answer=False),
                add_generation_prompt=True, tokenize=False)
            for c in rows["concepts_"]
        ]
        images = [[img.convert("RGB")] for img in rows["image"]]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
        out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[-1]:]
        preds.extend(t.strip() for t in processor.batch_decode(new_tokens, skip_special_tokens=True))
    return preds


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    import evaluate
    bleu, rouge, meteor = evaluate.load("bleu"), evaluate.load("rouge"), evaluate.load("meteor")
    nested = [[r] for r in references]
    m = {f"bleu{n}": bleu.compute(predictions=predictions, references=nested, max_order=n)["bleu"]
         for n in range(1, 5)}
    r = rouge.compute(predictions=predictions, references=references)
    m.update(rouge1=r["rouge1"], rouge2=r["rouge2"], rougeL=r["rougeL"],
             meteor=meteor.compute(predictions=predictions, references=references)["meteor"])
    return m


def run_evaluation(a, adapter_path: str | None) -> dict[str, float]:
    check_gpu()
    processor = load_processor(a, padding_side="left")  # left padding for generation
    quantize = bool(a.load_in_4bit)  # default off for eval
    model = load_model(a, quantize=quantize, adapter_path=adapter_path)

    test_ds = build_test_dataset(a)
    print(f"Evaluating on {len(test_ds)} samples | adapter={adapter_path or 'none (zero-shot)'} "
          f"| concepts={'on' if a.use_concepts else 'off'}")

    raw_preds = generate_reports(model, processor, test_ds, a)
    refs = [_clean(r) for r in test_ds["finding"]]
    preds = [_clean(p) for p in raw_preds]

    metrics = compute_metrics(preds, refs)

    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.run_name}__{a.test_name}"
    pd.DataFrame({"reference": refs, "prediction": preds,
                  "concepts": test_ds["concepts_"]}).to_csv(out_dir / f"{stem}_predictions.csv", index=False)
    with open(out_dir / f"{stem}_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "n_samples": len(preds), "adapter": adapter_path,
                   "config": {k: v for k, v in vars(a).items()}}, f, indent=2, default=str)

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>8}: {v:.4f}")
    print(f"\nSaved to {out_dir}/{stem}_{{predictions.csv,metrics.json}}")

    del model
    free_gpu()
    return metrics


# ---------------------------------------------------------------------------
def main() -> None:
    a = parse_args()
    torch.manual_seed(a.seed)

    adapter = a.adapter_path
    if a.mode in ("train", "train_eval"):
        adapter = run_training(a)
    if a.mode in ("zeroshot", "eval", "train_eval"):
        run_evaluation(a, adapter_path=None if a.mode == "zeroshot" else adapter)


if __name__ == "__main__":
    main()