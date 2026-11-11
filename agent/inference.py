"""End-to-end MedPAO-VL inference for a single chest X-ray.

    CXR image -> vision concept model -> "name(1.0), name(-1.0)" string -> agent -> structured report

Run from this folder (the agent scripts import ``tools/``):

    python inference.py --image cxr.jpg                      # deterministic agent (default)
    python inference.py --image cxr.jpg --mode warm          # PAO loop + concept cache
    python inference.py --image cxr.jpg --concepts-only      # vision step only, no LLM servers needed

LLM calls go to the HTTP servers configured in tools/*.py by default. To load the LLMs in this process instead:

    python inference.py --image cxr.jpg --vlm-model <path> [--vlm-adapter <lora>] \
        [--text-model <base>] --extractor-adapter <lora> --categorizer-adapter <lora>

The agent stage also needs the SNOMED index and SapBERT files (see README.md).
Research use only: not a medical device, not validated for clinical use.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))   # tools/ lives next to this file
from tools import llm_backend  # noqa: E402

# ───────────────────────────── CONFIG (edit here) ─────────────────────────────
# Local folder (converted model) or Hugging Face repo id. The Hub repo is gated: log in with `hf auth login`.
VISION_MODEL     = os.environ.get("MEDPAO_VISION_MODEL", "shrishSVaidya/slot-attn-cxr-medsiglip")
DEFAULT_MODE     = "deterministic"   # deterministic | cold | warm
MIN_CONCEPT_PROB = 0.6               # keep a present/absent concept only if its probability is >= this (0 = plain argmax)
TOP_K_OVERLAYS   = 6                 # attention maps drawn for the most confident present concepts
OVERLAY_ALPHA    = 0.6               # strength of the attention heat map
ATTN_GRID        = 32                # 448 px input / 14 px patches
OUTPUT_DIR       = "inference_outputs"
SCRATCH_DIR      = "inference_outputs/_agent_scratch"   # intermediate agent files (the batch scripts' SAVINGS_DIR)

PRESENT, UNCERTAIN, ABSENT = 0, 1, 2            # class index of the vision model
AGENT_MODULES = {                                # mode -> (script, single-image function)
    "deterministic": ("deterministic",     "run_deterministic_pipeline"),
    "cold":          ("react_cold_cache",  "run_pao_pipeline"),
    "warm":          ("react_warm_cache",  "run_pao_pipeline"),
}
PROTOCOL_NAMES = {"P": "Projection", "A": "Airways", "B": "Breathing", "C": "Circulation",
                  "D": "Diaphragm and below", "E": "External", "F": "Foreign material"}
# ──────────────────────────────────────────────────────────────────────────────

AGENT_DIR = Path(__file__).resolve().parent


# ── Vision concept model ─────────────────────────────────────────────────────
@dataclass
class ConceptPrediction:
    names: List[str]          # [T] concept names
    probs: np.ndarray         # [T, 3] softmax over (present, uncertain, absent)
    attn: np.ndarray          # [T, P] attention over the 32 x 32 patch grid

    @property
    def pred(self) -> np.ndarray:
        return self.probs.argmax(-1)

    def _select(self, cls: int, min_prob: float = 0.0) -> List[Tuple[str, float]]:
        hits = [(self.names[i], float(self.probs[i, cls])) for i in np.where(self.pred == cls)[0]
                if self.probs[i, cls] >= min_prob]
        return sorted(hits, key=lambda x: -x[1])

    def present(self, min_prob: float = 0.0) -> List[Tuple[str, float]]:
        return self._select(PRESENT, min_prob)

    def absent(self, min_prob: float = 0.0) -> List[Tuple[str, float]]:
        return self._select(ABSENT, min_prob)

    def concept_string(self, min_prob: float = 0.0) -> str:
        """The format the agent consumes: 'atelectasis(1.0), pleural effusion(-1.0)'. Uncertain concepts are left out."""
        parts = [f"{n}(1.0)" for n, _ in self.present(min_prob)] + [f"{n}(-1.0)" for n, _ in self.absent(min_prob)]
        return ", ".join(parts)

    def overlays(self, image: Image.Image, k: int = TOP_K_OVERLAYS, min_prob: float = 0.0,
                 alpha: float = OVERLAY_ALPHA) -> List[Tuple[Image.Image, str]]:
        """Attention heat maps for the k most confident present concepts."""
        idx = {n: i for i, n in enumerate(self.names)}
        out = []
        for name, p in self.present(min_prob)[:k]:
            out.append((attention_overlay(image, self.attn[idx[name]], alpha), f"{name} ({p:.2f})"))
        return out


class VisionConceptPredictor:
    """Loads the converted slot-attention model with AutoModel (no Habana software needed)."""

    def __init__(self, model_name_or_path: str = VISION_MODEL, device: Optional[str] = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.names = list(getattr(self.model.config, "task_names", None) or [])
        if not self.names:
            raise ValueError(f"{model_name_or_path} has no task_names in its config; cannot name the concepts.")

    def predict(self, image: Image.Image) -> ConceptPrediction:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            out = self.model(**inputs)
        return ConceptPrediction(
            names=self.names,
            probs=out.logits.softmax(-1)[0].float().cpu().numpy(),
            attn=out.attentions[0].float().cpu().numpy(),
        )


def _jet(x: np.ndarray) -> np.ndarray:
    """Matplotlib-free 'jet' colour map, x in [0, 1] -> RGB in [0, 1]."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], -1)


def attention_overlay(image: Image.Image, attn: np.ndarray, alpha: float = OVERLAY_ALPHA) -> Image.Image:
    """Overlay one concept's attention (flattened 32 x 32 grid) on the image."""
    grid = int(round(attn.size ** 0.5))
    a = attn.reshape(grid, grid).astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    heat = Image.fromarray((a * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    w = np.asarray(heat, dtype=np.float32)[..., None] / 255.0
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mix = (1 - alpha * w) * base + alpha * w * _jet(w[..., 0])
    return Image.fromarray((np.clip(mix, 0, 1) * 255).astype(np.uint8))


# ── Agent stage ──────────────────────────────────────────────────────────────
def run_agent(mode: str, image_path: str, concepts_str: str) -> dict:
    """Run one of the batch scripts' single-image functions. Importing a script loads its heavy
    components (SNOMED index, SapBERT), so this happens lazily, once per mode."""
    if mode not in AGENT_MODULES:
        raise ValueError(f"mode must be one of {list(AGENT_MODULES)}, got {mode!r}")
    if str(AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT_DIR))
    module_name, fn_name = AGENT_MODULES[mode]
    module = importlib.import_module(module_name)
    image_path = str(Path(image_path).resolve())
    module.IMG_FOLDER = str(Path(image_path).parent)     # the scripts look the image up as IMG_FOLDER/<file name>
    scratch = Path(SCRATCH_DIR).resolve() / mode          # the scripts' own SAVINGS_DIR defaults are paths of the original machine
    scratch.mkdir(parents=True, exist_ok=True)
    module.SAVINGS_DIR = str(scratch)
    if hasattr(module, "CACHE_DIR"):
        module.CACHE_DIR = str(scratch) + module.CACHE_DIR_SUFFIX   # warm mode: persistent concept cache lives next to the scratch folder
    module.ontology_mapper.saving_dir = str(scratch)
    result = getattr(module, fn_name)(image_path, concepts_str)
    if result is None:
        raise RuntimeError("The agent skipped this image (report generation, concept extraction or the cache "
                           "check failed). See the console log for the failing step.")
    return result


def final_report_text(agent_result: dict) -> str:
    """The free-text report after self-correction (the deterministic baseline has no corrected version)."""
    return agent_result.get("full_report") or agent_result.get("full report") or agent_result.get("grounded_report", "")


# ── End to end ───────────────────────────────────────────────────────────────
def run_inference(image_path: str, mode: str = DEFAULT_MODE, predictor: Optional[VisionConceptPredictor] = None,
                  min_prob: float = MIN_CONCEPT_PROB, top_k: int = TOP_K_OVERLAYS,
                  concepts_only: bool = False) -> dict:
    predictor = predictor or VisionConceptPredictor()
    image = Image.open(image_path).convert("RGB")
    timings: Dict[str, float] = {}

    t = time.perf_counter()
    prediction = predictor.predict(image)
    concepts_str = prediction.concept_string(min_prob)
    timings["vision_time_sec"] = round(time.perf_counter() - t, 4)
    print("\nVision model concepts (probability >= %.2f):" % min_prob)
    print("  Present :", ", ".join(f"{n} ({p:.2f})" for n, p in prediction.present(min_prob)) or "-")
    print("  Absent  :", ", ".join(f"{n} ({p:.2f})" for n, p in prediction.absent(min_prob)) or "-")
    print("  Concept string sent to the VLM:", concepts_str or "-")

    result = {
        "image_path": str(image_path),
        "mode": mode,
        "vision_concepts": concepts_str,
        "present": prediction.present(min_prob),
        "absent": prediction.absent(min_prob),
        "overlays": prediction.overlays(image, top_k, min_prob),
        "agent": None,
        "timings": timings,
    }
    if concepts_only:
        return result
    if not concepts_str:
        raise RuntimeError("The vision model found no present or absent concepts above the probability "
                           "threshold, so there is nothing to ground a report on.")

    t = time.perf_counter()
    result["agent"] = run_agent(mode, image_path, concepts_str)
    timings["agent_time_sec"] = round(time.perf_counter() - t, 4)
    return result


def structured_report_markdown(structured: Optional[dict]) -> str:
    if not isinstance(structured, dict):
        return str(structured or "")
    lines = []
    for key in "PABCDEF":
        if key in structured:
            lines.append(f"**{key} – {PROTOCOL_NAMES[key]}:** {structured[key]}")
    extra = [k for k in structured if k not in PROTOCOL_NAMES]
    lines += [f"**{k}:** {structured[k]}" for k in extra]
    return "\n\n".join(lines)


def save_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for img, caption in result["overlays"]:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", caption.split(" (")[0])[:60]
        img.save(out_dir / f"attention_{name}.png")
    serializable = {k: v for k, v in result.items() if k != "overlays"}
    with open(out_dir / "result.json", "w") as f:
        json.dump(serializable, f, indent=4, default=str)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MedPAO-VL: single-image inference (research use only).")
    p.add_argument("--image", required=True, help="path to a chest X-ray (jpg/png)")
    p.add_argument("--mode", choices=list(AGENT_MODULES), default=DEFAULT_MODE,
                   help="deterministic: single pass | cold: PAO loop | warm: PAO loop + concept cache")
    p.add_argument("--vision-model", default=VISION_MODEL, help="local folder or Hub repo id of the vision model")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--min-prob", type=float, default=MIN_CONCEPT_PROB,
                   help="drop present/absent concepts whose probability is below this")
    p.add_argument("--top-k", type=int, default=TOP_K_OVERLAYS, help="number of attention maps to save")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--concepts-only", action="store_true", help="run only the vision step (no LLM servers needed)")
    llm_backend.add_cli_args(p)
    a = p.parse_args(argv)

    llm_backend.configure_from_args(a)
    predictor = VisionConceptPredictor(a.vision_model, a.device)
    if not a.concepts_only:
        llm_backend.preload()      # local backend: load every LLM now (no-op for the server backend)
    result = run_inference(a.image, a.mode, predictor, a.min_prob, a.top_k, a.concepts_only)
    out_dir = Path(a.output_dir) / Path(a.image).stem
    save_outputs(result, out_dir)

    print("\nPresent :", ", ".join(f"{n} ({p:.2f})" for n, p in result["present"]) or "-")
    print("Absent  :", ", ".join(f"{n} ({p:.2f})" for n, p in result["absent"]) or "-")
    if result["agent"]:
        print("\nReport:\n" + final_report_text(result["agent"]))
        print("\nStructured report:\n" + structured_report_markdown(result["agent"].get("structured_report")))
    print(f"\nTimings: {result['timings']}\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
