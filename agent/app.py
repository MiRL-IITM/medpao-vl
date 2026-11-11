"""Gradio demo for MedPAO-VL: upload a chest X-ray, get predicted concepts, attention maps and a structured report.

Run from this folder:

    pip install "gradio>=4.44,<5" "pydantic<2.11"   # Python 3.9 (Gradio 4.44 breaks with pydantic >= 2.11)
    # Python >= 3.10: pip install gradio
    python app.py                       # http://127.0.0.1:7860
    python app.py --share               # temporary public link

The report stages need the same LLM servers and index files as the batch scripts (see README.md).
Research use only: not a medical device, not validated for clinical use.
"""
from __future__ import annotations

import argparse
import threading
import traceback

import gradio as gr
from PIL import Image

import inference as mp
from inference import (DEFAULT_MODE, MIN_CONCEPT_PROB, TOP_K_OVERLAYS, VISION_MODEL, final_report_text,
                       structured_report_markdown)

# ───────────────────────────── CONFIG (edit here) ─────────────────────────────
SERVER_NAME = "127.0.0.1"          # use 0.0.0.0 to serve on the local network
SERVER_PORT = 7860
MAX_OVERLAYS = 12                  # upper bound of the "attention maps" slider
# ──────────────────────────────────────────────────────────────────────────────

DISCLAIMER = ("**Research use only.** This demo is not a medical device and has not been validated for clinical use. "
              "Do not use its output for diagnosis or patient management.")
MODE_HELP = {
    "deterministic": "Single pass: grounded report → concepts → SNOMED CT → ABCDEF structured report.",
    "cold": "Plan-Act-Observe loop (self-correction), no concept-cache lookup.",
    "warm": "Plan-Act-Observe loop with the concept cache (skips ontology mapping for known concepts).",
}

_predictor = None
_lock = threading.Lock()   # the agent scripts share one scratch folder, so requests run one at a time


def _get_predictor(model_path: str):
    global _predictor
    if _predictor is None:
        _predictor = mp.VisionConceptPredictor(model_path)
    return _predictor


def analyze(image_path, mode, top_k, min_prob, concepts_only, model_path):
    if not image_path:
        raise gr.Error("Upload a chest X-ray first.")
    with _lock:
        try:
            result = mp.run_inference(image_path, mode, _get_predictor(model_path), float(min_prob), int(top_k),
                                      bool(concepts_only))
        except gr.Error:
            raise
        except Exception as e:   # show the failing step in the UI, keep the full trace in the console
            traceback.print_exc()
            raise gr.Error(f"{type(e).__name__}: {e}")

    present = ", ".join(f"{n} ({p:.2f})" for n, p in result["present"]) or "–"
    absent = ", ".join(f"{n} ({p:.2f})" for n, p in result["absent"]) or "–"
    concepts_md = f"**Present:** {present}\n\n**Absent:** {absent}"

    agent = result["agent"]
    grounded = agent["grounded_report"] if agent else ""
    final = final_report_text(agent) if agent else ""
    structured = structured_report_markdown(agent.get("structured_report")) if agent else ""
    details = {"vision_concepts": result["vision_concepts"], "timings": result["timings"]}
    if agent:
        for key in ("missing_concepts", "iterations", "correction_log", "cache_hit", "cache_miss"):
            if key in agent:
                details[key] = agent[key]
        details["agent_timings"] = agent.get("timings")
    return concepts_md, result["overlays"], grounded, final, structured, details


def build_demo(model_path: str = VISION_MODEL) -> gr.Blocks:
    with gr.Blocks(title="MedPAO-VL") as demo:
        gr.Markdown("# MedPAO-VL\nInterpretable, protocol-compliant structured radiology report generation from a "
                    "chest X-ray.\n\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Chest X-ray")
                mode = gr.Dropdown(list(mp.AGENT_MODULES), value=DEFAULT_MODE, label="Agent mode",
                                   info=MODE_HELP[DEFAULT_MODE])
                top_k = gr.Slider(1, MAX_OVERLAYS, value=TOP_K_OVERLAYS, step=1, label="Attention maps to show")
                min_prob = gr.Slider(0.0, 0.95, value=MIN_CONCEPT_PROB, step=0.05,
                                     label="Minimum concept probability",
                                     info="Concepts below this confidence are dropped before report generation.")
                concepts_only = gr.Checkbox(False, label="Vision step only (no report; no LLM servers needed)")
                run = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                concepts = gr.Markdown(label="Predicted concepts")
                gallery = gr.Gallery(label="Attention maps (most confident present concepts)", columns=3, height="auto")
                with gr.Tabs():
                    with gr.Tab("Structured report"):
                        structured = gr.Markdown()
                    with gr.Tab("Report"):
                        final = gr.Textbox(lines=8, label="Report (after self-correction in cold/warm modes)")
                    with gr.Tab("Grounded report (first draft)"):
                        grounded = gr.Textbox(lines=8, label="Grounded report")
                    with gr.Tab("Details"):
                        details = gr.JSON(label="Concepts, correction log, timings")

        mode.change(lambda m: gr.update(info=MODE_HELP[m]), mode, mode, api_name=False)
        run.click(lambda *a: analyze(*a, model_path),
                  [image, mode, top_k, min_prob, concepts_only],
                  [concepts, gallery, grounded, final, structured, details], api_name="generate")
    return demo


def main(argv=None):
    p = argparse.ArgumentParser(description="MedPAO-VL Gradio demo (research use only).")
    p.add_argument("--vision-model", default=VISION_MODEL, help="local folder or Hub repo id of the vision model")
    p.add_argument("--host", default=SERVER_NAME)
    p.add_argument("--port", type=int, default=SERVER_PORT)
    p.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    a = p.parse_args(argv)
    demo = build_demo(a.vision_model)
    demo.queue(default_concurrency_limit=1).launch(server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
