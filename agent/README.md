# CXR report agents

Pipelines that turn a chest X-ray plus the concepts predicted by the vision model (e.g. `atelectasis(1.0), pleural effusion(-1.0)`) into a **structured P-A-B-C-D-E-F report**. Each image goes through the steps below; the three scripts differ only in which steps they use.

```
image + vision concepts
  └─ grounded report ─ extract concepts ─ [self-correction] ─ [cache check] ─ SNOMED mapping ─ categorize (PABCDEF) ─ structured report
```

| Script | Self-correction | Concept cache | Use it for |
|---|:-:|:-:|---|
| `deterministic.py` | no | no | Baseline: one straight pass |
| `react_cold_cache.py` | yes | not read (builds the vocabulary) | Full agent, every concept is mapped from scratch |
| `react_warm_cache.py` | yes | yes | Full agent, already-seen concepts are reused from the cache |

**Self-correction:** the extracted concepts are compared with the vision concepts; for any that are missing, the VLM writes extra lines for just those concepts, concepts are re-extracted, and the check repeats (max 3 passes, stops early if nothing improves).

## Files

| File | What it does |
|---|---|
| `deterministic.py` | Baseline pipeline, no self-correction, no cache. |
| `react_cold_cache.py` | Pipeline with the self-correction loop. Does not look up the cache; the categorization step still appends to `cached_vocab.json`. |
| `react_warm_cache.py` | Same as cold, plus a cache check: concepts already in `cached_vocab.json` skip ontology mapping and categorization. Wipes its scratch folder before every image. |
| `inference.py` | Single-image, end-to-end inference: CXR image → vision concept model → agent → structured report (CLI). |
| `app.py` | Gradio demo of the same pipeline: upload a CXR, see concepts, attention maps and the structured report. |
| `tools/generate_report.py` | Asks the VLM (image + concepts) for the findings; also writes the extra lines for missing concepts. |
| `tools/get_concept.py` | Asks an LLM to pull `{concept: source sentence}` out of a report. |
| `tools/compare_concepts.py` | Compares vision concepts with report concepts using SapBERT similarity; returns the missing ones. |
| `tools/check_cache.py` | Fuzzy-matches concepts against the cached vocabulary; splits them into hits and new concepts. |
| `tools/ontology_map_sapbert.py` | Maps concepts to SNOMED CT with SapBERT embeddings + a FAISS index (`SnomedMatcher`). |
| `tools/categorize_concepts.py` | Asks an LLM to assign each concept to P/A/B/C/D/E/F using its SNOMED ancestors. |
| `tools/generate_structured_report.py` | Asks an LLM to write the final per-category report (JSON). |

## Setup

1. **Python packages:** `pandas tqdm pillow requests numpy huggingface_hub transformers torch scikit-learn faiss-cpu llama-cpp-python psutil rapidfuzz` (`rapidfuzz` is only needed by the warm script).
2. **Three TGI-compatible servers** must be running (or, instead, give model paths to load the LLMs locally: see [Local models instead of servers](#local-models-instead-of-servers)) (change the URLs at the top of each tool if yours differ):

   | Port | Used by | Type |
   |---|---|---|
   | 8082 | `generate_report.py` | multimodal (image + text) |
   | 8081 | `get_concept.py` | text generation |
   | 8080 | `categorize_concepts.py`, `generate_structured_report.py` | OpenAI-compatible chat |

3. **SNOMED index files and `sapbert.gguf`:** set `SAVE_DIR` and `SAPBERT_GGUF` at the top of `tools/ontology_map_sapbert.py`.
4. **Input CSV** with an `image_path` column and a `standardized_concepts` column (format: `name(1.0)` present, `name(-1.0)` absent, comma separated), plus a folder with the images (matched by file name).
5. **Edit the `CONFIG` block** at the top of the script you run: `SAVINGS_DIR`, `IMG_FOLDER`, `INPUT_CSV`, row range, etc. The defaults are paths from the original machine.

## Run

The scripts import `tools/`, so run them from this folder:

```bash
cd agent

python deterministic.py        # baseline            -> entire_results_deterministic.json
python react_cold_cache.py     # agent, no cache     -> entire_results_cold_cache.json
python react_warm_cache.py     # agent, with cache   -> entire_results_warm_cache.json
```

Each run processes the first half of the CSV by default (`ROW_START` / `ROW_END` in the `CONFIG` block). Partial results are saved every few images, and the final file is written to the current folder.

Quick check of the SNOMED matcher on its own:

```bash
python tools/ontology_map_sapbert.py
```

## Inference on a single image

The batch scripts above take the vision concepts from a CSV. `inference.py` runs the whole chain on one chest X-ray: it predicts the concepts with the vision model, turns them into the `name(1.0), name(-1.0)` string (uncertain concepts are dropped), and passes them to the chosen agent.

```bash
cd agent
python inference.py --image cxr.jpg                     # deterministic agent (default)
python inference.py --image cxr.jpg --mode cold         # PAO loop, no cache lookup
python inference.py --image cxr.jpg --mode warm         # PAO loop + concept cache
python inference.py --image cxr.jpg --concepts-only     # vision step only (no LLM servers needed)
```

Results (JSON, attention maps) are saved in `inference_outputs/<image name>/`. Useful options: `--vision-model` (local folder or Hub repo id, default `shrishSVaidya/slot-attn-cxr-medsiglip`, also settable with `MEDPAO_VISION_MODEL`), `--min-prob` (drop low-confidence concepts), `--top-k` (number of attention maps), `--device`.

The vision model is loaded with `AutoModel` (no Habana software needed). The Hub repo is gated: request access, then `hf auth login`. The agent stage needs the same servers and files as the batch scripts.

### Gradio demo

```bash
pip install "gradio>=4.44,<5" "pydantic<2.11"    # Python 3.9; on Python >= 3.10 just: pip install gradio
cd agent
python app.py                                     # http://127.0.0.1:7860
```

Upload a chest X-ray, pick the agent mode (default `deterministic`) and press **Generate**. The page shows the predicted concepts, attention maps for the most confident present concepts, the structured ABCDEF report, the free-text report (and the first draft), and the correction log and timings. Tick *Vision step only* to try the interface without the LLM servers. Requests are processed one at a time because the agent scripts share one scratch folder. Options: `--vision-model`, `--host`, `--port`, `--share`.

## Local models instead of servers

By default the agent calls the three TGI servers. It can instead **load the LLMs in the same process** from model paths. Give the paths and every model is loaded up front (`preload`), then used whenever a step needs it. The server mode stays the default and is unchanged.

Two model groups are needed:

| Role | Used for | Replaces |
|---|---|---|
| `vlm` (+ optional LoRA `vlm-adapter`) | grounded report, missing-concept lines | server on port 8082 |
| `text` base: `Qwen/Qwen3-4B-Instruct-2507` (loaded once) | shared by the three steps below | servers on 8081 and 8080 |
| ↳ concept extraction | raw text completion, **extractor LoRA** | port 8081 |
| ↳ concept categorization | chat, **categorizer LoRA** | port 8080 |
| ↳ structured report | chat, plain base model (no LoRA) | port 8080 |

The two LoRAs are attached to the same base weights and switched per call, so the text side costs one 4B model in memory. Set the base id and the two adapter paths at the top of `tools/llm_backend.py` (`TEXT_BASE_MODEL`, `EXTRACTOR_ADAPTER`, `CATEGORIZER_ADAPTER`), or pass them as flags / environment variables. Each path can be a local folder or a Hub id. SapBERT is unchanged.

```bash
cd agent

# one image
python inference.py --image cxr.jpg \
    --vlm-model google/medgemma-4b-it --vlm-adapter ../hf_upload/hf_lora_cxr \
    --extractor-adapter <lora> --categorizer-adapter <lora>      # [--text-model <base>] to change the base

# Gradio demo (models are loaded at start-up; --no-preload loads them on the first request instead)
python app.py --vlm-model <path> --vlm-adapter <lora> --extractor-adapter <lora> --categorizer-adapter <lora>

# batch scripts take the same settings from environment variables
MEDPAO_BACKEND=local MEDPAO_VLM_MODEL=<path> MEDPAO_VLM_ADAPTER=<lora> \
MEDPAO_EXTRACTOR_ADAPTER=<lora> MEDPAO_CATEGORIZER_ADAPTER=<lora> python deterministic.py
```

Other options: `--llm-dtype` (default `auto`: bfloat16 on GPUs that support it, float16 otherwise, bfloat16 on CPU), `--device-map` (default `auto`, spreads large models over all GPUs), `--load-in-4bit` (needs `bitsandbytes`). Giving any model/adapter path switches to the local backend; `--backend server` forces the servers.

Notes:
- The local models need GPU memory for the VLM plus one 4B text model (the two LoRAs are small); use `--load-in-4bit` if tight.
- Text generations run one at a time (the LoRAs are switched on the shared model).
- Server-only options such as the OpenAI-style `frequency_penalty` are reproduced with a standard logits penalty, so outputs can differ slightly from the servers. The values are the `*_PENALTY` constants at the top of `tools/generate_report.py`.
- Replies wrapped in a markdown fence are handled as before: the JSON is found with a regex.
- The SNOMED index and SapBERT files are still required (they were never served over HTTP).

## Output

One entry per image: the vision concepts, the grounded report, extracted concepts, SNOMED matches, categorized concepts, the structured report and per-step timings. The agent scripts also include `missing_concepts` (left after correction), `iterations` (correction passes) and `correction_log` (what each pass added). The warm script adds `cache_hit` / `cache_miss`.

`deterministic.py` and `react_warm_cache.py` skip images whose report or concept extraction fails (they do not appear in the output). `react_cold_cache.py` does not skip them yet; a failed image still produces an entry.

## Notes

- **Reusing the cache:** the warm script reads `<SAVINGS_DIR>_cache/cached_vocab.json`. Cold and warm have different `SAVINGS_DIR` values, so their caches are separate. Copy the file over (or use the same `SAVINGS_DIR` base) to start warm with a filled cache.
- **Scratch files:** intermediate JSON files go in `SAVINGS_DIR`. The warm script deletes it after every image; the other two keep it.
- **`react_cold_cache.py` quirks:** it still reads an unused `apis.json` at start-up (create one containing `{}` if you don't have it), and its batch run starts at row 475.
