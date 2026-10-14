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
| `tools/generate_report.py` | Asks the VLM (image + concepts) for the findings; also writes the extra lines for missing concepts. |
| `tools/get_concept.py` | Asks an LLM to pull `{concept: source sentence}` out of a report. |
| `tools/compare_concepts.py` | Compares vision concepts with report concepts using SapBERT similarity; returns the missing ones. |
| `tools/check_cache.py` | Fuzzy-matches concepts against the cached vocabulary; splits them into hits and new concepts. |
| `tools/ontology_map_sapbert.py` | Maps concepts to SNOMED CT with SapBERT embeddings + a FAISS index (`SnomedMatcher`). |
| `tools/categorize_concepts.py` | Asks an LLM to assign each concept to P/A/B/C/D/E/F using its SNOMED ancestors. |
| `tools/generate_structured_report.py` | Asks an LLM to write the final per-category report (JSON). |

## Setup

1. **Python packages:** `pandas tqdm pillow requests numpy huggingface_hub transformers torch scikit-learn faiss-cpu llama-cpp-python psutil rapidfuzz` (`rapidfuzz` is only needed by the warm script).
2. **Three TGI-compatible servers** must be running (change the URLs at the top of each tool if yours differ):

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

## Output

One entry per image: the vision concepts, the grounded report, extracted concepts, SNOMED matches, categorized concepts, the structured report and per-step timings. The agent scripts also include `missing_concepts` (left after correction), `iterations` (correction passes) and `correction_log` (what each pass added). The warm script adds `cache_hit` / `cache_miss`.

`deterministic.py` and `react_warm_cache.py` skip images whose report or concept extraction fails (they do not appear in the output). `react_cold_cache.py` does not skip them yet; a failed image still produces an entry.

## Notes

- **Reusing the cache:** the warm script reads `<SAVINGS_DIR>_cache/cached_vocab.json`. Cold and warm have different `SAVINGS_DIR` values, so their caches are separate. Copy the file over (or use the same `SAVINGS_DIR` base) to start warm with a filled cache.
- **Scratch files:** intermediate JSON files go in `SAVINGS_DIR`. The warm script deletes it after every image; the other two keep it.
- **`react_cold_cache.py` quirks:** it still reads an unused `apis.json` at start-up (create one containing `{}` if you don't have it), and its batch run starts at row 475.
