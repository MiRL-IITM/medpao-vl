# MIMIC-CXR Concept Extraction, Standardization & Filtering

A four-stage pipeline that turns raw MIMIC-CXR radiology reports into a clean, SNOMED CT-standardized concept vocabulary and a per-report label matrix.

```
MIMIC-CXR reports (.txt)
   │  1. mimic_simple_filtering.py     keep reports with HISTORY + FINDINGS + IMPRESSION
   ▼
filtered_mimic.csv
   │  2. concept_extraction.py         LLM (few-shot, served via TGI) extracts concepts
   ▼
concepts_df.csv
   │  3. concept_standardization.py    normalize → SNOMED CT (SapBERT) → group
   ▼
standardization_out/
   │  4. concept_filtering.py          drop rare groups, build label matrix
   ▼
standardization_out/filtered/   (label_matrix.npy, ...)
```

---

## Requirements

- Python 3.9+ with `pandas`, `numpy`, `tqdm`, `huggingface_hub`, `nltk`
- Whatever `snomedct_mapper.py` needs for SapBERT matching (a GPU is recommended for the SNOMED matching step)
- Docker + a GPU (or Gaudi) node to host the extraction model with Text Generation Inference (TGI)
- MIMIC-CXR access (PhysioNet credentialed)
- SNOMED CT International Edition release files

```bash
pip install pandas numpy tqdm huggingface_hub nltk
```

`concept_standardization.py` downloads the NLTK `wordnet` / `omw-1.4` data automatically on first run. If the machine is offline, download them beforehand, or pass `--no-lemmatize`.

---

## Step 1 — Keep only reports with HISTORY, FINDINGS and IMPRESSION

`mimic_simple_filtering.py` scans the study-level report files and keeps only those that contain all three sections with non-empty text.

Expected layout (standard MIMIC-CXR / MIMIC-CXR-JPG):

```
<root>/files/p10/p10000032/s50414267.txt
<root>/files/p10/p10000032/s50414267/<image>.jpg|dcm
```

```bash
python mimic_simple_filtering.py \
    --root /data/mimic-cxr \
    --output filtered_mimic.csv
```

| Option | Default | Description |
|---|---|---|
| `--root` | required | MIMIC-CXR root folder |
| `--output` | `mimic_reports.csv` | Output CSV |
| `--absolute-paths` | off | Store absolute paths instead of paths relative to `--root` |
| `--keep-empty` | off | Keep reports where a required header exists but its text is empty |
| `--workers` | 8 | Reader threads |
| `--limit` | none | Only scan the first N reports (debug) |

**Output:** `filtered_mimic.csv` with columns `path, findings, impression, history`. A summary of how many reports were dropped (and why) is logged.

---

## Step 2 — Concept extraction

Concepts are extracted from the `findings` text with a fine-tuned LLM using a few-shot prompt, served by a TGI server.

### 2a. Download the model

Model: <https://huggingface.co/shrishSVaidya/medllama_finetuned>

```bash
huggingface-cli download shrishSVaidya/medllama_finetuned --local-dir /mnt/hf_model/medllama_finetuned
```

### 2b. Start the TGI server

The extraction script talks to `http://localhost:8081` by default, so map the container port to 8081. Adjust the image tag, GPU flags and token limits to your hardware.

```bash
volume=/mnt/hf_model                         # directory where model weights are stored
model=shrishSVaidya/medllama_finetuned       # or /data/medllama_finetuned for a local copy

docker run --name medllama_tgi \
    --gpus all --shm-size 1g \
    -p 8081:80 \
    -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id $model \
    --max-input-tokens 3072 \
    --max-total-tokens 4096
```

<details>
<summary>Example for Intel Gaudi2 accelerators</summary>

```bash
volume=/mnt/hf_model
model=shrishSVaidya/medllama_finetuned

docker run --name medllama_tgi -p 8081:80 \
    --runtime=habana --cap-add=sys_nice --ipc=host \
    -v $volume:/data \
    -e MAX_TOTAL_TOKENS=4056 \
    -e BATCH_BUCKET_SIZE=256 \
    -e PREFILL_BATCH_BUCKET_SIZE=4 \
    -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
    -e RUST_LOG=debug \
    ghcr.io/huggingface/text-generation-inference:3.2.3-gaudi \
    --model-id $model \
    --max-input-tokens 6048 --max-total-tokens 8056 \
    --max-batch-prefill-tokens 6096 --max-batch-size 2 \
    --max-waiting-tokens 7 --waiting-served-ratio 1.2
```

Add `--sharded true --num-shard N` if the model is split across several cards.
</details>

Check that the server is up before continuing:

```bash
curl http://localhost:8081/health
```

### 2c. Run the extraction

```bash
python concept_extraction.py \
    --annotation filtered_mimic.csv \
    --base-url http://localhost:8081 \
    --batch-size 4 \
    --limit 4000 \
    --prompt-mode concepts_only \
    --output concepts_df.csv
```

| Option | Default | Description |
|---|---|---|
| `--annotation` | required | CSV from step 1 (needs `findings` and `path` columns) |
| `--limit` | 4000 | Max reports to process (`-1` = all) |
| `--output` | `save1.csv` | Final output CSV |
| `--base-url` | `http://localhost:8081` | TGI server URL |
| `--batch-size` | 4 | Concurrent requests per batch (tune to the TGI max batch size) |
| `--max-new-tokens` | 200 | Generation length cap |
| `--prompt-mode` | `concepts_only` | `concepts_only`: concepts only. `with_sentences`: concept → source sentence |
| `--num-shots` | 3 | Number of few-shot examples (1–5) |
| `--checkpoint-every` | 5000 | Save a checkpoint every N completed records (0 disables) |
| `--checkpoint-prefix` | `model_history_preds` | Checkpoints are saved as `<prefix>_<N>.csv` |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |

**Output:** `concepts_df.csv` with columns:

| Column | Content |
|---|---|
| `mimic_findings` | Findings text of the report |
| `concepts` | Comma-joined concept list, e.g. `no pleural effusion,cardiomegaly,normal heart contours` |
| `explainability` | Comma-joined source sentences (only filled in `with_sentences` mode) |
| `path` | Report path from the input CSV |

Reports whose request failed or whose output could not be parsed as JSON are skipped. The failure count is logged at the end.

---

## Step 3 — Concept standardization

Maps every extracted concept to a SNOMED CT concept and groups concepts that share the same SNOMED target.

### 3a. Download SNOMED CT

Download the International Edition release from <https://www.nlm.nih.gov/healthit/snomedct/international.html> and unzip it. You need the English description snapshot and the relationship snapshot:

```
SnomedCT_.../Snapshot/Terminology/sct2_Description_Snapshot-en_INT_*.txt
SnomedCT_.../Snapshot/Terminology/sct2_Relationship_Snapshot_INT_*.txt
```

### 3b. Run

```bash
python concept_standardization.py \
    --concepts-csv concepts_df.csv \
    --out-dir standardization_out \
    --desc-file SnomedCT_.../Snapshot/Terminology/sct2_Description_Snapshot-en_INT_*.txt \
    --rel-file  SnomedCT_.../Snapshot/Terminology/sct2_Relationship_Snapshot_INT_*.txt \
    --snomed-index-dir snomed_index
```

`--snomed-index-dir` is where the SNOMED embedding index is cached, so later runs can reuse it. The SNOMED mapper options (`--desc-file`, `--rel-file`, `--snomed-index-dir`, `--query-batch-size`, ...) are defined in `snomedct_mapper.py`. Run with `--help` for the full list.

### What it does

1. **Parse** the `concepts` column and count raw concepts.
2. **Preprocess**: lowercase, strip punctuation (keeping `/` and `-`), detect and remove negation prefixes (`no evidence of`, `negative for`, `absence of`, `free of`, `without`, `no`, `not`) and record polarity (+1 present, −1 negated), then noun-lemmatize (plural → singular; verbs are left alone so "enlarged" stays "enlarged").
3. **Map to SNOMED CT** with SapBERT. The top-k candidates are stored and top-1 is used.
4. **Foreign-device rules**: keyword rules override the SNOMED group for device-related concepts (first matching rule wins):
   - `foreign1`: PICC
   - `foreign2`: pacemaker / defibrillator / ICD
   - `foreign3`: tubes
   - `foreign4`: surgical clips/hardware (sternotomy, sternal, clips, stents, valves, grafts, hardware)
   - `foreign5`: implants (arthroplasty, prosthesis, replacement, implant)
5. **Build the final mapping** (raw concept → group) and the inverted dictionary (group → raw concepts).
6. **Count group frequencies** and write per-report standardized concepts.

### Options

| Option | Default | Description |
|---|---|---|
| `--concepts-csv` | required | CSV from step 2 |
| `--concepts-column` | `concepts` | Column holding the concept lists |
| `--out-dir` | required | Output directory |
| `--no-lemmatize` | off | Skip lemmatization |
| `--top-k` | 5 | SNOMED candidates stored per concept (top-1 is used) |
| `--min-similarity` | 0.0 | Top-1 matches below this cosine similarity are kept as their own `unmapped::<text>` group instead of being merged into a SNOMED concept |
| `--reuse-matches` | off | Skip SNOMED matching if `snomed_candidates.json` already exists in `--out-dir` (saves the GPU step when re-running with different settings) |

### Output files (all inside `--out-dir`)

| File | Description |
|---|---|
| `unique_concepts_raw.txt` | Every distinct raw concept string, one per line |
| `concept_frequencies_raw.csv` | Raw concept → number of occurrences across all reports (sorted descending) |
| `preprocessed_concepts.csv` | `raw_concept`, normalized text (lowercased, cleaned, negation removed, lemmatized), `polarity`, `count` |
| `unique_concepts_normalized.txt` | Distinct normalized texts; exactly what is sent to SNOMED matching |
| `snomed_candidates.json` | Normalized text → top-k SNOMED candidates (id, matched surface, preferred description, similarity). Reusable via `--reuse-matches` |
| `snomed_mapping_top1.csv` | Top-1 SNOMED match per normalized text with similarity and a `mapped` flag. Sorted by ascending similarity so doubtful matches are on top for review |
| `foreign_device_assignments.csv` | Concepts re-routed to `foreign1`–`foreign5`, with the SNOMED group they would otherwise have joined |
| `final_mapping.json` / `final_mapping.csv` | Raw concept → `[group_name, group_id, polarity]` for **all** groups (unfiltered) |
| `inverted_dict.json` | `group_id` → `{name, positive: [raw concepts], negative: [raw concepts]}` for all groups |
| `group_frequencies_all.csv` | Per group: positive / negative / total frequency, number of raw concepts, `has_negative`. Input for step 4 |
| `concepts_df_standardized.csv` | Input CSV plus `standardized_concepts`: JSON list of `[group_id, group_name, polarity]` per report (all groups, unfiltered) |
| `summary_stats.json` | Counts at each stage of the pipeline |
| `run_config.json` | All CLI arguments used for this run (reproducibility) |

---

## Step 4 — Concept filtering

Removes concept groups that are too rare and builds the final label matrix. Run it on the output directory of step 3.

```bash
python concept_filtering.py \
    --standardization-dir standardization_out \
    --min-freq 100
```

| Option | Default | Description |
|---|---|---|
| `--standardization-dir` | required | The `--out-dir` used for step 3 |
| `--out-dir` | `<standardization-dir>/filtered` | Where to write results |
| `--min-freq` | 100 | Keep groups whose total frequency (positive + negative) is ≥ this value |

### Output files (all inside `--out-dir`, default `<standardization-dir>/filtered`)

| File | Description |
|---|---|
| `groups_accepted.csv` / `groups_rejected.csv` | Groups with total frequency ≥ / < `--min-freq`. Columns: `group_id, group_name, positive, negative, total, n_raw_concepts, has_negative` |
| `inverted_dict_filtered.json` | `group_id` → `{name, positive, negative}`, accepted groups only |
| `final_mapping_filtered.json` | Raw concept → `[group_name, group_id, polarity]`, accepted groups only |
| `concepts_df_filtered.csv` | The report CSV with `standardized_concepts` restricted to accepted groups |
| `label_matrix.npy` | `int8` matrix of shape `(n_reports, n_accepted_groups)` |
| `label_matrix_columns.json` | Column order: `columns` (group ids) and `names` (group names) |
| `filtering_summary.json` | Group / concept / report counts before and after filtering, plus the fraction of all concept mentions covered by the kept groups |
| `filtering_config.json` | CLI arguments used for this run |

### Label matrix values

| Value | Meaning |
|---|---|
| `1` | Concept present |
| `-1` | Concept explicitly absent (negated) |
| `0` | Not mentioned, or contradictory within the same report (both present and absent) |

Rows follow the row order of `concepts_df_filtered.csv`. Columns follow `label_matrix_columns.json`.

```python
import json, numpy as np

labels = np.load("standardization_out/filtered/label_matrix.npy")
cols = json.load(open("standardization_out/filtered/label_matrix_columns.json"))
print(labels.shape, cols["names"][:5])
```

---

## Quick start (all steps)

```bash
# 1. filter reports
python mimic_simple_filtering.py --root /data/mimic-cxr --output filtered_mimic.csv

# 2. start TGI (see step 2b), then extract concepts
python concept_extraction.py --annotation filtered_mimic.csv --base-url http://localhost:8081 \
    --batch-size 4 --limit 4000 --prompt-mode concepts_only --output concepts_df.csv

# 3. standardize to SNOMED CT
python concept_standardization.py --concepts-csv concepts_df.csv --out-dir standardization_out \
    --desc-file <description snapshot> --rel-file <relationship snapshot> --snomed-index-dir snomed_index

# 4. filter rare concept groups
python concept_filtering.py --standardization-dir standardization_out --min-freq 100
```

## Tips


- **Extend the rules.** Negation prefixes (`NEGATION_PREFIXES`) and device rules (`FOREIGN_DEVICE_RULES`) are plain lists at the top of the relevant sections in `concept_standardization.py`.