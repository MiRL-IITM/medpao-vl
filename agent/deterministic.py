import json
import os
import time

import pandas as pd
from tqdm import tqdm
from tools.generate_report import generate_the_grounded_report
from tools.get_concept import get_the_concept
from tools.categorize_concepts import categorize_the_concepts
from tools.generate_structured_report import generate_the_report
from tools.ontology_map_sapbert import SnomedMatcher

# ───────────────────────────── CONFIG (edit here) ─────────────────────────────
# --- Folders ---
SAVINGS_DIR = "/home/user/agent/mcp_codes/deterministics_savings"   # intermediate files + checkpoints
IMG_FOLDER  = '/home/user/agent/data/disk1/mimic_test_jpg'

# --- Pipeline ---
CONCEPT_EXTRACTION_RETRIES = 1     # extra attempts if concept extraction returns unparseable text
ONTOLOGY_TOP_K             = 1     # SNOMED candidates per concept

# --- Batch run (python deterministic.py) ---
INPUT_CSV          = '/home/user/agent/stratified_ablation_sample.csv'
IMAGE_COL          = 'image_path'
CONCEPTS_COL       = 'standardized_concepts'
ROW_START          = 0             # first row to process
ROW_END            = None          # None -> first half of the CSV
CHECKPOINT_EVERY   = 20            # write partial results into SAVINGS_DIR every N finished images
FINAL_RESULTS_FILE = 'entire_results_deterministic.json'
# ──────────────────────────────────────────────────────────────────────────────

ontology_mapper = SnomedMatcher(SAVING_DIR = SAVINGS_DIR)


def _is_failed_report(report) -> bool:
    """generate_the_grounded_report signals failure by *returning* an "Error generating ..." string
    (or a {'prediction': 'Image not found'} dict) instead of raising, so check for it explicitly."""
    return (not isinstance(report, str)) or (not report.strip()) or report.startswith("Error generating")


def _usable_concepts(extracted) -> bool:
    """get_the_concept returns {concept: source sentence}; on a parse failure it returns text such as 'NULLA'."""
    return isinstance(extracted, dict) and len(extracted) > 0


def run_deterministic_pipeline(image_path: str, vision_concepts_str: str) -> dict:
    """Run the full deterministic pipeline for a single image and its vision concepts.
    Returns None when the image has to be skipped."""

    total_start_time = time.perf_counter()

    timings = {}

    # 1. Parse vision concepts
    img_path = os.path.join(IMG_FOLDER, image_path.split('/')[-1])

    print(img_path)

    # ------------------------------------------------------------------
    # 2. Generate grounded report
    # ------------------------------------------------------------------
    start = time.perf_counter()

    grounded_report = generate_the_grounded_report(
        img_path,
        vision_concepts_str,
        saving_dir=SAVINGS_DIR
    )

    timings["grounded_report_time_sec"] = round(
        time.perf_counter() - start, 4
    )

    if _is_failed_report(grounded_report):
        print(f"Grounded report generation failed ({grounded_report!r}). SKIPPING...........................")
        return None

    # ------------------------------------------------------------------
    # 3. Extract concepts
    # ------------------------------------------------------------------
    start = time.perf_counter()

    extracted_concepts = None
    for _ in range(1 + CONCEPT_EXTRACTION_RETRIES):
        extracted_concepts = get_the_concept(
            grounded_report,
            saving_dir=SAVINGS_DIR
        )
        if _usable_concepts(extracted_concepts):
            break

    timings["concept_extraction_time_sec"] = round(
        time.perf_counter() - start, 4
    )

    if not _usable_concepts(extracted_concepts):
        print("Concept extraction failed. SKIPPING...........................")
        return None

    # ------------------------------------------------------------------
    # 4. Ontology mapping
    # ------------------------------------------------------------------
    start = time.perf_counter()

    mapped_ontologies = ontology_mapper.match(extracted_concepts, top_k=ONTOLOGY_TOP_K, saving_dir=SAVINGS_DIR)

    timings["ontology_mapping_time_sec"] = round(
        time.perf_counter() - start, 4
    )

    # ------------------------------------------------------------------
    # 5. Categorize concepts
    # ------------------------------------------------------------------
    start = time.perf_counter()

    categorized_concepts = categorize_the_concepts(
        mapped_ontologies,
        saving_dir=SAVINGS_DIR
    )

    timings["concept_categorization_time_sec"] = round(
        time.perf_counter() - start, 4
    )

    # ------------------------------------------------------------------
    # 6. Generate structured report
    # ------------------------------------------------------------------
    start = time.perf_counter()

    structured_report = generate_the_report(
        categorized_concepts,
        saving_dir=SAVINGS_DIR
    )

    timings["structured_report_generation_time_sec"] = round(
        time.perf_counter() - start, 4
    )

    print("Structured report generated.", structured_report)
    print("*" * 89)
    print("\n\n")

    # ------------------------------------------------------------------
    # Final total timing
    # ------------------------------------------------------------------
    total_time = round(
        time.perf_counter() - total_start_time,
        4
    )

    timings["total_pipeline_time_sec"] = total_time

    return {
        "image_path": image_path,
        "Vision_concepts": vision_concepts_str,
        "grounded_report": grounded_report,
        "extracted_concepts": extracted_concepts,
        "mapped_ontologies": mapped_ontologies,
        "categorized_concepts": categorized_concepts,
        "structured_report": structured_report,
        "timings": timings
    }


if __name__=='__main__':
    full_df = pd.read_csv(INPUT_CSV)

    rows = full_df.iloc[ROW_START:ROW_END if ROW_END is not None else len(full_df)//2]

    entire_json = []
    for idx, row in tqdm(rows.iterrows(), total=len(rows)):
        image_path = row[IMAGE_COL]
        vision_concepts = row[CONCEPTS_COL]
        results = run_deterministic_pipeline(image_path, vision_concepts)
        if results is None:
            continue
        entire_json.append(results)
        if len(entire_json) % CHECKPOINT_EVERY == 0:
            os.makedirs(SAVINGS_DIR, exist_ok=True)
            with open(os.path.join(SAVINGS_DIR, f'entire_results_{idx}.json'), 'w') as json_file:
                json.dump(entire_json, json_file, indent=4)

    with open(FINAL_RESULTS_FILE, 'w') as f:
        json.dump(entire_json, f, indent=4)
