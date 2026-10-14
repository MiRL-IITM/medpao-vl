import json
import os

import time
from typing import List

import pandas as pd
from tqdm import tqdm
from tools.generate_report import generate_the_grounded_report
from tools.get_concept import get_the_concept, CONCEPTS_FILE
from tools.categorize_concepts import categorize_the_concepts
from tools.generate_structured_report import generate_the_report
from tools.compare_concepts import Comparison
from tools.ontology_map_sapbert import SnomedMatcher

# ── Self-correction settings ──────────────────────────────────────────────────
SELF_CORRECTION_ITERATIONS = 3     # max passes of "regenerate lines for the missing concepts"
CONCEPT_EXTRACTION_RETRIES = 1     # extra attempts if re-extracting concepts from the updated report fails
STOP_IF_NO_PROGRESS        = True  # stop when a pass leaves the missing-concept set unchanged
                                   # (re-prompting with the same list would only append duplicate lines)

# ── Concept cache ─────────────────────────────────────────────────────────────
SAVINGS_DIR = "/home/user/agent/mcp_codes/agent_savings_cold_cache"
IMG_FOLDER='/home/user/agent/data/disk1/mimic_test_jpg'

with open('/home/user/agent/mcp_codes/apis.json') as f:
        apis = json.load(f)


my_api_key = apis.get('API_KEY')

# ontology_mapper = OntologyMapper(api_key=my_api_key)
ontology_mapper = SnomedMatcher(SAVING_DIR = SAVINGS_DIR)
Comp = Comparison()


def _parse_concepts_from_vision_string(vision_str: str) -> List[str]:
    """Extract bare concept names from a vision concepts string."""
    concepts = []
    for c in vision_str.split(','):
        c = c.strip()
        if c.endswith('(1.0)'):
            concepts.append(c[:-5].strip().lower())
        elif c.endswith('(-1.0)'):
            concepts.append('no '+c[:-6].strip().lower())
        elif c:
            concepts.append(c.lower())
    return list(set(concepts))


def _is_failed_report(report) -> bool:
    """generate_the_grounded_report signals failure by *returning* an "Error generating ..." string
    (or a {'prediction': 'Image not found'} dict) instead of raising, so check for it explicitly."""
    return (not isinstance(report, str)) or (not report.strip()) or report.startswith("Error generating")


def _usable_concepts(extracted) -> bool:
    """get_the_concept returns {concept: source sentence}; on a parse failure it returns text such as 'NULLA'."""
    return isinstance(extracted, dict) and len(extracted) > 0


def _extract_concepts(report: str):
    """Extract concepts from `report`, retrying on parse failures. Returns a dict, or None if all attempts fail."""
    for _ in range(1 + CONCEPT_EXTRACTION_RETRIES):
        extracted = get_the_concept(report, saving_dir=SAVINGS_DIR)
        if _usable_concepts(extracted):
            return extracted
    return None


def _restore_concepts_file(extracted_concepts: dict) -> None:
    """get_the_concept overwrites the concepts file even when parsing failed; later stages
    (ontology mapping, structured report) read that file, so put the last good extraction back."""
    with open(os.path.join(SAVINGS_DIR, CONCEPTS_FILE), 'w') as f:
        json.dump(extracted_concepts, f, indent=4)


def _find_missing_concepts(vision_concepts_str: str, extracted_concepts: dict) -> List[str]:
    """Vision concepts not covered by the report's concepts ('no <x>' for absent ones)."""
    missing, _ = Comp.compare_concepts(vision_concepts_str, extracted_concepts)
    # compare_concepts returns an int (not a list) from its early-exit branches, e.g. no vision concepts
    return list(missing) if isinstance(missing, list) else []


def run_pao_pipeline(image_path: str, vision_concepts_str: str, iterations=SELF_CORRECTION_ITERATIONS) -> dict:
    """Run the full deterministic pipeline for a single image and its vision concepts."""

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

    # print("Grounded report generated.", grounded_report)
    # print("*" * 89)
    # print("\n\n")

    # ------------------------------------------------------------------
    # 3. Extract concepts
    # ------------------------------------------------------------------
    start = time.perf_counter()

    extracted_concepts = get_the_concept(
        grounded_report,
        saving_dir=SAVINGS_DIR
    )

    timings["concept_extraction_time_sec"] = round(
        time.perf_counter() - start, 4
    )

    # print("Concepts extracted from grounded report.", extracted_concepts)
    # print("*" * 89)
    # print("\n\n")

    #-------------------------Self correction phase-------------------------------
    # One pass = (1) ask the VLM for extra lines covering ONLY the still-missing concepts,
    # (2) re-extract concepts from the WHOLE updated report, (3) re-compare with the vision concepts.
    # The loop ends when nothing is missing, a pass fails or adds no text, a pass fixes nothing,
    # or `iterations` passes have run. The comparison after each pass means the final
    # `missing_concepts` / `extracted_concepts` always describe the report that is returned.
    start = time.perf_counter()
    full_report = grounded_report
    missing_concepts: List[str] = []
    correction_log = []

    if _is_failed_report(grounded_report):
        print(f"Grounded report generation failed ({grounded_report!r}); skipping self-correction.")
    elif not _usable_concepts(extracted_concepts):
        print("Concept extraction from the grounded report failed; skipping self-correction.")
    else:
        missing_concepts = _find_missing_concepts(vision_concepts_str, extracted_concepts)
        print("initial missing concepts:", missing_concepts)

        for iteration in range(iterations):
            if not missing_concepts:
                break

            entry = {"iteration": iteration, "missing_before": list(missing_concepts)}
            correction_log.append(entry)

            # id=iteration makes the tool append missing_report{0..iteration} to the first report
            corrected_report = generate_the_grounded_report(
                img_path,
                vision_concepts_str,
                saving_dir=SAVINGS_DIR,
                missing_prompt=missing_concepts,
                id=iteration
            )
            if _is_failed_report(corrected_report):
                entry["status"] = "generation_failed"
                entry["error"] = str(corrected_report)
                print(f"Pass {iteration}: generation failed, keeping the previous report. {corrected_report}")
                break

            # the tool returns the cumulative report: what this pass added is the new tail
            added_lines = (corrected_report[len(full_report):] if corrected_report.startswith(full_report)
                           else corrected_report).strip()
            entry["added_lines"] = added_lines
            if not added_lines:
                entry["status"] = "no_new_lines"
                print(f"Pass {iteration}: the model produced no new lines.")
                break

            new_concepts = _extract_concepts(corrected_report)
            if new_concepts is None:
                # cannot verify the new lines: keep report, concepts file and concepts consistent (previous state)
                _restore_concepts_file(extracted_concepts)
                entry["status"] = "concept_extraction_failed"
                print(f"Pass {iteration}: could not extract concepts from the updated report; reverted.")
                break

            full_report, extracted_concepts = corrected_report, new_concepts
            new_missing = _find_missing_concepts(vision_concepts_str, extracted_concepts)
            entry["missing_after"] = list(new_missing)
            entry["status"] = "ok"
            print(f"Pass {iteration}: added {added_lines!r}\n  still missing: {new_missing}")

            no_progress = set(new_missing) == set(missing_concepts)
            missing_concepts = new_missing
            if STOP_IF_NO_PROGRESS and no_progress:
                entry["status"] = "no_progress"
                break

    timings["self_correction_time_sec"] = round(
        time.perf_counter() - start, 4
    )


    # ------------------------------------------------------------------
    # 4. Ontology mapping
    # ------------------------------------------------------------------
    start = time.perf_counter()

    mapped_ontologies = ontology_mapper.match(extracted_concepts, top_k=1, saving_dir=SAVINGS_DIR)
    # mapped_ontologies = ontology_mapper.map_concepts(
    #     extracted_concepts,
    #     saving_dir=SAVINGS_DIR
    # )

    timings["ontology_mapping_time_sec"] = round(
        time.perf_counter() - start, 4
    )


    # ------------------------------------------------------------------
    # 5. Filter ontologies
    # ------------------------------------------------------------------
    # start = time.perf_counter()

    # filtered_ontologies = filter_the_ontology(
    #     mapped_ontologies,
    #     saving_dir=SAVINGS_DIR
    # )

    # timings["ontology_filtering_time_sec"] = round(
    #     time.perf_counter() - start, 4
    # )
    filtered_ontologies = mapped_ontologies


    # ------------------------------------------------------------------
    # 6. Categorize concepts
    # ------------------------------------------------------------------
    start = time.perf_counter()

    categorized_concepts = categorize_the_concepts(
        filtered_ontologies,
        saving_dir=SAVINGS_DIR
    )

    timings["concept_categorization_time_sec"] = round(
        time.perf_counter() - start, 4
    )

    # print("Concept categorization completed.", categorized_concepts)
    # print("*" * 89)
    # print("\n\n")

    # ------------------------------------------------------------------
    # 7. Generate structured report
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
        "full report": full_report,
        "extracted_concepts": extracted_concepts,
        "missing_concepts": missing_concepts,
        "mapped_ontologies": mapped_ontologies,
        # "filtered_ontologies": filtered_ontologies,
        "categorized_concepts": categorized_concepts,
        "structured_report": structured_report,
        "iterations": len(correction_log),   # self-correction passes attempted
        "correction_log": correction_log,
        "timings": timings
    }


if __name__=='__main__':
    os.makedirs(SAVINGS_DIR, exist_ok=True)
    full_df = pd.read_csv('/home/user/agent/stratified_ablation_sample.csv')

    first_half = full_df.iloc[475:len(full_df)//2]

    entire_json = []
    for idx, row in tqdm(first_half.iterrows(), total=len(first_half)):
        image_path = row['image_path']
        vision_concepts = row['standardized_concepts']
        results = run_pao_pipeline(image_path, vision_concepts)
        if results is None:
            continue
        entire_json.append(results)
        if len(entire_json)%20==0:
            os.makedirs(SAVINGS_DIR, exist_ok=True)
            with open(os.path.join(SAVINGS_DIR, f'entire_results_{idx+475}.json'), 'w') as json_file:
                json.dump(entire_json, json_file, indent=4)

    with open('entire_results_cold_cache.json', 'w') as f:
        json.dump(entire_json, f, indent=4)

    