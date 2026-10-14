#!/usr/bin/env python3
"""Extract medical concepts (and optionally their source sentences) from MIMIC-CXR
reports using a few-shot prompt against a TGI server.

"""

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from huggingface_hub import AsyncInferenceClient
from tqdm import tqdm

logger = logging.getLogger("extract_concepts")

# --------------------------------------------------------------------------- #
# Few-shot examples: (report, {concept: source sentence})
# --------------------------------------------------------------------------- #
FEW_SHOT_EXAMPLES: List[Tuple[str, Dict[str, str]]] = [
    (
        "No focal consolidation, pleural effusion, or pneumothorax is seen. "
        "Heart and mediastinal contours are within normal limits.  "
        "There is no evidence for pulmonary edema.",
        {
            "no focal consolidation": "No focal consolidation, pleural effusion, or pneumothorax is seen.",
            "no pleural effusion": "No focal consolidation, pleural effusion, or pneumothorax is seen.",
            "no pneumothorax": "No focal consolidation, pleural effusion, or pneumothorax is seen.",
            "normal heart contours": "Heart and mediastinal contours are within normal limits.",
            "normal mediastinal contours": "Heart and mediastinal contours are within normal limits.",
            "no pulmonary edema": " There is no evidence for pulmonary edema.",
        },
    ),
    (
        "PA and lateral views of the chest provided.   Lung volumes are somewhat low  "
        "though allowing for this, there is no focal consolidation, effusion, or  "
        "pneumothorax. The cardiomediastinal silhouette is notable for an unfolded  "
        "thoracic aorta.  Imaged osseous structures are intact.  No free air below the  "
        "right hemidiaphragm is seen.",
        {
            "PA and lateral views": "PA and lateral views of the chest provided.",
            "low lung volumes": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
            "likely no focal consolidation": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
            "likely no effusion": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
            "likely no pneumothorax": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
            "noteable cardiomediastinal silhouette": "The cardiomediastinal silhouette is notable for an unfolded  thoracic aorta. ",
            "intact osseous structures": "Imaged osseous structures are intact.",
            "no free air below right hemidiaphragm": " No free air below the  right hemidiaphragm is seen.",
        },
    ),
    (
        "Previously visualized left lower lobe opacity has improved and is suggestive  "
        "of resolving pneumonia.  No new consolidations are identified.  There is no  "
        "pleural effusion or pneumothorax.  Cardiac and mediastinal silhouettes are  "
        "normal.  No acute fractures are identified.",
        {
            "improving left lower lobe opacity": "Previously visualized left lower lobe opacity has improved and is suggestive  of resolving pneumonia.",
            "likely due to resolving pneumonia": "Previously visualized left lower lobe opacity has improved and is suggestive  of resolving pneumonia.",
            "no new consolidations": "No new consolidations are identified.",
            "no pleural effusion": "There is no  pleural effusion or pneumothorax.",
            "normal cardiac silhouette": "Cardiac and mediastinal silhouettes are  normal.",
            "normal mediastinal silhouette": "Cardiac and mediastinal silhouettes are  normal.",
            "no acute fractures": "No acute fractures are identified.",
        },
    ),
    (
        "PA and lateral views of the chest provided demonstrate bilateral  lower lung "
        "patchy opacities most confluent in the left lower lung which is  compatible "
        "with pneumonia.  No large effusion or pneumothorax.   Cardiomediastinal "
        "silhouette is normal.  Bony structures are intact.",
        {
            "PA and lateral views": "PA and lateral views of the chest provided demonstrate bilateral  lower lung patchy opacities most confluent in the left lower lung which is  compatible with pneumonia.",
            "bilateral lower lung patchy opacities with left lower lung involvement": "PA and lateral views of the chest provided demonstrate bilateral  lower lung patchy opacities most confluent in the left lower lung which is  compatible with pneumonia.",
            "likely due to pneumonia": "PA and lateral views of the chest provided demonstrate bilateral  lower lung patchy opacities most confluent in the left lower lung which is  compatible with pneumonia.",
            "no large effusion": "No large effusion or pneumothorax.",
            "no pneumothorax": "No large effusion or pneumothorax.",
            "normal cardiomediastinal silhouette": "Cardiomediastinal silhouette is normal.",
            "normal bony structures": "Bony structures are intact.",
        },
    ),
    (
        "PA and lateral views of the chest provided.   The heart is markedly enlarged.  "
        "There is a tiny left pleural effusion. Mild central congestion noted. No  "
        "pneumothorax. Mediastinal contour is normal. Bony structures are intact.",
        {
            "PA and lateral views": "PA and lateral views of the chest provided.",
            "markedly enlarged heart": "The heart is markedly enlarged.",
            "tiny left pleural effusion": "There is a tiny left pleural effusion.",
            "mild central congestion": "Mild central congestion noted.",
            "no pneumothorax": "No  pneumothorax.",
            "normal mediastinal contour": "Mediastinal contour is normal.",
            "normal bony structures": "Bony structures are intact.",
        },
    ),
]

PROMPT_MODES = ("with_sentences", "concepts_only")

INSTRUCTIONS = {
    "with_sentences": (
        "Assume you are an expert radiology practitioner.you have to extract the "
        "medical concepts from the given report and also indicate from which "
        "sentence the concept was picked.\n"
    ),
    "concepts_only": (
        "Assume you are an expert radiology practitioner.you have to extract the "
        "medical concepts from the given report, according to given examples.\n"
    ),
}

QUERY_TEMPLATE = "Extracted concepts and their corresponding sentence from {report} are "


# --------------------------------------------------------------------------- #
# Prompt construction & output parsing
# --------------------------------------------------------------------------- #
def build_prompt(report: str, mode: str = "with_sentences", num_shots: int = 3) -> str:
    """Build a few-shot prompt.

    mode="with_sentences": examples map concept -> source sentence.
    mode="concepts_only":  examples map concept -> "" (concepts only).
    """
    if mode not in PROMPT_MODES:
        raise ValueError(f"mode must be one of {PROMPT_MODES}, got {mode!r}")

    parts = [INSTRUCTIONS[mode]]
    for ex_report, ex_concepts in FEW_SHOT_EXAMPLES[:num_shots]:
        if mode == "concepts_only":
            ex_concepts = {k: "" for k in ex_concepts}
        parts.append(f"{QUERY_TEMPLATE.format(report=ex_report)}{ex_concepts} ##\n")
    parts.append(QUERY_TEMPLATE.format(report=report))
    return "".join(parts)


def parse_llm_output(text: str) -> Optional[dict]:
    """Extract the first {...} JSON object from the model output.

    Returns the parsed dict, or None if no valid JSON object could be found.
    """
    text = text.replace("'", '"')
    text = text.split("##")[0]
    text = re.sub(r"\n\s*", " ", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.debug("No JSON object found in output: %s", text)
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- #
# Async inference
# --------------------------------------------------------------------------- #
async def generate_single(
    client: AsyncInferenceClient, prompt: str, idx: int, max_new_tokens: int
) -> Tuple[int, Optional[str]]:
    """Run one text-generation request; returns (idx, response or None on error)."""
    try:
        response = await client.text_generation(prompt, max_new_tokens=max_new_tokens)
        return idx, response
    except Exception as e:  # noqa: BLE001
        logger.error("[idx=%d] Request failed: %s", idx, e)
        return idx, None


async def process_batch(
    client: AsyncInferenceClient,
    prompts_with_idx: List[Tuple[int, str]],
    max_new_tokens: int,
) -> List[Tuple[int, Optional[str]]]:
    """Fire all requests in a batch concurrently and wait for all of them."""
    tasks = [
        generate_single(client, prompt, idx, max_new_tokens)
        for idx, prompt in prompts_with_idx
    ]
    return await asyncio.gather(*tasks)


def build_dataframe(findings, concepts, explainability, paths) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mimic_findings": findings,
            "concepts": concepts,
            "explainability": explainability,
            "path": paths,
        }
    )


async def run_inference(
    reports: List[str], paths: List[str], args: argparse.Namespace
) -> pd.DataFrame:
    client = AsyncInferenceClient(base_url=args.base_url)

    findings, concepts, explainability, final_paths = [], [], [], []
    failures = 0
    next_checkpoint = args.checkpoint_every
    total = len(reports)

    for batch_start in tqdm(range(0, total, args.batch_size), desc="Batches"):
        batch_end = batch_start + args.batch_size
        batch_reports = reports[batch_start:batch_end]
        batch_paths = paths[batch_start:batch_end]

        prompts_with_idx = [
            (i, build_prompt(r.replace("\n", " "), args.prompt_mode, args.num_shots))
            for i, r in enumerate(batch_reports)
        ]
        results = await process_batch(client, prompts_with_idx, args.max_new_tokens)

        for i, response in sorted(results, key=lambda x: x[0]):
            if response is None:
                failures += 1
                continue

            parsed = parse_llm_output(response)
            if parsed is None:
                logger.warning("Parse failure at global index %d", batch_start + i)
                failures += 1
                continue

            findings.append(batch_reports[i])
            concepts.append(",".join(parsed.keys()))
            explainability.append(",".join(str(v) for v in parsed.values()))
            final_paths.append(batch_paths[i])

        # Checkpoint each time we cross another `checkpoint_every` records
        if args.checkpoint_every > 0 and len(final_paths) >= next_checkpoint:
            ckpt = Path(f"{args.checkpoint_prefix}_{len(final_paths)}.csv")
            build_dataframe(findings, concepts, explainability, final_paths).to_csv(
                ckpt, index=False
            )
            logger.info("Checkpoint saved: %s", ckpt)
            next_checkpoint = (len(final_paths) // args.checkpoint_every + 1) * args.checkpoint_every

    logger.info("Done. Parse/request failures: %d", failures)
    return build_dataframe(findings, concepts, explainability, final_paths)


# --------------------------------------------------------------------------- #
# Data loading & CLI
# --------------------------------------------------------------------------- #
def load_data(annotation_path: str, limit: Optional[int]):
    df = pd.read_csv(annotation_path)
    reports, paths = df['findings'].tolist(), df['path'].tolist()
    return reports, paths


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract medical concepts from MIMIC-CXR reports via a TGI server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("input / output")
    io.add_argument("--annotation", required=True, help="Path to csv file.")
    io.add_argument("--limit", type=int, default=4000,
                    help="Max number of reports to process (use -1 for all).")
    io.add_argument("--output", default="save1.csv", help="Final output CSV path.")
    io.add_argument("--checkpoint-every", type=int, default=5000,
                    help="Save a checkpoint every N completed records (0 disables).")
    io.add_argument("--checkpoint-prefix", default="model_history_preds",
                    help="Checkpoint filename prefix -> <prefix>_<N>.csv")

    srv = p.add_argument_group("inference server")
    srv.add_argument("--base-url", default="http://localhost:8081", help="TGI server URL.")
    srv.add_argument("--batch-size", type=int, default=4,
                     help="Concurrent requests per batch (tune to TGI max_batch_size).")
    srv.add_argument("--max-new-tokens", type=int, default=200)

    pr = p.add_argument_group("prompting")
    pr.add_argument("--prompt-mode", choices=PROMPT_MODES, default="concepts_only",
                    help="with_sentences: concept->sentence; concepts_only: concepts only.")
    pr.add_argument("--num-shots", type=int, default=3,
                    help=f"Number of few-shot examples (1-{len(FEW_SHOT_EXAMPLES)}).")

    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = p.parse_args(argv)

    if args.limit is not None and args.limit < 0:
        args.limit = None
    if not 1 <= args.num_shots <= len(FEW_SHOT_EXAMPLES):
        p.error(f"--num-shots must be between 1 and {len(FEW_SHOT_EXAMPLES)}")
    if args.batch_size < 1:
        p.error("--batch-size must be >= 1")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    reports, paths = load_data(args.annotation, args.limit)
    logger.info("Loaded %d reports", len(reports))

    df = asyncio.run(run_inference(reports, paths, args))
    df.to_csv(args.output, index=False)
    logger.info("Total records processed: %d -> %s", len(df), args.output)


if __name__ == "__main__":
    main()