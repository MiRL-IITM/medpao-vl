#!/usr/bin/env python3
"""Build a CSV of MIMIC-CXR reports that contain explicit HISTORY, FINDINGS and
IMPRESSION sections.

Expected layout (standard MIMIC-CXR / MIMIC-CXR-JPG):
    <root>/files/p10/p10000032/s50414267.txt
    <root>/files/p10/p10000032/s50414267/<image>.jpg|dcm

Example:
    python build_mimic_report_csv.py --root /data/mimic-cxr --output reports.csv
"""

import argparse
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("build_mimic_report_csv")

# A section header = an ALL-CAPS label (letters, spaces, / & -) at the start of a
# line, followed by a colon, e.g. "FINDINGS:", "CLINICAL HISTORY:", "REASON FOR EXAMINATION:".
HEADER_RE = re.compile(r"^[ \t]*([A-Z][A-Z /&\-]{2,60}?)[ \t]*:", re.MULTILINE)

REQUIRED = ("HISTORY", "FINDINGS", "IMPRESSION")


def split_sections(text: str) -> Dict[str, str]:
    """Split a report into {HEADER: text}. Repeated headers are concatenated."""
    matches = list(HEADER_RE.finditer(text))
    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        header = re.sub(r"\s+", " ", m.group(1)).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = re.sub(r"\s+", " ", text[m.end():end]).strip()
        sections[header] = f"{sections[header]} {body}".strip() if header in sections else body
    return sections


def read_report(path: Path, root: Path, absolute: bool, keep_empty: bool):
    """Return (row | None, reason). `reason` explains why a report was dropped."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Could not read %s: %s", path, e)
        return None, "unreadable"

    sections = split_sections(text)

    missing = [h for h in REQUIRED if h not in sections]
    if missing:
        return None, "missing:" + "+".join(missing)

    if not keep_empty and any(not sections[h] for h in REQUIRED):
        return None, "empty_section"

    return {
        "path": str(path if absolute else path.relative_to(root)),
        "findings": sections["FINDINGS"],
        "impression": sections["IMPRESSION"],
        "history": sections["HISTORY"],
    }, None


def find_reports(root: Path):
    """Find study-level report files (s<study_id>.txt) under the root."""
    search_root = root / "files" if (root / "files").is_dir() else root
    return sorted(search_root.rglob("s[0-9]*.txt"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a CSV (path, findings, impression, history) from MIMIC-CXR "
                    "reports that contain HISTORY, FINDINGS and IMPRESSION sections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", required=True, help="MIMIC-CXR root folder.")
    p.add_argument("--output", default="mimic_reports.csv", help="Output CSV path.")
    p.add_argument("--absolute-paths", action="store_true",
                   help="Store absolute report paths instead of paths relative to --root.")
    p.add_argument("--keep-empty", action="store_true",
                   help="Keep reports where a required header exists but its text is empty.")
    p.add_argument("--workers", type=int, default=8, help="Reader threads.")
    p.add_argument("--limit", type=int, default=None, help="Only scan the first N reports (debug).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root folder not found: {root}")

    report_files = find_reports(root)
    if args.limit:
        report_files = report_files[: args.limit]
    if not report_files:
        raise SystemExit(f"No report (.txt) files found under {root}")
    logger.info("Found %d report files", len(report_files))

    rows, dropped = [], Counter()

    def work(f: Path):
        return read_report(f, root, args.absolute_paths, args.keep_empty)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row, reason in tqdm(ex.map(work, report_files), total=len(report_files), desc="Reports"):
            if row is not None:
                rows.append(row)
            else:
                dropped[reason] += 1

    df = pd.DataFrame(rows, columns=["path", "findings", "impression", "history"])
    df.to_csv(args.output, index=False)

    logger.info("Kept %d / %d reports -> %s", len(df), len(report_files), args.output)
    for reason, n in dropped.most_common():
        logger.info("  dropped (%s): %d", reason, n)


if __name__ == "__main__":
    main()