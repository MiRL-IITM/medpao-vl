"""
Concept standardization pipeline for MIMIC-CXR report concepts.

    concepts CSV
      -> parse + count raw concepts
      -> preprocess (lowercase, clean, negation stripping, lemmatization)
      -> map every unique preprocessed concept to a SNOMED CT concept (SapBERT)
      -> post-process "foreign device" categories with keyword rules
      -> group concepts, count group frequencies
      -> write many output files (described at the bottom of this file)

Input: the CSV written by concept_extraction.py (column `concepts`, comma-joined).
Frequency filtering is NOT done here; run concept_filtering.py on this script's
output directory afterwards.
    
"""

import argparse
import ast
import json
import os
import re
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

from snomedct_mapper import add_mapper_args, mapper_from_args


# =========================================================================== #
# 1. Parsing                                                                    #
# =========================================================================== #
def parse_concept_cell(cell) -> list:
    """Parse one `concepts` cell. Supported formats:
      * concept_extraction.py output : "no pleural effusion,cardiomegaly,normal heart contours"
      * stringified Python list      : "['no pleural effusion', 'cardiomegaly']"
    """
    if isinstance(cell, (list, tuple)):
        return [str(c).strip() for c in cell if str(c).strip()]
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    cell = str(cell).strip()
    if cell.startswith("[") and cell.endswith("]"):
        try:
            val = ast.literal_eval(cell)
            if isinstance(val, (list, tuple)):
                return [str(c).strip() for c in val if str(c).strip()]
        except (ValueError, SyntaxError):
            cell = cell[1:-1].replace("'", "").replace('"', "")   # legacy fallback
    return [c.strip() for c in cell.split(",") if c.strip()]


# =========================================================================== #
# 2. Preprocessing                                                              #
# =========================================================================== #
# Longest prefixes first. Extend freely.
NEGATION_PREFIXES = [
    "no evidence of", "negative for", "absence of", "free of", "without", "no", "not",
]
_NEG_RE = re.compile(r"^(?:" + "|".join(re.escape(p) for p in NEGATION_PREFIXES) + r")\s+")


def _get_lemmatizer(enabled: bool):
    if not enabled:
        return None
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        lem = WordNetLemmatizer()
        lem.lemmatize("tests")  # force load
        return lem
    except Exception as e:  # nltk missing / no network for download
        print(f"[warn] lemmatization disabled ({e}). `pip install nltk` and download 'wordnet'.")
        return None


class Preprocessor:
    """raw concept -> (normalized text, polarity).  polarity: +1 present, -1 negated."""

    def __init__(self, lemmatize: bool = True):
        self.lemmatizer = _get_lemmatizer(lemmatize)
        self._lemma = lru_cache(maxsize=None)(self._lemma_uncached)

    def _lemma_uncached(self, word: str) -> str:
        # Noun lemmatization only (plural -> singular). Verb lemmatization would
        # turn "enlarged" into "enlarge", which hurts medical matching.
        return self.lemmatizer.lemmatize(word, pos="n")

    def __call__(self, raw: str):
        text = raw.lower().strip()
        text = re.sub(r"[^a-z0-9\s/\-]", " ", text)   # drop punctuation, keep / and -
        text = re.sub(r"\s+", " ", text).strip()

        polarity = 1.0
        m = _NEG_RE.match(text)
        if m:
            polarity = -1.0
            text = text[m.end():].strip()

        if self.lemmatizer is not None:
            text = " ".join(self._lemma(w) for w in text.split())

        return text, polarity


# =========================================================================== #
# 3. SNOMED mapping                                                             #
# =========================================================================== #
def strip_semantic_tag(fsn: str) -> str:
    """'Pleural effusion (disorder)' -> 'Pleural effusion'."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", fsn).strip()


def map_to_snomed(mapper, texts: list, top_k: int, query_batch_size: int) -> dict:
    """normalized text -> list of top-k SNOMED candidates."""
    results = mapper.match(texts, top_k=top_k, query_batch_size=query_batch_size, show_progress=True)
    return {t: hits for t, hits in zip(texts, results)}


# =========================================================================== #
# 4. Foreign-device post-processing                                             #
# =========================================================================== #
# (group_id, group_name, [regexes]) — first matching rule wins (order = priority).
FOREIGN_DEVICE_RULES = [
    ("foreign1", "PICC", [r"\bpicc\b"]),
    ("foreign2", "pacemaker", [r"\bpacemakers?\b", r"\bdefibrillators?\b", r"\bicd\b", r"\baicd\b"]),
    ("foreign3", "Tubes", [r"\btubes?\b"]),
    ("foreign4", "surgical clips/hardware",
     [r"\bsternotom(?:y|ies)\b", r"\bsternal\b", r"\bclips?\b", r"\bhardware\b",
      r"\bstents?\b", r"\bvalves?\b", r"\bgrafts?\b"]),
    ("foreign5", "implants",
     [r"\barthroplast(?:y|ies)\b", r"\bprosthes[ie]s\b", r"\bprosthetic\b",
      r"\breplacements?\b", r"\bimplants?\b"]),
]
_FOREIGN_COMPILED = [(gid, name, [re.compile(p) for p in pats]) for gid, name, pats in FOREIGN_DEVICE_RULES]


def foreign_device_group(text: str):
    for gid, name, pats in _FOREIGN_COMPILED:
        if any(p.search(text) for p in pats):
            return gid, name
    return None


# =========================================================================== #
# Main                                                                          #
# =========================================================================== #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--concepts-csv", required=True, help="CSV with one row per report and a column of concept lists")
    p.add_argument("--concepts-column", default="concepts")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-lemmatize", action="store_true")
    p.add_argument("--top-k", type=int, default=5, help="SNOMED candidates stored per concept (top-1 is used)")
    p.add_argument("--min-similarity", type=float, default=0.0,
                   help="Top-1 SNOMED matches below this cosine similarity are kept as their own "
                        "'unmapped::<text>' group instead of being merged into a SNOMED concept")
    p.add_argument("--reuse-matches", action="store_true",
                   help="If snomed_candidates.json already exists in out-dir, skip the SNOMED matching step")
    add_mapper_args(p)
    return p


def main():
    args = build_parser().parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    path = lambda name: os.path.join(out, name)
    written = []

    def dump_json(obj, name):
        with open(path(name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        written.append(name)

    def dump_csv(df, name, index=False):
        df.to_csv(path(name), index=index)
        written.append(name)

    # ---- 1. load + parse ---------------------------------------------------
    print("[1/6] Loading concepts ...")
    df = pd.read_csv(args.concepts_csv)
    row_concepts = [parse_concept_cell(c) for c in df[args.concepts_column]]
    raw_counts = Counter(c for row in row_concepts for c in row)
    print(f"  {len(df):,} reports | {len(raw_counts):,} unique raw concepts")

    with open(path("unique_concepts_raw.txt"), "w", encoding="utf-8") as fh:
        fh.writelines(f"{c}\n" for c in sorted(raw_counts))
    written.append("unique_concepts_raw.txt")
    dump_csv(pd.DataFrame(raw_counts.most_common(), columns=["concept", "count"]), "concept_frequencies_raw.csv")

    # ---- 2. preprocess -----------------------------------------------------
    print("[2/6] Preprocessing ...")
    pre = Preprocessor(lemmatize=not args.no_lemmatize)
    prep = {raw: pre(raw) for raw in raw_counts}                  # raw -> (text, polarity)
    prep_df = pd.DataFrame(
        [(raw, t, pol, raw_counts[raw]) for raw, (t, pol) in prep.items()],
        columns=["raw_concept", "normalized", "polarity", "count"],
    ).sort_values("count", ascending=False)
    dump_csv(prep_df, "preprocessed_concepts.csv")

    unique_norm = sorted({t for t, _ in prep.values() if t})
    with open(path("unique_concepts_normalized.txt"), "w", encoding="utf-8") as fh:
        fh.writelines(f"{t}\n" for t in unique_norm)
    written.append("unique_concepts_normalized.txt")
    print(f"  {len(raw_counts):,} raw -> {len(unique_norm):,} unique normalized concepts "
          f"({int((prep_df.polarity < 0).sum()):,} negated raw concepts)")

    # ---- 3. SNOMED mapping -------------------------------------------------
    print("[3/6] Mapping to SNOMED CT ...")
    cand_file = path("snomed_candidates.json")
    if args.reuse_matches and os.path.exists(cand_file):
        with open(cand_file, encoding="utf-8") as fh:
            candidates = json.load(fh)
        print(f"  reused {cand_file}")
    else:
        mapper = mapper_from_args(args)
        candidates = map_to_snomed(mapper, unique_norm, args.top_k, args.query_batch_size)
        # slim candidates for disk (drop bulky synonym / ancestor lists)
        slim = {t: [{k: h[k] for k in ("concept_id", "matched_surface", "preferred_desc", "similarity")}
                    for h in hits] for t, hits in candidates.items()}
        dump_json(slim, "snomed_candidates.json")
        candidates = slim

    snomed_rows = []
    snomed_top1 = {}   # normalized text -> (concept_id, name, similarity)
    for t in unique_norm:
        hits = candidates.get(t, [])
        if hits and hits[0]["similarity"] >= args.min_similarity:
            h = hits[0]
            snomed_top1[t] = (h["concept_id"], strip_semantic_tag(h["preferred_desc"]), h["similarity"])
        else:
            snomed_top1[t] = (f"unmapped::{t}", t, hits[0]["similarity"] if hits else 0.0)
        snomed_rows.append({
            "normalized": t,
            "snomed_id": snomed_top1[t][0],
            "snomed_name": snomed_top1[t][1],
            "similarity": snomed_top1[t][2],
            "mapped": not snomed_top1[t][0].startswith("unmapped::"),
        })
    dump_csv(pd.DataFrame(snomed_rows).sort_values("similarity"), "snomed_mapping_top1.csv")
    print(f"  {sum(r['mapped'] for r in snomed_rows):,}/{len(snomed_rows):,} mapped "
          f"(min similarity {args.min_similarity})")

    # ---- 4. foreign devices -----------------------------------------------
    print("[4/6] Foreign-device post-processing ...")
    group_of_text = {}        # normalized text -> (group_id, group_name)
    foreign_rows = []
    for t, (sid, sname, _) in snomed_top1.items():
        fd = foreign_device_group(t)
        if fd:
            group_of_text[t] = fd
            foreign_rows.append({"normalized": t, "foreign_group": fd[0], "foreign_name": fd[1],
                                 "previous_snomed_id": sid, "previous_snomed_name": sname})
        else:
            group_of_text[t] = (sid, sname)
    dump_csv(pd.DataFrame(foreign_rows, columns=["normalized", "foreign_group", "foreign_name",
                                                  "previous_snomed_id", "previous_snomed_name"]),
             "foreign_device_assignments.csv")
    print(f"  {len(foreign_rows):,} normalized concepts reassigned to foreign-device groups")

    # ---- 5. final mapping + inverted dict ---------------------------------
    print("[5/6] Building final mapping ...")
    final_mapping = {}        # raw concept -> [group_name, group_id, polarity]
    for raw, (t, pol) in prep.items():
        if not t:
            continue
        gid, gname = group_of_text[t]
        final_mapping[raw] = [gname, gid, pol]
    dump_json(final_mapping, "final_mapping.json")
    dump_csv(pd.DataFrame([(r, *v, raw_counts[r]) for r, v in final_mapping.items()],
                          columns=["raw_concept", "group_name", "group_id", "polarity", "count"])
             .sort_values("count", ascending=False), "final_mapping.csv")

    inverted = {}
    group_names = {}
    for raw, (gname, gid, pol) in final_mapping.items():
        group_names[gid] = gname
        inverted.setdefault(gid, {"name": gname, "positive": [], "negative": []})
        inverted[gid]["positive" if pol > 0 else "negative"].append(raw)
    dump_json(inverted, "inverted_dict.json")

    # ---- 6. group frequencies + per-report standardized concepts ------------
    print("[6/6] Computing group frequencies and per-report concepts ...")
    freq_rows = []
    for gid, v in inverted.items():
        pos = sum(raw_counts[c] for c in v["positive"])
        neg = sum(raw_counts[c] for c in v["negative"])
        freq_rows.append({"group_id": gid, "group_name": v["name"], "positive": pos, "negative": neg,
                          "total": pos + neg, "n_raw_concepts": len(v["positive"]) + len(v["negative"]),
                          "has_negative": neg > 0})
    freq_df = pd.DataFrame(freq_rows).sort_values("total", ascending=False).reset_index(drop=True)
    dump_csv(freq_df, "group_frequencies_all.csv")

    std_rows = []
    for row in row_concepts:
        seen, entries = set(), []
        for raw in row:
            m = final_mapping.get(raw)
            if not m:
                continue
            gname, gid, pol = m
            if (gid, pol) not in seen:
                seen.add((gid, pol))
                entries.append([gid, gname, pol])
        std_rows.append(json.dumps(entries))
    df_out = df.copy()
    df_out["standardized_concepts"] = std_rows
    dump_csv(df_out, "concepts_df_standardized.csv")

    stats = {
        "n_reports": len(df),
        "n_unique_raw_concepts": len(raw_counts),
        "n_unique_normalized_concepts": len(unique_norm),
        "n_snomed_mapped": int(sum(r["mapped"] for r in snomed_rows)),
        "n_foreign_reassigned": len(foreign_rows),
        "n_groups_total": len(freq_df),
        "min_similarity": args.min_similarity,
        "lemmatization": not args.no_lemmatize,
    }
    dump_json(stats, "summary_stats.json")
    dump_json({k: (v if not isinstance(v, set) else sorted(v)) for k, v in vars(args).items()}, "run_config.json")

    print("\nDone. Files written to", out)
    for f in written:
        print("  -", f)
    print(json.dumps(stats, indent=2))
    print("\nNext: python concept_filtering.py --standardization-dir", out)


if __name__ == "__main__":
    main()

