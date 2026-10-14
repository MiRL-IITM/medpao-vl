"""
Concept filtering: drop concept groups that are too rare.

Run AFTER concept_standardization.py and point it at that script's --out-dir:

    

Reads (from --standardization-dir):
    group_frequencies_all.csv, final_mapping.json, inverted_dict.json,
    concepts_df_standardized.csv
Writes everything into --out-dir (default: <standardization-dir>/filtered).

Main output, concepts_df_filtered.csv, has one row per report:
    <all original input columns>, <group name>_bin ... (one per accepted group),
    standardized_concepts
where each *_bin column is 1 (present), -1 (explicitly absent) or 0 (not mentioned /
contradictory) and standardized_concepts is a readable string such as
    "pneumothorax(-1), normal bones(1), clear lungs(1)"
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--standardization-dir", required=True,
                   help="--out-dir that was used for concept_standardization.py")
    p.add_argument("--out-dir", default=None, help="Default: <standardization-dir>/filtered")
    p.add_argument("--min-freq", type=int, default=100,
                   help="Keep concept groups whose total frequency (positive + negative) >= this value")
    return p


def main():
    args = build_parser().parse_args()
    src = args.standardization_dir
    out = args.out_dir or os.path.join(src, "filtered")
    os.makedirs(out, exist_ok=True)
    s_path = lambda n: os.path.join(src, n)
    o_path = lambda n: os.path.join(out, n)
    written = []

    def dump_json(obj, name):
        with open(o_path(name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        written.append(name)

    def dump_csv(df, name):
        df.to_csv(o_path(name), index=False)
        written.append(name)

    # ---- load standardization outputs --------------------------------------
    print("[1/4] Loading standardization outputs ...")
    freq_df = pd.read_csv(s_path("group_frequencies_all.csv"), dtype={"group_id": str})
    with open(s_path("final_mapping.json"), encoding="utf-8") as fh:
        final_mapping = json.load(fh)
    with open(s_path("inverted_dict.json"), encoding="utf-8") as fh:
        inverted = json.load(fh)
    df = pd.read_csv(s_path("concepts_df_standardized.csv"))
    print(f"  {len(freq_df):,} groups | {len(df):,} reports")

    # ---- filter groups ------------------------------------------------------
    print(f"[2/4] Filtering groups (total frequency >= {args.min_freq}) ...")
    freq_df = freq_df.sort_values("total", ascending=False).reset_index(drop=True)
    accepted_df = freq_df[freq_df["total"] >= args.min_freq]
    rejected_df = freq_df[freq_df["total"] < args.min_freq]
    accepted_ids = accepted_df["group_id"].tolist()
    accepted_set = set(accepted_ids)
    group_names = dict(zip(freq_df["group_id"], freq_df["group_name"]))
    print(f"  accepted {len(accepted_ids):,} | rejected {len(rejected_df):,}")
    dump_csv(accepted_df, "groups_accepted.csv")
    dump_csv(rejected_df, "groups_rejected.csv")

    dump_json({g: inverted[g] for g in accepted_ids}, "inverted_dict_filtered.json")
    filtered_mapping = {r: v for r, v in final_mapping.items() if str(v[1]) in accepted_set}
    dump_json(filtered_mapping, "final_mapping_filtered.json")

    # ---- per-report outputs -------------------------------------------------
    print("[3/4] Filtering per-report concepts + building label matrix ...")
    col_idx = {g: i for i, g in enumerate(accepted_ids)}
    label_matrix = np.zeros((len(df), len(accepted_ids)), dtype=np.int8)
    readable = []
    for i, cell in enumerate(df["standardized_concepts"]):
        entries = [e for e in json.loads(cell) if str(e[0]) in accepted_set] if isinstance(cell, str) else []
        pos_hits = {str(g) for g, _, pol in entries if pol > 0}
        neg_hits = {str(g) for g, _, pol in entries if pol < 0}
        for g in pos_hits - neg_hits:
            label_matrix[i, col_idx[g]] = 1
        for g in neg_hits - pos_hits:
            label_matrix[i, col_idx[g]] = -1
        # group both present and absent in the same report -> contradictory -> stays 0
        readable.append(", ".join(f"{name}({int(pol)})" for _, name, pol in entries))

    # One "<group name>_bin" column per accepted group (suffix the id if two groups share a name)
    bin_names = [f"{group_names[g]}_bin" for g in accepted_ids]
    dupes = {n for n in bin_names if bin_names.count(n) > 1}
    bin_names = [f"{group_names[g]} [{g}]_bin" if n in dupes else n for g, n in zip(accepted_ids, bin_names)]
    bin_df = pd.DataFrame(label_matrix, columns=bin_names, index=df.index)

    # original columns | *_bin columns | standardized_concepts  (same layout as the reference CSV)
    df_out = pd.concat([df.drop(columns=["standardized_concepts"]), bin_df], axis=1)
    df_out["standardized_concepts"] = readable
    dump_csv(df_out, "concepts_df_filtered.csv")
    np.save(o_path("label_matrix.npy"), label_matrix)
    written.append("label_matrix.npy")
    dump_json({"columns": accepted_ids, "names": [group_names[g] for g in accepted_ids]},
              "label_matrix_columns.json")

    # ---- stats --------------------------------------------------------------
    print("[4/4] Writing summary ...")
    n_with_any = int((label_matrix != 0).any(axis=1).sum()) if len(accepted_ids) else 0
    stats = {
        "min_freq": args.min_freq,
        "n_groups_total": len(freq_df),
        "n_groups_accepted": len(accepted_ids),
        "n_groups_rejected": int(len(rejected_df)),
        "n_groups_accepted_with_negative": int(accepted_df["has_negative"].sum()),
        "n_raw_concepts_kept": len(filtered_mapping),
        "n_raw_concepts_dropped": len(final_mapping) - len(filtered_mapping),
        "n_reports": len(df),
        "n_reports_with_at_least_one_kept_group": n_with_any,
        "mention_coverage_kept": round(float(accepted_df["total"].sum() / max(freq_df["total"].sum(), 1)), 4),
    }
    dump_json(stats, "filtering_summary.json")
    dump_json(vars(args), "filtering_config.json")

    print("\nDone. Files written to", out)
    for f in written:
        print("  -", f)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()