"""
SNOMED CT mapper: embeds SNOMED CT terms with SapBERT and matches free-text
queries to concepts by exact cosine similarity (dense matmul, no ANN index).

Can be used
  * as a library  (see `add_mapper_args` / `mapper_from_args`, used by
    concept_standardization.py), or
  * as a CLI:
        python snomedct_mapper.py \
            --desc-file  <sct2_Description_Snapshot-en_INT_*.txt> \
            --rel-file   <sct2_Relationship_Snapshot_INT_*.txt> \
            --save-dir   ./snomed_index \
            --queries "vague opacity" "pleural effusion" --top-k 5
"""

import argparse
import json
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd
import psutil
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# --------------------------------------------------------------------------- #
# Peak-RAM helper (optional, enabled with --measure-ram)                        #
# --------------------------------------------------------------------------- #
class PeakRAM:
    def __init__(self, process, interval=0.01):
        self.process = process
        self.interval = interval
        self.stop = False
        self.peak = 0

    def monitor(self):
        while not self.stop:
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)


@contextmanager
def measure_peak_ram(name="Block"):
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss
    mon = PeakRAM(process)
    thread = threading.Thread(target=mon.monitor, daemon=True)
    thread.start()
    try:
        yield
    finally:
        mon.stop = True
        thread.join()
        after = process.memory_info().rss
        print(f"\n[{name}] RAM  before={before / 1024**3:.3f} GB  "
              f"after={after / 1024**3:.3f} GB  peak={mon.peak / 1024**3:.3f} GB  "
              f"peak increase={(mon.peak - before) / 1024**3:.3f} GB")


# --------------------------------------------------------------------------- #
# Mapper                                                                        #
# --------------------------------------------------------------------------- #
class SnomedCTMapping:
    ISA_TYPE_ID = "116680003"             # relationship type: IS A
    FSN_TYPE_ID = "900000000000003001"    # description type: Fully Specified Name

    DEFAULT_TARGET_ROOTS = {
        "404684003",  # Clinical Finding
        "71388002",   # Procedure
        "123037004",  # Body Structure
    }

    def __init__(
        self,
        desc_file: str,
        rel_file: str,
        save_dir: str,
        target_roots: Optional[set] = None,
        sapbert_model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: str = "auto",
        max_length: int = 64,
        embed_batch_size: int = 256,
        force_rebuild: bool = False,
    ):
        self.desc_file = desc_file
        self.rel_file = rel_file
        self.save_dir = save_dir
        self.target_roots = set(target_roots) if target_roots else set(self.DEFAULT_TARGET_ROOTS)
        self.max_length = max_length
        self.embed_batch_size = embed_batch_size

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested but not available -> falling back to CPU.")
            device = "cpu"
        self.device = device

        os.makedirs(self.save_dir, exist_ok=True)
        self.EMB_FILE = os.path.join(self.save_dir, "snomed_embeddings.npy")
        self.EMB_META_FILE = os.path.join(self.save_dir, "embedding_metadata.jsonl")
        self.CONCEPT_META_FILE = os.path.join(self.save_dir, "concept_metadata.jsonl")
        self.CONCEPT_ID_MAP = os.path.join(self.save_dir, "concept_id_to_index.json")

        print(f"Loading SapBERT ({sapbert_model_name}) on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(sapbert_model_name)
        self.model = AutoModel.from_pretrained(sapbert_model_name).to(self.device)
        self.model.eval()

        self.embeddings = None
        self.emb_offsets = None
        self.con_offsets = None
        self.cid_to_idx = None
        self.emb_meta_fh = None
        self.con_meta_fh = None

        index_ready = all(
            os.path.exists(p)
            for p in (self.EMB_FILE, self.EMB_META_FILE, self.CONCEPT_META_FILE, self.CONCEPT_ID_MAP)
        )
        if force_rebuild or not index_ready:
            if not (desc_file and rel_file):
                raise ValueError("--desc-file and --rel-file are required to build the index.")
            self._build()
        else:
            self._load()

    def __del__(self):
        for fh in (getattr(self, "emb_meta_fh", None), getattr(self, "con_meta_fh", None)):
            try:
                fh.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # BUILD                                                               #
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        print("Loading Descriptions...")
        df_desc = pd.read_csv(self.desc_file, sep="\t", dtype=str)
        df_desc = df_desc[df_desc["active"] == "1"]

        id_to_fsn: dict = {}
        id_to_syns: dict = {}
        for cid, term, type_id in tqdm(
            zip(df_desc["conceptId"], df_desc["term"], df_desc["typeId"]),
            total=len(df_desc), desc="Parsing Terms",
        ):
            if type_id == self.FSN_TYPE_ID:
                id_to_fsn[cid] = term
            else:
                id_to_syns.setdefault(cid, []).append(term)

        print("Loading Relationships...")
        df_rel = pd.read_csv(self.rel_file, sep="\t", dtype=str)
        df_rel = df_rel[(df_rel["active"] == "1") & (df_rel["typeId"] == self.ISA_TYPE_ID)]

        child_to_parents: dict = {}
        parent_to_children: dict = {}
        for src, dest in tqdm(zip(df_rel["sourceId"], df_rel["destinationId"]),
                              total=len(df_rel), desc="Parsing Graph"):
            child_to_parents.setdefault(src, set()).add(dest)
            parent_to_children.setdefault(dest, set()).add(src)

        print("Collecting allowed concepts via BFS...")

        def get_all_descendants(root_id: str) -> set:
            visited, queue = set(), deque([root_id])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                queue.extend(c for c in parent_to_children.get(node, []) if c not in visited)
            return visited

        allowed_concepts: set = set()
        for root_id in self.target_roots:
            d = get_all_descendants(root_id)
            print(f"  {id_to_fsn.get(root_id, root_id)}: {len(d):,} concepts")
            allowed_concepts.update(d)
        print(f"Total allowed concepts: {len(allowed_concepts):,}")

        def get_ancestor_labels(cid: str, max_hops: int = 5) -> list:
            seen, frontier, ancestors, hops = set(), list(child_to_parents.get(cid, [])), [], 0
            while frontier and hops < max_hops:
                nxt = []
                for pid in frontier:
                    if pid in seen:
                        continue
                    seen.add(pid)
                    label = id_to_fsn.get(pid, "")
                    if label:
                        ancestors.append(label)
                    nxt.extend(child_to_parents.get(pid, []))
                frontier = nxt
                hops += 1
            return ancestors

        embedding_rows, concept_rows, cid_to_con_idx = [], [], {}
        print("Constructing per-concept metadata and embedding rows...")
        for cid, fsn in tqdm(id_to_fsn.items()):
            if cid not in allowed_concepts:
                continue
            all_terms = [fsn] + id_to_syns.get(cid, [])
            cid_to_con_idx[cid] = len(concept_rows)
            concept_rows.append({
                "concept_id": cid,
                "preferred_desc": fsn,
                "all_descs": all_terms,
                "ancestors": [{"desc": l} for l in get_ancestor_labels(cid)],
            })
            for term in all_terms:
                embedding_rows.append({"concept_id": cid, "surface_form": term})

        print(f"  Concepts to index : {len(concept_rows):,}")
        print(f"  Embedding rows    : {len(embedding_rows):,}")

        embeddings = self._encode([r["surface_form"] for r in embedding_rows], show_progress=True)
        assert len(embeddings) == len(embedding_rows), "Vector count must match metadata rows!"

        print("Saving outputs...")
        np.save(self.EMB_FILE, embeddings)
        with open(self.EMB_META_FILE, "w", encoding="utf-8") as fh:
            for row in embedding_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(self.CONCEPT_META_FILE, "w", encoding="utf-8") as fh:
            for row in concept_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(self.CONCEPT_ID_MAP, "w", encoding="utf-8") as fh:
            json.dump(cid_to_con_idx, fh)
        print(f"  Index written to {self.save_dir}")

        self._load()

    # ------------------------------------------------------------------ #
    # LOAD                                                                #
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        print(f"Loading embedding matrix from {self.EMB_FILE} ...")
        emb = np.load(self.EMB_FILE).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.embeddings = emb / norms
        print(f"  {self.embeddings.shape[0]:,} vectors | dim={self.embeddings.shape[1]} "
              f"({self.embeddings.nbytes / 1024**2:.1f} MB)")

        self.emb_offsets = self._build_offsets(self.EMB_META_FILE)
        self.con_offsets = self._build_offsets(self.CONCEPT_META_FILE)
        with open(self.CONCEPT_ID_MAP, encoding="utf-8") as fh:
            self.cid_to_idx = json.load(fh)
        self.emb_meta_fh = open(self.EMB_META_FILE, "rb")
        self.con_meta_fh = open(self.CONCEPT_META_FILE, "rb")
        print("SnomedCTMapping ready.\n")

    @staticmethod
    def _build_offsets(filepath: str) -> np.ndarray:
        offsets = []
        with open(filepath, "rb") as fh:
            while True:
                pos = fh.tell()
                if not fh.readline():
                    break
                offsets.append(pos)
        return np.array(offsets, dtype=np.int64)

    def _get_emb_meta(self, idx: int) -> dict:
        self.emb_meta_fh.seek(int(self.emb_offsets[idx]))
        return json.loads(self.emb_meta_fh.readline().decode("utf-8"))

    def _get_concept_meta(self, concept_id: str) -> dict:
        self.con_meta_fh.seek(int(self.con_offsets[int(self.cid_to_idx[concept_id])]))
        return json.loads(self.con_meta_fh.readline().decode("utf-8"))

    # ------------------------------------------------------------------ #
    # ENCODE                                                              #
    # ------------------------------------------------------------------ #
    def _encode(self, texts: list, show_progress: bool = False) -> np.ndarray:
        """CLS-pooled SapBERT embeddings -> float32 array (N, hidden)."""
        reps = []
        rng = range(0, len(texts), self.embed_batch_size)
        it = tqdm(rng, desc="Embedding") if show_progress else rng
        with torch.no_grad():
            for i in it:
                toks = self.tokenizer(
                    texts[i:i + self.embed_batch_size],
                    padding=True, max_length=self.max_length,
                    truncation=True, return_tensors="pt",
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}
                out = self.model(**toks)
                reps.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.concatenate(reps, axis=0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # MATCH                                                               #
    # ------------------------------------------------------------------ #
    def match(self, queries: list, top_k: int = 5, search_multiplier: int = 8,
              query_batch_size: int = 64, show_progress: bool = False) -> list:
        """
        Returns list (per query) of up to `top_k` dicts, one per unique SNOMED
        concept, best surface form kept:
            query, concept_id, matched_surface, preferred_desc, all_descs,
            ancestors, similarity
        Queries are processed in chunks of `query_batch_size` so the
        (queries x corpus) similarity matrix never blows up memory.
        """
        if self.embeddings is None:
            raise RuntimeError("Embeddings not loaded/built yet.")

        all_results: list = []
        starts = range(0, len(queries), query_batch_size)
        it = tqdm(starts, desc="SNOMED matching") if show_progress else starts
        for s in it:
            all_results.extend(self._match_chunk(queries[s:s + query_batch_size], top_k, search_multiplier))
        return all_results

    def _match_chunk(self, queries: list, top_k: int, search_multiplier: int) -> list:
        q = self._encode(queries)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        q = q / norms

        fetch_k = min(max(top_k * search_multiplier, top_k), self.embeddings.shape[0])
        sims = q @ self.embeddings.T
        part_idx = np.argpartition(-sims, kth=fetch_k - 1, axis=1)[:, :fetch_k]
        rows = np.arange(sims.shape[0])[:, None]
        part_sims = sims[rows, part_idx]
        order = np.argsort(-part_sims, axis=1)
        indices = part_idx[rows, order]
        similarities = part_sims[rows, order]

        results = []
        for i, query in enumerate(queries):
            seen, hits = set(), []
            for sim_val, emb_idx in zip(similarities[i].tolist(), indices[i].tolist()):
                emb_row = self._get_emb_meta(emb_idx)
                cid = emb_row["concept_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                concept = self._get_concept_meta(cid)
                hits.append({
                    "query": query,
                    "concept_id": cid,
                    "matched_surface": emb_row["surface_form"],
                    "preferred_desc": concept["preferred_desc"],
                    "all_descs": concept["all_descs"],
                    "ancestors": concept["ancestors"],
                    "similarity": round(sim_val, 6),
                })
                if len(hits) >= top_k:
                    break
            results.append(hits)
        return results


# --------------------------------------------------------------------------- #
# CLI plumbing (reused by concept_standardization.py)                           #
# --------------------------------------------------------------------------- #
def add_mapper_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("SNOMED CT mapper")
    g.add_argument("--desc-file", help="SNOMED RF2 Description snapshot .txt (needed only when building the index)")
    g.add_argument("--rel-file", help="SNOMED RF2 Relationship snapshot .txt (needed only when building the index)")
    g.add_argument("--snomed-index-dir", "--save-dir", dest="save_dir", required=False,
                   help="Directory where the SNOMED embedding index is stored / loaded from")
    g.add_argument("--target-roots", nargs="+", default=None,
                   help="SNOMED root concept IDs to index (default: Clinical Finding, Procedure, Body Structure)")
    g.add_argument("--sapbert-model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    g.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', 'cuda:1', ...")
    g.add_argument("--max-length", type=int, default=64)
    g.add_argument("--embed-batch-size", type=int, default=256)
    g.add_argument("--query-batch-size", type=int, default=64,
                   help="Queries per similarity-matrix chunk when matching (lower = less RAM)")
    g.add_argument("--force-rebuild", action="store_true", help="Rebuild the SNOMED embedding index")
    return parser


def mapper_from_args(args) -> SnomedCTMapping:
    if not args.save_dir:
        raise SystemExit("error: --snomed-index-dir is required")
    return SnomedCTMapping(
        desc_file=args.desc_file,
        rel_file=args.rel_file,
        save_dir=args.save_dir,
        target_roots=set(args.target_roots) if args.target_roots else None,
        sapbert_model_name=args.sapbert_model,
        device=args.device,
        max_length=args.max_length,
        embed_batch_size=args.embed_batch_size,
        force_rebuild=args.force_rebuild,
    )


def main():
    parser = argparse.ArgumentParser(description="Build / query a SNOMED CT SapBERT index.")
    add_mapper_args(parser)
    parser.add_argument("--queries", nargs="*", default=[], help="Free-text queries to match")
    parser.add_argument("--queries-file", help="Text file with one query per line")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", help="Write full match results to this JSON file")
    parser.add_argument("--measure-ram", action="store_true", help="Report peak RAM")
    args = parser.parse_args()

    queries = list(args.queries)
    if args.queries_file:
        with open(args.queries_file, encoding="utf-8") as fh:
            queries += [l.strip() for l in fh if l.strip()]

    ctx = measure_peak_ram("snomed mapper") if args.measure_ram else _null_ctx()
    with ctx:
        mapper = mapper_from_args(args)
        if not queries:
            print("Index ready; no queries given.")
            return
        start = time.perf_counter()
        results = mapper.match(queries, top_k=args.top_k, query_batch_size=args.query_batch_size)
        print(f"Matched {len(queries)} queries in {time.perf_counter() - start:.2f}s")

    for q, hits in zip(queries, results):
        print(f"\n{q!r}")
        for h in hits:
            print(f"  [{h['similarity']:.4f}] {h['concept_id']} | {h['matched_surface']!r} -> {h['preferred_desc']}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)


@contextmanager
def _null_ctx():
    yield


if __name__ == "__main__":
    main()