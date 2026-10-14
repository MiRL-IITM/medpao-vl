import json
import faiss
import numpy as np
from llama_cpp import Llama
import psutil
import os
import threading
import time

# ───────────────────────────── CONFIG (edit here) ─────────────────────────────
# --- SNOMED index + SapBERT model (machine-specific paths: change these first) ---
SAVE_DIR          = '/home/user/agent/mcp_codes/tools/snomed_rich_index_less_conc'   # folder holding the index files
SAPBERT_GGUF      = '/home/user/agent/mcp_codes/tools/sapbert.gguf'                  # SapBERT in GGUF format

# --- File names inside SAVE_DIR ---
EMB_FILE          = f'{SAVE_DIR}/snomed_embeddings.npy'
EMB_META_FILE     = f'{SAVE_DIR}/embedding_metadata.jsonl'
CONCEPT_META_FILE = f'{SAVE_DIR}/concept_metadata.jsonl'
CONCEPT_ID_MAP    = f'{SAVE_DIR}/concept_id_to_index.json'
FAISS_INDEX       = f'{SAVE_DIR}/snomed_ivf_sq8.faiss'

# --- Retrieval / model settings ---
TOP_K             = 5      # SNOMED candidates returned per query
N_GPU_LAYERS      = 0      # llama.cpp layers offloaded to GPU (0 = CPU only)
N_CTX             = 512    # llama.cpp context length for the encoder
NPROBE            = 32     # FAISS IVF cells searched per query
NLIST             = 256    # FAISS IVF cells (only used by build_and_save_index)
RAM_POLL_INTERVAL_S = 0.05 # PeakRAMMonitor sampling interval

# --- Pipeline files in the working "savings" folder ---
SAVINGS_DIR_NAME  = 'savings'                   # default: <cwd>/savings
NEW_CONCEPTS_FILE = 'new_concepts_output.json'  # cache-miss concepts (preferred input if present)
CONCEPTS_FILE     = 'concepts_output.json'      # all extracted concepts (written by get_concept.py)
SEGREGATED_FILE   = 'segregated.json'           # output: SNOMED matches

# --- Demo run (python ontology_map_sapbert.py) ---
EXAMPLE_SAVING_DIR = '/home/user/agent'
EXAMPLE_QUERIES    = 'clear lungs, stable cardiomediastinal silhouette, stable hilar contours, sternotomy wires, mediastinal clips, no pleural effusion, no pneumothorax'
EXAMPLE_TOP_K      = 1
# ──────────────────────────────────────────────────────────────────────────────


class SnomedMatcher:
    def __init__(
        self,
        SAVING_DIR,
        faiss_index:       str = FAISS_INDEX,
        emb_meta_file:     str = EMB_META_FILE,
        concept_meta_file: str = CONCEPT_META_FILE,
        concept_id_map:    str = CONCEPT_ID_MAP,
        gguf_path:         str = SAPBERT_GGUF,
        n_gpu_layers:      int = N_GPU_LAYERS,
        nprobe:            int = NPROBE,
    ):
        self.gguf_path    = gguf_path
        self.n_gpu_layers = n_gpu_layers
        self.saving_dir = SAVING_DIR

        # ── 1. Load FAISS index from disk (no embedding matrix needed) ───────
        print(f"Loading FAISS index from {faiss_index} …")
        self.index = faiss.read_index(faiss_index)
        self.index.nprobe = nprobe          # must re-set after loading
        print(f"  {self.index.ntotal:,} vectors in index  |  nprobe={nprobe}")

        # ── 2. Byte-offset indices for lazy JSONL reads ──────────────────────
        print("Building byte-offset indices …")
        self.emb_offsets = self._build_offsets(emb_meta_file)
        self.con_offsets = self._build_offsets(concept_meta_file)
        print(f"  emb rows : {len(self.emb_offsets):,}")
        print(f"  con rows : {len(self.con_offsets):,}")

        # ── 3. concept_id → line-index map ───────────────────────────────────
        with open(concept_id_map, encoding='utf-8') as fh:
            self.cid_to_idx: dict[str, int] = json.load(fh)

        # ── 4. Keep JSONL file handles open for fast seeks ───────────────────
        self.emb_meta_fh = open(emb_meta_file,     'rb')
        self.con_meta_fh = open(concept_meta_file, 'rb')

        # ── 5. Load SapBERT GGUF once ────────────────────────────────────────
        print(f"Loading SapBERT GGUF (n_gpu_layers={n_gpu_layers}) …")
        self.model = Llama(
            model_path   = gguf_path,
            embedding    = True,
            n_ctx        = N_CTX,
            n_gpu_layers = n_gpu_layers,
            verbose      = False,
        )
        print("SnomedMatcher ready.\n")

    def __del__(self):
        for fh in (self.emb_meta_fh, self.con_meta_fh):
            try:
                fh.close()
            except Exception:
                pass

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _build_offsets(self, filepath: str) -> np.ndarray:
        offsets = []
        with open(filepath, 'rb') as fh:
            while True:
                pos  = fh.tell()
                line = fh.readline()
                if not line:
                    break
                offsets.append(pos)
        return np.array(offsets, dtype=np.int64)

    def _get_emb_meta(self, idx: int) -> dict:          # idx is a plain Python int
        self.emb_meta_fh.seek(int(self.emb_offsets[idx]))
        return json.loads(self.emb_meta_fh.readline().decode('utf-8'))

    def _get_concept_meta(self, concept_id: str) -> dict:
        row_idx = int(self.cid_to_idx[concept_id])
        self.con_meta_fh.seek(int(self.con_offsets[row_idx]))
        return json.loads(self.con_meta_fh.readline().decode('utf-8'))

    # ── Encoder ──────────────────────────────────────────────────────────────
    def _encode(self, queries: list[str]) -> np.ndarray:
        """Returns float32 numpy array (N, hidden) — what FAISS expects."""
        vecs = []
        for q in queries:
            res = self.model.create_embedding(q)
            vec = np.array(res["data"][0]["embedding"], dtype=np.float32)
            if vec.ndim == 2:           # token-level → mean pool
                vec = vec.mean(axis=0)  # (hidden,)
            vecs.append(vec)
        out = np.stack(vecs).astype(np.float32)    # (N, hidden) contiguous float32
        print(f"Encoded {len(queries)} query/queries: {out.shape}")
        return out

    # ── Main API ─────────────────────────────────────────────────────────────
    def match(self, queries, top_k: int = TOP_K, saving_dir = os.path.join(os.getcwd(), SAVINGS_DIR_NAME)) -> list[list[dict]]:
        """
        Parameters
        ----------
        queries : free-text strings to normalise against SNOMED
        top_k   : candidates per query, sorted by L2 ascending

        Returns
        -------
        list[list[dict]]  — outer = queries, inner = top_k hits:
            query, concept_id, matched_surface, preferred_desc,
            all_descs, ancestors, distance
        """
        print("===========================================Doing sapbert Ontology mapping=================================")

        new_path = os.path.join(saving_dir, NEW_CONCEPTS_FILE)
        path = os.path.join(saving_dir, CONCEPTS_FILE)
        if os.path.exists(new_path):
            try:
                with open(new_path) as f:
                    data = json.load(f)
                concept_list = list(data.keys())
            except Exception as e:
                return "From if"
        
        elif os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                concept_list = list(data.keys())
            except Exception as e:
                return data
            
        else:
            concept_list = queries.split(', ')

        if not len(concept_list):
            return None

        queries = concept_list

        # ── Step 1: Encode ───────────────────────────────────────────────────
        q_vecs = self._encode(queries)                  # (N_q, hidden) float32

        # ── Step 2: FAISS ANN search ─────────────────────────────────────────
        # distances and indices are both (N_q, top_k) numpy arrays
        distances, indices = self.index.search(q_vecs, top_k)

        # ── Step 3: Resolve metadata ─────────────────────────────────────────
        all_results: list[list[dict]] = []

        for i, query in enumerate(queries):
            print(f"Top distances for '{query}':", distances[i].tolist())

            for dist_val, emb_idx in zip(distances[i].tolist(), indices[i].tolist()):
                if emb_idx == -1:           # FAISS pads with -1 when index < top_k
                    continue

                emb_row = self._get_emb_meta(emb_idx)   # emb_idx is now a plain int
                cid     = emb_row["concept_id"]
                concept = self._get_concept_meta(cid)

                q_results = {
                    "query"          : query,
                    "concept_id"     : cid,
                    "matched_surface": emb_row["surface_form"],
                    "preferred_desc" : concept["preferred_desc"],
                    "all_descs"      : concept["all_descs"],
                    "ancestors"      : concept["ancestors"],
                    "distance"       : round(dist_val, 6),
                }

            all_results.append(q_results)

        file_path = os.path.join(self.saving_dir, SEGREGATED_FILE)
        with open(file_path, 'w') as json_file:
            json.dump(all_results, json_file, indent=4)    

        return all_results


# ── Build index (run once offline, skip if already exists) ───────────────────
def build_and_save_index(
    emb_file: str,
    out_path: str,
    nlist:    int = NLIST,
    nprobe:   int = NPROBE,
) -> None:
    if os.path.exists(out_path):
        print(f"FAISS index already exists at {out_path}, skipping build.")
        return

    print("Building FAISS IVF+SQ8 index …")
    emb = np.load(emb_file).astype(np.float32)
    dim = emb.shape[1]

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, nlist,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_L2,
    )
    index.train(emb)
    index.add(emb)
    index.nprobe = nprobe
    faiss.write_index(index, out_path)
    print(f"Saved index: {index.ntotal:,} vectors → {out_path}")
    del emb     # free before SnomedMatcher loads


# ── Peak RAM monitor ─────────────────────────────────────────────────────────
class PeakRAMMonitor:
    def __init__(self, interval_s: float = RAM_POLL_INTERVAL_S):
        self._proc     = psutil.Process(os.getpid())
        self._interval = interval_s
        self._peak_mb  = 0.0
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss / 1024 ** 2
            if rss > self._peak_mb:
                self._peak_mb = rss
            time.sleep(self._interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return self._peak_mb


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # build_and_save_index(EMB_FILE, FAISS_INDEX)     # no-op if file exists

    monitor = PeakRAMMonitor().start()

    matcher = SnomedMatcher(SAVING_DIR=EXAMPLE_SAVING_DIR)

    start_time = time.perf_counter()
    results    = matcher.match(queries=EXAMPLE_QUERIES, top_k=EXAMPLE_TOP_K)
    end_time   = time.perf_counter()
    print(results)
    for hit in results:
        print(hit)
        print(f"[{hit['distance']:.4f}]  {hit['concept_id']}  |  "
              f"{hit['matched_surface']!r}  →  {hit['preferred_desc']}")

    peak = monitor.stop()
    print(f"\nPeak RAM usage      : {peak:.2f} MB")
    print(f"Total query latency : {end_time - start_time:.3f} seconds")