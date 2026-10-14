import os
import json
from rapidfuzz import fuzz, process

# ───────────────────────────── CONFIG (edit here) ─────────────────────────────
SIMILARITY_THRESHOLD = 95   # Scale of 0–100 (equivalent to 0.95 cosine)
TOP_K_MATCHES        = 5    # fuzzy matches retrieved per concept

SAVINGS_DIR_NAME     = "savings"                    # default: <cwd>/savings
CACHE_DIR_SUFFIX     = "_cache"                     # cache dir = <savings_dir> + this suffix
CACHED_VOCAB_FILE    = "cached_vocab.json"          # inside the cache dir
CONCEPTS_FILE        = "concepts_output.json"       # input  (written by get_concept.py)
CATEGORIZED_FILE     = "categorized_concepts.json"  # output: cache hits
NEW_CONCEPTS_FILE    = "new_concepts_output.json"   # output: cache misses
# ──────────────────────────────────────────────────────────────────────────────

def check_the_cache(concepts=None, savings_dir=None):
    if savings_dir is None:
        savings_dir = os.path.join(os.getcwd(), SAVINGS_DIR_NAME)

    cache_dir = savings_dir + CACHE_DIR_SUFFIX
    os.makedirs(cache_dir, exist_ok=True)

    vocab_path = os.path.join(cache_dir, CACHED_VOCAB_FILE)
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
    else:
        vocab = {}

    concepts_path = os.path.join(savings_dir, CONCEPTS_FILE)
    with open(concepts_path) as f:
        current_concepts = json.load(f)

    savings = {}
    new_concept_out = {}
    comparison_details = []

    vocab_keys = list(vocab.keys())

    for query1, sentence in current_concepts.items():
        # Get top-k fuzzy matches from vocab
        top5 = process.extract(
            query1,
            vocab_keys,
            scorer=fuzz.token_sort_ratio,  # handles word-order variations
            limit=TOP_K_MATCHES
        )

        pair_scores = [
            {"cached_concept": match, "similarity": round(score / 100, 4)}
            for match, score, _ in top5
        ]

        best_match, best_score_raw, _ = top5[0] if top5 else (None, 0, None)
        best_score = round(best_score_raw / 100, 4)
        found = best_score_raw >= SIMILARITY_THRESHOLD

        detail = {
            "query_concept": query1,
            "matched": found,
            "best_match": best_match,
            "best_score": best_score,
            "threshold": SIMILARITY_THRESHOLD / 100,
            "top5_pairs": pair_scores,
        }
        comparison_details.append(detail)

        if found:
            savings[query1] = vocab[best_match]
        else:
            new_concept_out[query1] = sentence

    # Save outputs
    with open(os.path.join(savings_dir, CATEGORIZED_FILE), "w") as f:
        json.dump(savings, f, indent=4)

    with open(os.path.join(savings_dir, NEW_CONCEPTS_FILE), "w") as f:
        json.dump(new_concept_out, f, indent=4)

    # Uncomment to save comparison log:
    # with open(os.path.join(savings_dir, "cache_comparison_log.json"), "w") as f:
    #     json.dump(comparison_details, f, indent=4)

    cache_hit = len(savings)
    cache_miss = len(new_concept_out)
    return cache_hit, cache_miss