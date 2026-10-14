import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────────────────── CONFIG (edit here) ─────────────────────────────
SAPBERT_MODEL        = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"   # HF id or local path
SIMILARITY_THRESHOLD = 0.7     # default cosine-similarity cut-off for "concept is covered"
MAX_TOKEN_LENGTH     = 25      # padded/truncated length used when embedding one concept
MAX_CONCEPT_TOKENS   = 512     # generated concepts longer than this are skipped
TOP_K_PAIRS_LOGGED   = 5       # best-matching pairs kept per concept in the comparison log

SAVINGS_DIR_NAME     = "savings"                          # <cwd>/savings
COMPARISON_LOG_FILE  = "concept_comparison_details.json"  # output: full comparison log
MISSING_FILE         = "missing_concepts.json"            # output: only written if something is missing
# ──────────────────────────────────────────────────────────────────────────────


class Comparison:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL)
        self.model = AutoModel.from_pretrained(SAPBERT_MODEL)
            
    def _parse_vision_concepts_to_dict(self, vision_concepts):
        """Convert vision concepts string or dict into {concept: 'present'/'absent'} dict."""
        if isinstance(vision_concepts, dict):
            return vision_concepts
        
        if isinstance(vision_concepts, str):
            result = {}
            for c in vision_concepts.split(','):
                c = c.strip()
                if c.endswith('(1.0)'):
                    result[c[:-5].strip()] = 'present'
                elif c.endswith('(-1.0)'):
                    result[c[:-6].strip()] = 'absent'
                elif c:
                    result[c] = 'present'
            return result
        
        return {}


    def _parse_generated_concepts(self, generated_concepts):
        """Extract a list of concept strings from the get_concept tool output."""
        if isinstance(generated_concepts, list):
            # Could be [dict, filename] tuple serialized as list
            if len(generated_concepts) >= 1 and isinstance(generated_concepts[0], dict):
                return list(generated_concepts[0].keys())
            # Could be list of strings
            result = []
            for item in generated_concepts:
                if isinstance(item, dict):
                    result.extend(item.keys())
                elif isinstance(item, str):
                    # Try parsing as JSON
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, dict):
                            result.extend(parsed.keys())
                            continue
                    except (json.JSONDecodeError, TypeError):
                        pass
                    # Skip filenames like 'savings/concepts_output.json'
                    if '/' in item or item.endswith('.json') or item.endswith('.csv'):
                        continue
                    result.append(item)
            return result if result else [str(x) for x in generated_concepts]
        
        if isinstance(generated_concepts, dict):
            return list(generated_concepts.keys())
        
        if isinstance(generated_concepts, str):
            # Try JSON parse — could be a JSON dict or a serialized [dict, string] tuple
            try:
                data = json.loads(generated_concepts)
                if isinstance(data, dict):
                    return list(data.keys())
                if isinstance(data, list):
                    # [dict, filename] format
                    if len(data) >= 1 and isinstance(data[0], dict):
                        return list(data[0].keys())
                    # [json_string, filename] format  
                    if len(data) >= 1 and isinstance(data[0], str):
                        try:
                            inner = json.loads(data[0])
                            if isinstance(inner, dict):
                                return list(inner.keys())
                        except (json.JSONDecodeError, TypeError):
                            pass
                    return [str(x) for x in data if not (isinstance(x, str) and (x.endswith('.json') or '/' in x))]
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Try extracting ALL dicts from string (handles tuple repr, multiple dicts, etc.)
            try:
                import regex as re
                # Find the FIRST top-level dict which is typically the concepts dict
                obj_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', generated_concepts, re.DOTALL)
                if obj_match:
                    try:
                        d = json.loads(obj_match.group())
                        if isinstance(d, dict):
                            return list(d.keys())
                    except json.JSONDecodeError:
                        # Try with single quotes replaced
                        try:
                            fixed = obj_match.group().replace("'", '"')
                            d = json.loads(fixed)
                            if isinstance(d, dict):
                                return list(d.keys())
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass
            
            # Fallback: comma-separated
            return [c.strip() for c in generated_concepts.split(',') if c.strip() and not c.strip().endswith('.json')]
        
        return []


    def _convert_numpy(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj



    def compare_concepts(self, vision_concepts, generated_report_concepts, similarity_threshold=SIMILARITY_THRESHOLD):
        """Compare vision concepts against generated report concepts using SapBERT similarity."""
        print("============================================Entered compare Concepts===============================================")
        # Parse inputs
        vision_dict = self._parse_vision_concepts_to_dict(vision_concepts)
        generated_list = self._parse_generated_concepts(generated_report_concepts)
        
        if not vision_dict:
            return 0, []
        
        if not generated_list:
            # All vision concepts are missing
            all_concepts = list(vision_dict.keys())
            return len(all_concepts), all_concepts

        query_concepts = []
        for key, value in vision_dict.items():
            if value == 'present':
                query_concepts.append(key)
            else:
                query_concepts.append('no ' + str(key))

        # Pre-compute embeddings for all generated concepts
        generated_embeddings = {}
        for gc in generated_list:
            toks = self.tokenizer._encode_plus(str(gc), padding="max_length", max_length=MAX_TOKEN_LENGTH, truncation=True, return_tensors="pt")
            token_count = len(
                self.tokenizer.encode(
                    str(gc),
                    add_special_tokens=True
                )
            )

            if token_count > MAX_CONCEPT_TOKENS:
                print("Skipping oversized concept:")
                print(gc[:500])
                continue
            output = self.model(**toks)
            generated_embeddings[gc] = output[0][:,0,:].cpu().detach().numpy()

        missing_concepts = []
        matched_concepts = []
        comparison_details = []

        for query1 in query_concepts:
            query1_toks = self.tokenizer._encode_plus([query1], padding="max_length", max_length=MAX_TOKEN_LENGTH, truncation=True, return_tensors="pt")
            query1_output = self.model(**query1_toks)
            query1_cls_rep = query1_output[0][:,0,:].cpu().detach().numpy()

            pair_scores = []
            best_score = -1.0
            best_match = None

            for query2 in generated_list:
                query2_cls_rep = generated_embeddings[query2]
                sim = cosine_similarity(query1_cls_rep, query2_cls_rep)[0][0]
                pair_scores.append({"generated_concept": query2, "similarity": round(float(sim), 4)})
                if sim > best_score:
                    best_score = sim
                    best_match = query2

            found = best_score >= similarity_threshold

            detail = {
                "vision_concept": query1,
                "matched": found,
                "best_match": best_match,
                "best_score": round(float(best_score), 4),
                "threshold": similarity_threshold,
                "top5_pairs": sorted(pair_scores, key=lambda x: x["similarity"], reverse=True)[:TOP_K_PAIRS_LOGGED],
            }
            comparison_details.append(detail)

            if found:
                matched_concepts.append({"vision": query1, "matched_to": best_match, "score": round(float(best_score), 4)})
            else:
                missing_concepts.append(query1)

        # Save detailed comparison log
        comparison_log = self._convert_numpy({
            "threshold": similarity_threshold,
            "num_vision_concepts": len(query_concepts),
            "num_generated_concepts": len(generated_list),
            "num_matched": len(matched_concepts),
            "num_missing": len(missing_concepts),
            "matched": matched_concepts,
            "missing": missing_concepts,
            "generated_concepts": generated_list,
            "details": comparison_details,
        })
        savings_dir = os.path.join(os.getcwd(), SAVINGS_DIR_NAME)
        os.makedirs(savings_dir, exist_ok=True)
        with open(os.path.join(savings_dir, COMPARISON_LOG_FILE), 'w') as f:
            json.dump(comparison_log, f, indent=4)

        if len(missing_concepts) > 0:
            llm_out = {
                'missing_concepts': missing_concepts,
                'vision_concepts': vision_dict if isinstance(vision_concepts, dict) else str(vision_concepts),
                'generated_report_concepts': generated_list
            }
            with open(os.path.join(savings_dir, MISSING_FILE), 'w') as f:
                json.dump(llm_out, f, indent=4)

        return missing_concepts, comparison_log