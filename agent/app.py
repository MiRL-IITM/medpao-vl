"""Gradio demo for MedPAO-VL: upload a chest X-ray, watch the agent run, and trace every sentence of the structured report back to its sources.

Run from this folder:

    pip install "gradio>=4.44,<5" "pydantic<2.11" rapidfuzz opencv-python-headless   # Python 3.9 (Gradio 4.44 breaks with pydantic >= 2.11)
    # Python >= 3.10: pip install gradio rapidfuzz opencv-python-headless
    python app.py                       # http://127.0.0.1:7860
    python app.py --share               # temporary public link

    # load the LLMs in this process instead of calling the HTTP servers (all models are loaded at start-up):
    python app.py --vlm-model <path> [--vlm-adapter <lora>] [--text-model <base>] \\
                  --extractor-adapter <lora> --categorizer-adapter <lora>

Traceability ("Transformation Studio")
--------------------------------------
Every sentence of the structured (ABCDEF) report is linked to
    the sentence of the free-text report it came from  ->  the concept that produced it
    ->  that concept's SNOMED CT mapping  ->  the vision-model attention map of that concept.
Hover a sentence on EITHER side: its counterpart lights up (same colour and link number), everything else dims, and
hovering a structured sentence opens a drawer with the whole chain. Pure CSS (:has), no extra model calls.
The links are fuzzy matches (rapidfuzz), shown with their match confidence; they are a reading aid, not ground truth.

The report stages need the same LLM servers / models and index files as the batch scripts (see README.md).
Research use only: not a medical device, not validated for clinical use.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import queue
import re
import threading
import time
import traceback

import gradio as gr
from rapidfuzz import fuzz, process

import inference as mp
from tools import llm_backend
from inference import (DEFAULT_MODE, MIN_CONCEPT_PROB, VISION_MODEL, attention_overlay, final_report_text)

# ───────────────────────────── CONFIG (edit here) ─────────────────────────────
APP_TITLE   = "MedPAO-VL"
SERVER_NAME = "127.0.0.1"          # use 0.0.0.0 to serve on the local network
SERVER_PORT = 7860
MAX_OVERLAYS = 12                  # attention maps listed in the "Attention maps" tab
TRACE_SCORE_THRESHOLD = 55.0       # min rapidfuzz score (0-100) to link a structured sentence to a concept
SOURCE_SCORE_THRESHOLD = 70.0      # min score to locate a concept's source text in the free-text report
VISION_MATCH_THRESHOLD = 75.0      # min score to link a report concept to a vision-model concept
THUMB_WIDTH = 260                  # width (px) of the attention thumbnail in the hover drawer
MAX_LINKS = 40                     # CSS rules are generated for this many free-text sentences
# ──────────────────────────────────────────────────────────────────────────────

DISCLAIMER = ("Research use only. This demo is not a medical device and has not been validated for clinical use. "
              "Do not use its output for diagnosis or patient management.")
MODE_HELP = {
    "deterministic": "Single pass: grounded report → concepts → SNOMED CT → ABCDEF structured report.",
    "cold": "Plan-Act-Observe loop (self-correction), no concept-cache lookup.",
    "warm": "Plan-Act-Observe loop with the concept cache (skips ontology mapping for known concepts).",
}

SECTION_META = {
    "P": ("Projection",  "📐", "#4A9EFF"),
    "A": ("Airways",     "🫧", "#A78BFA"),
    "B": ("Breathing",   "🫁", "#34D399"),
    "C": ("Circulation", "❤️",  "#F87171"),
    "D": ("Diaphragm",   "⬇️",  "#FBBF24"),
    "E": ("External",    "🦴", "#FB923C"),
    "F": ("Foreign",     "🔩", "#60A5FA"),
}

# step key (see inference._instrument) -> label; the steps shown depend on the agent mode
STEP_LABELS = {
    "vision":     "Vision Concepts",
    "grounded":   "Grounded Report",
    "concepts":   "Concept Extraction",
    "correction": "Self-correction",
    "cache":      "Cache Lookup",
    "ontology":   "Ontology Mapping",
    "categorize": "ABCDEF Classification",
    "structured": "Report Generation",
}
MODE_STEPS = {
    "deterministic": ["vision", "grounded", "concepts", "ontology", "categorize", "structured"],
    "cold":          ["vision", "grounded", "concepts", "correction", "ontology", "categorize", "structured"],
    "warm":          ["vision", "grounded", "concepts", "correction", "cache", "ontology", "categorize", "structured"],
}
STEP_STATUS = {
    "vision": "Predicting concepts…", "grounded": "Writing the grounded report…",
    "concepts": "Extracting concepts…", "correction": "Self-correcting missing concepts…",
    "cache": "Checking the concept cache…", "ontology": "Mapping to SNOMED CT…",
    "categorize": "Classifying ABCDEF…", "structured": "Generating the structured report…",
}
LINK_COLORS = ["#38BDF8", "#F472B6", "#34D399", "#FBBF24", "#A78BFA", "#FB923C", "#2DD4BF", "#F87171"]

_predictor = None
_lock = threading.Lock()   # the agent scripts share one scratch folder, so requests run one at a time


def _get_predictor(model_path: str):
    global _predictor
    if _predictor is None:
        _predictor = mp.VisionConceptPredictor(model_path)
    return _predictor


# ── small helpers ────────────────────────────────────────────────────────────
def _esc(s) -> str:
    return "" if s is None else html.escape(str(s), quote=True)


def _fmt(s: float) -> str:
    return f"{s:.2f}s"


def _link_color(i: int) -> str:
    return LINK_COLORS[i % len(LINK_COLORS)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _split_sentences(text: str) -> list:
    text = (text or "").strip()
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]


def _thumb_uri(img) -> str:
    img = img.copy()
    img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 2))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ── traceability ─────────────────────────────────────────────────────────────
def _locate_source(source: str, src_sents: list) -> list:
    """Indices of the free-text report sentences that a concept's source text covers."""
    if not source or not src_sents:
        return []
    ns = _norm(source)
    exact = [i for i, s in enumerate(src_sents) if _norm(s) and (_norm(s) in ns or ns in _norm(s))]
    if exact:
        return [i for i in exact if i < MAX_LINKS]
    res = process.extractOne(ns, [_norm(s) for s in src_sents], scorer=fuzz.partial_ratio)
    if res is not None and res[1] >= SOURCE_SCORE_THRESHOLD and res[2] < MAX_LINKS:
        return [res[2]]
    return []


def _match_vision_concept(concept: str, prediction, min_prob: float):
    """Link a concept extracted from the report to a concept of the vision model: (index, score) or None.
    Concepts the vision model reported (present/absent above the threshold) are preferred."""
    clean = re.sub(r"^(no|without|absence of|negative for|not)\s+", "", _norm(concept))
    if not clean:
        return None
    names = prediction.names
    chosen = [n for n, _ in prediction.present(min_prob)] + [n for n, _ in prediction.absent(min_prob)]
    for pool, threshold in ((chosen, VISION_MATCH_THRESHOLD), (names, VISION_MATCH_THRESHOLD + 10)):
        if not pool:
            continue
        res = process.extractOne(clean, [_norm(n) for n in pool], scorer=fuzz.WRatio)
        if res is not None and res[1] >= threshold:
            return names.index(pool[res[2]]), float(res[1])
    return None


def build_trace(acc: dict, prediction, image, min_prob: float) -> dict:
    """{section: [{"sentence", "trace": None | {concept, source, ontology, category, score, src_idx, vision}}]}"""
    report = acc.get("structured_report") or {}
    if not isinstance(report, dict) or not report:
        return {}
    concepts = acc.get("concepts") or {}
    categories = acc.get("categories") or {}
    onto = acc.get("ontology") or {}
    src_sents = _split_sentences(acc.get("report", ""))
    overlay_cache = {}

    def vision_info(concept):
        if prediction is None or image is None:
            return None
        hit = _match_vision_concept(concept, prediction, min_prob)
        if hit is None:
            return None
        idx, score = hit
        if idx not in overlay_cache:
            overlay_cache[idx] = _thumb_uri(attention_overlay(image, prediction.attn[idx]))
        probs = prediction.probs[idx]
        cls = int(probs.argmax())
        return {"name": prediction.names[idx], "uri": overlay_cache[idx], "score": round(score, 1),
                "label": ("present", "uncertain", "absent")[cls], "prob": float(probs[cls])}

    trace = {}
    for section, text in report.items():
        in_section = [c for c, cat in categories.items() if cat == section and c in concepts]
        pool = in_section or list(concepts)      # cached concepts have no category in this run: fall back to all
        rows = []
        for sent in _split_sentences(str(text)):
            best = None
            for concept in pool:
                source = concepts.get(concept, "")
                score = fuzz.token_set_ratio(sent, concept)
                if source:
                    score = max(score, 0.9 * fuzz.partial_ratio(_norm(sent), _norm(source)))
                if best is None or score > best[1]:
                    best = (concept, score)
            info = None
            if best and best[1] >= TRACE_SCORE_THRESHOLD:
                concept, score = best
                source = concepts.get(concept, "")
                info = {
                    "concept": concept, "source": source, "score": round(float(score), 1),
                    "ontology": onto.get(concept, ""),
                    "category": categories.get(concept, section),
                    "src_idx": _locate_source(source, src_sents),
                    "vision": vision_info(concept),
                }
            rows.append({"sentence": sent, "trace": info})
        trace[section] = rows
    return trace


# ── HTML builders ────────────────────────────────────────────────────────────
def _pipeline_html(steps: list, active=None, done_steps=(), times=None, total=None, all_done=False) -> str:
    times = times or {}
    n = len(steps)
    completed = n if all_done else len([s for s in steps if s in done_steps])
    items = ""
    for i, key in enumerate(steps):
        if all_done or key in done_steps:
            state, icon = "step-done", "✓"
        elif key == active:
            state, icon = "step-active", f"{i + 1:02d}"
        else:
            state, icon = "step-pending", f"{i + 1:02d}"
        badge = (f'<div class="step-time">{_fmt(times[key])}</div>' if key in times and state == "step-done"
                 else '<div class="step-time">&nbsp;</div>')
        conn = ""
        if i < n - 1:
            conn = f'<div class="connector {"connector-done" if state == "step-done" else ""}"></div>'
        items += (f'<div class="pipeline-item"><div class="step-bubble {state}">{icon}</div>'
                  f'<div class="step-label {state}">{STEP_LABELS[key]}</div>{badge}</div>{conn}')
    total_row = ""
    if total is not None:
        total_row = (f'<div class="pipeline-total"><span class="total-label">⏱ Total pipeline time</span>'
                     f'<span class="total-value">{_fmt(total)}</span></div>')
    pct = int(round(100 * completed / n)) if n else 0
    return (f'<div class="pipeline-wrap"><div class="pipeline-progress"><div class="pipeline-progress-fill" '
            f'style="width:{pct}%;"></div></div><div class="pipeline-track">{items}</div>{total_row}</div>')


def _status_html(kind: str, msg: str) -> str:
    return f'<div class="status-bar status-{kind}"><div class="status-dot"></div>{msg}</div>'


def _vision_html(prediction, min_prob: float) -> str:
    if prediction is None:
        return '<p class="empty-msg">Vision concepts appear here once the image is analysed.</p>'

    def chips(items, color):
        if not items:
            return '<span class="empty-inline">none above the threshold</span>'
        return "".join(f'<span class="cat-chip" style="border-color:{color};color:{color};">{_esc(n)} · {p:.2f}</span>'
                       for n, p in items)
    return (f'<div class="cat-list"><div class="cat-row"><div class="cat-badge" style="background:#34D39922;color:#34D399;'
            f'border:1px solid #34D39955;">present</div><div class="cat-chips">{chips(prediction.present(min_prob), "#34D399")}</div></div>'
            f'<div class="cat-row"><div class="cat-badge" style="background:#F8717122;color:#F87171;border:1px solid #F8717155;">'
            f'absent</div><div class="cat-chips">{chips(prediction.absent(min_prob), "#F87171")}</div></div></div>')


def _concepts_html(concepts: dict) -> str:
    if not concepts:
        return '<p class="empty-msg">No concepts extracted yet.</p>'
    rows = "".join(f'<div class="concept-row"><span class="concept-tag">{_esc(c)}</span>'
                   f'<span class="concept-src">↳ {_esc(s)}</span></div>' for c, s in concepts.items())
    return f'<div class="concepts-grid">{rows}</div>'


def _ontology_html(onto: dict) -> str:
    if not onto:
        return '<p class="empty-msg">No ontology data.</p>'
    rows = "".join(f'<div class="onto-row"><div class="onto-concept">{_esc(c)}</div>'
                   f'<div class="onto-info">{_esc(info)}</div></div>' for c, info in onto.items())
    return f'<div class="onto-list">{rows}</div>'


def _categories_html(cats: dict) -> str:
    if not cats:
        return '<p class="empty-msg">No categories yet.</p>'
    buckets: dict = {}
    for concept, cat in cats.items():
        buckets.setdefault(cat, []).append(concept)
    rows = ""
    for cat in sorted(buckets):
        label, icon, color = SECTION_META.get(cat, (cat, "📋", "#94A3B8"))
        chips = "".join(f'<span class="cat-chip" style="border-color:{color};color:{color};">{_esc(c)}</span>'
                        for c in buckets[cat])
        rows += (f'<div class="cat-row"><div class="cat-badge" style="background:{color}22;color:{color};'
                 f'border:1px solid {color}55;">{icon} {_esc(cat)} · {_esc(label)}</div><div class="cat-chips">{chips}</div></div>')
    return f'<div class="cat-list">{rows}</div>'


def _badges(idxs: list) -> str:
    return "".join(f'<sup class="lk" style="--lc:{_link_color(i)};">{i + 1}</sup>' for i in idxs)


def _studio_html(source_text: str, report, trace: dict = None) -> str:
    """Side by side: free-text report (left) ↔ structured report (right); the hover drawer shows the whole chain."""
    trace = trace or {}
    src_sents = _split_sentences(source_text)
    linked_src, n_out, n_linked = set(), 0, 0
    for rows in trace.values():
        for r in rows:
            n_out += 1
            if r["trace"]:
                n_linked += 1
                linked_src.update(r["trace"]["src_idx"])

    if src_sents:
        parts = []
        for i, s in enumerate(src_sents):
            if i in linked_src:
                parts.append(f'<span class="ss l{i}" style="--lc:{_link_color(i)};"><sup class="lk">{i + 1}</sup>{_esc(s)}</span>')
            else:
                parts.append(f'<span class="su">{_esc(s)}</span>')
        src_body = " ".join(parts)
    else:
        src_body = '<span class="empty-inline">The free-text report appears here once it is generated.</span>'

    if not isinstance(report, dict) or not report:
        out_body = '<p class="empty-msg">Structured report not generated yet.</p>'
    else:
        cards = ""
        for key, text in report.items():
            label, icon, color = SECTION_META.get(key, (key, "📋", "#94A3B8"))
            dim = " dim" if str(text).strip().lower().rstrip(".") in ("no findings", "no findings identified") else ""
            rows = trace.get(key)
            if rows:
                pieces = []
                for row in rows:
                    sent, t = _esc(row["sentence"]), row["trace"]
                    if not t:
                        pieces.append(f'<div class="tu">{sent}</div>')
                        continue
                    idxs = t["src_idx"]
                    cls = " ".join(f"l{i}" for i in idxs)
                    lc = _link_color(idxs[0]) if idxs else color
                    cat_letter = t["category"]
                    cat_label = SECTION_META.get(cat_letter, (cat_letter, "", ""))[0]
                    v = t["vision"]
                    vision_row = (f'<div class="dr-row"><span class="dr-k">Attention</span><span class="dr-v">'
                                  f'<img class="dr-img" src="{v["uri"]}" alt="attention map"/>'
                                  f'<span class="dr-cap">{_esc(v["name"])} · {v["label"]} ({v["prob"]:.2f})</span></span></div>'
                                  if v else '<div class="dr-row"><span class="dr-k">Attention</span><span class="dr-v dr-none">'
                                            'no matching vision-model concept</span></div>')
                    source_row = (f'<div class="dr-row"><span class="dr-k">Source</span><span class="dr-v">“{_esc(t["source"])}”</span></div>'
                                  if t["source"] else "")
                    pieces.append(f"""
                    <div class="ts {cls}" style="--lc:{lc};">
                        <div class="ts-main">{_badges(idxs)}<span class="ts-text">{sent}</span></div>
                        <div class="ts-drawer"><div class="ts-drawer-in">
                            {source_row}
                            <div class="dr-row"><span class="dr-k">Concept</span><span class="dr-v dr-concept">{_esc(t['concept'])}</span></div>
                            <div class="dr-row"><span class="dr-k">SNOMED CT</span><span class="dr-v dr-onto">{_esc(t['ontology']) or '—'}</span></div>
                            {vision_row}
                            <div class="dr-foot">
                                <span class="dr-pill" style="color:{color};border-color:{color}66;">{_esc(cat_letter)} · {_esc(cat_label)}</span>
                                <span class="dr-conf"><span class="dr-bar"><span style="width:{min(100, t['score'])}%;"></span></span>{t['score']}%</span>
                            </div>
                        </div></div>
                    </div>""")
                body = "".join(pieces)
            else:
                body = f'<div class="tu">{_esc(text)}</div>'
            cards += f"""
            <div class="report-card{dim}">
                <div class="card-header" style="border-left:3px solid {color};">
                    <span class="card-icon">{icon}</span><span class="card-key" style="color:{color};">{_esc(key)}</span>
                    <span class="card-label">{label}</span>
                </div>
                <div class="card-text">{body}</div>
            </div>"""
        out_body = f'<div class="report-grid">{cards}</div>'

    stats = ""
    if trace:
        stats = (f'<div class="studio-stats"><div class="stat"><b>{n_out}</b><span>Output sentences</span></div>'
                 f'<div class="stat"><b>{n_linked}</b><span>Source-linked</span></div>'
                 f'<div class="stat"><b>{len(linked_src)}</b><span>Source spans used</span></div></div>')
    return f"""
    <div class="studio">{stats}
        <div class="studio-cols">
            <div class="studio-panel"><div class="studio-title"><span class="studio-num">1</span>Free-text report</div>
                <div class="src-body">{src_body}</div></div>
            <div class="studio-link" aria-hidden="true"><span></span></div>
            <div class="studio-panel"><div class="studio-title"><span class="studio-num">2</span>Structured report</div>
                <div class="out-body">{out_body}</div></div>
        </div>
    </div>"""


def _link_css() -> str:
    rules = []
    for i in range(MAX_LINKS):
        rules.append(f"""
.studio:has(.ts.l{i}:hover) .ss.l{i}, .studio:has(.ss.l{i}:hover) .ss.l{i} {{
    background:color-mix(in srgb, var(--lc) 30%, transparent);
    box-shadow:0 0 0 1px var(--lc), 0 0 22px color-mix(in srgb, var(--lc) 45%, transparent); color:#fff !important; }}
.studio:has(.ts.l{i}:hover) .ss.l{i} .lk, .studio:has(.ss.l{i}:hover) .ss.l{i} .lk {{ background:var(--lc); color:#06101c !important; }}
.studio:has(.ss.l{i}:hover) .ts.l{i}, .studio:has(.ts.l{i}:hover) .ts.l{i} {{
    background:color-mix(in srgb, var(--lc) 16%, transparent); border-color:var(--lc);
    box-shadow:0 0 0 1px color-mix(in srgb, var(--lc) 50%, transparent); }}
.studio:has(.ss.l{i}:hover) .ts.l{i} .lk {{ background:var(--lc); color:#06101c !important; }}
.studio:has(.ts.l{i}:hover) .ss:not(.l{i}), .studio:has(.ss.l{i}:hover) .ss:not(.l{i}),
.studio:has(.ts.l{i}:hover) .su, .studio:has(.ss.l{i}:hover) .su {{ opacity:.28; }}
.studio:has(.ss.l{i}:hover) .ts:not(.l{i}), .studio:has(.ss.l{i}:hover) .tu,
.studio:has(.ts.l{i}:hover) .ts:not(.l{i}), .studio:has(.ts.l{i}:hover) .tu {{ opacity:.3; }}""")
    return "\n".join(rules)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@400;500&family=Sora:wght@300;400;500;600;700&display=swap');
:root { --bg:#0B0F1A; --surface:#111827; --surface2:#1A2235; --surface3:#202B45; --border:#243554; --text:#E2E8F0;
    --text-mid:#94A3B8; --accent:#38BDF8; --accent2:#F59E0B; --danger:#F87171; --success:#34D399;
    --fh:'DM Serif Display',serif; --fb:'Sora',sans-serif; --fm:'IBM Plex Mono',monospace; --r:10px; --r-lg:16px;
    --shadow:0 8px 28px rgba(0,0,0,.35); }
body, .gradio-container, .gradio-container *, .prose, fieldset, legend, .wrap, .label-wrap {
    background-color:transparent; color:var(--text) !important; font-family:var(--fb) !important; }
body, .gradio-container { background: radial-gradient(circle at 15% -10%, #16213a 0%, transparent 45%),
    radial-gradient(circle at 100% 0%, #14203a 0%, transparent 35%), var(--bg) !important; }
.app-header { display:flex; align-items:center; gap:18px; margin-bottom:28px; padding-bottom:22px; position:relative; }
.app-header::after { content:''; position:absolute; left:0; right:0; bottom:0; height:1px;
    background:linear-gradient(90deg, var(--accent) 0%, var(--border) 35%, transparent 100%); }
.header-logo { width:54px; height:54px; background:linear-gradient(135deg,#0EA5E9,#6366F1); border-radius:16px;
    display:flex; align-items:center; justify-content:center; font-size:27px; flex-shrink:0;
    box-shadow:0 0 32px #0EA5E950, inset 0 1px 0 rgba(255,255,255,.15); }
.header-text h1 { font-family:var(--fh) !important; font-size:2.1rem; color:#F8FAFC !important; letter-spacing:-0.5px; line-height:1; }
.header-text p { font-size:.74rem; color:var(--text-mid) !important; margin-top:6px; font-family:var(--fm) !important; letter-spacing:.06em; }
.header-badge { margin-left:auto; padding:7px 16px; border:1px solid #0EA5E950; border-radius:20px; font-size:.72rem;
    font-family:var(--fm) !important; color:var(--accent) !important; background:linear-gradient(135deg,#0EA5E914,#6366F114); white-space:nowrap; }
.disclaimer { font-size:.74rem; color:var(--text-mid) !important; border:1px dashed var(--border); border-radius:var(--r);
    padding:9px 14px; margin-bottom:16px; }
textarea, input[type="text"], input[type="number"] { background:var(--surface2) !important; border:1px solid var(--border) !important;
    color:var(--text) !important; border-radius:var(--r) !important; font-family:var(--fm) !important; font-size:.82rem !important; }
button.primary, button[variant="primary"] { background:linear-gradient(135deg,#0284C7,#6366F1) !important; border:none !important;
    color:#fff !important; font-weight:600 !important; border-radius:var(--r) !important; box-shadow:0 4px 20px #6366F140 !important; }
button.secondary, button[variant="secondary"] { background:var(--surface2) !important; border:1px solid var(--border) !important;
    color:var(--text) !important; border-radius:var(--r) !important; }
button[role="tab"] { font-family:var(--fm) !important; font-size:.78rem !important; color:var(--text-mid) !important;
    background:transparent !important; border:none !important; border-bottom:2px solid transparent !important; }
button[role="tab"][aria-selected="true"] { color:var(--accent) !important; border-bottom-color:var(--accent) !important; }
.section-label { font-family:var(--fm); font-size:.68rem; letter-spacing:.14em; text-transform:uppercase;
    color:var(--text-mid) !important; margin-bottom:10px; display:flex; align-items:center; gap:8px; }
.section-label::after { content:''; flex:1; height:1px; background:var(--border); }

.pipeline-wrap { background:linear-gradient(180deg, var(--surface) 0%, #0E1524 100%); border:1px solid var(--border);
    border-radius:var(--r-lg); margin-bottom:14px; overflow:hidden; box-shadow:var(--shadow); }
.pipeline-progress { height:3px; background:var(--surface2); }
.pipeline-progress-fill { height:100%; background:linear-gradient(90deg, var(--accent), var(--success)); transition:width .4s ease; }
.pipeline-track { display:flex; align-items:flex-start; padding:22px 22px 14px; overflow-x:auto; }
.pipeline-item { display:flex; flex-direction:column; align-items:center; gap:6px; min-width:100px; }
.connector { flex:1; height:1px; background:var(--border); min-width:10px; margin-bottom:44px; margin-top:18px; }
.connector.connector-done { background:linear-gradient(90deg, var(--success), var(--accent)); height:2px; }
.step-bubble { width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-family:var(--fm); font-size:.72rem; font-weight:500; flex-shrink:0; }
.step-bubble.step-pending { background:var(--surface2); color:var(--text-mid) !important; border:1px solid var(--border); }
.step-bubble.step-active { background:#0EA5E920; color:var(--accent) !important; border:1px solid var(--accent); box-shadow:0 0 16px #38BDF850; animation:pulse 1.4s infinite; }
.step-bubble.step-done { background:#10B98120; color:var(--success) !important; border:1px solid var(--success); }
@keyframes pulse { 0%,100%{box-shadow:0 0 8px #38BDF840} 50%{box-shadow:0 0 24px #38BDF890} }
.step-label { font-size:.65rem; font-family:var(--fm); text-align:center; line-height:1.3; }
.step-label.step-pending { color:var(--text-mid) !important; }
.step-label.step-active { color:var(--accent) !important; font-weight:600; }
.step-label.step-done { color:var(--success) !important; }
.step-time { font-family:var(--fm); font-size:.62rem; color:var(--accent2) !important; text-align:center; min-height:14px; }
.pipeline-total { display:flex; align-items:center; justify-content:space-between; padding:10px 22px;
    border-top:1px solid var(--border); background:#0EA5E908; }
.total-label { font-family:var(--fm); font-size:.72rem; color:var(--text-mid) !important; }
.total-value { font-family:var(--fm); font-size:.92rem; font-weight:600; color:var(--accent2) !important; }
.status-bar { padding:11px 18px; border-radius:var(--r); font-family:var(--fm); font-size:.78rem; margin-bottom:20px;
    display:flex; align-items:center; gap:9px; border:1px solid transparent; }
.status-idle { background:#1E293B; color:var(--text-mid) !important; border-color:var(--border); }
.status-running { background:#0EA5E915; color:var(--accent) !important; border-color:#0EA5E930; }
.status-done { background:#10B98115; color:var(--success) !important; border-color:#10B98130; }
.status-error { background:#F8717115; color:var(--danger) !important; border-color:#F8717130; }
.status-bar strong { color:var(--accent2) !important; }
.status-dot { width:7px; height:7px; border-radius:50%; background:currentColor; flex-shrink:0; }
.status-running .status-dot { animation:blink 1.1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

.hint-banner { display:flex; align-items:flex-start; gap:10px; background:linear-gradient(135deg,#0EA5E912,#6366F112);
    border:1px dashed #0EA5E955; border-radius:var(--r); padding:11px 15px; margin-bottom:14px; font-size:.78rem;
    color:var(--text-mid) !important; line-height:1.55; }
.hint-banner b { color:var(--accent) !important; font-weight:600; }
.concepts-grid { display:flex; flex-direction:column; gap:8px; }
.concept-row { background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:10px 14px;
    display:flex; flex-direction:column; gap:5px; }
.concept-tag { font-family:var(--fm); font-size:.8rem; color:var(--accent) !important; font-weight:500; }
.concept-src { font-size:.77rem; line-height:1.5; }

.studio { background:linear-gradient(180deg, var(--surface) 0%, #0E1524 100%); border:1px solid var(--border);
    border-radius:var(--r-lg); padding:18px; box-shadow:var(--shadow); }
.studio-stats { display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap; }
.stat { flex:1; min-width:130px; display:flex; align-items:baseline; gap:9px; padding:10px 14px;
    border:1px solid var(--border); border-radius:var(--r); background:var(--surface2); }
.stat b { font-family:var(--fh) !important; font-size:1.5rem; font-weight:400; color:#F8FAFC !important; }
.stat span { font-size:.68rem; color:var(--text-mid) !important; font-family:var(--fm); }
.studio-cols { display:flex; align-items:stretch; }
.studio-panel { flex:1 1 0; min-width:0; background:#0A1120; border:1px solid var(--border); border-radius:14px; padding:16px 18px 18px; }
.studio-link { width:34px; flex-shrink:0; display:flex; align-items:center; justify-content:center; }
.studio-link span { width:100%; height:2px; border-radius:2px; opacity:.55;
    background:repeating-linear-gradient(90deg, var(--accent) 0 5px, transparent 5px 9px); }
.studio-title { display:flex; align-items:center; gap:9px; margin-bottom:14px; font-size:.84rem; font-weight:600; color:#F8FAFC !important; }
.studio-num { width:22px; height:22px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
    font-family:var(--fm); font-size:.68rem; color:var(--accent) !important; border:1px solid var(--accent); background:#38BDF812; }
.src-body { font-family:var(--fm); font-size:.8rem; line-height:2.15; max-height:760px; overflow-y:auto; padding-right:6px; }
.empty-inline { color:var(--text-mid) !important; font-family:var(--fm); font-size:.78rem; }
.ss, .su { transition:opacity .2s, background .2s, box-shadow .2s; }
.ss { --lc:#38BDF8; position:relative; cursor:pointer; padding:3px 6px; margin:0 1px; border-radius:6px;
    background:color-mix(in srgb, var(--lc) 9%, transparent); box-decoration-break:clone; -webkit-box-decoration-break:clone;
    border-bottom:2px solid color-mix(in srgb, var(--lc) 55%, transparent); }
.lk { display:inline-flex; align-items:center; justify-content:center; min-width:15px; height:15px; padding:0 4px; margin-right:6px;
    border-radius:8px; vertical-align:middle; position:relative; top:-1px; font-family:var(--fm); font-size:.6rem; font-weight:600;
    line-height:1; background:color-mix(in srgb, var(--lc) 25%, transparent); color:var(--lc) !important;
    border:1px solid color-mix(in srgb, var(--lc) 60%, transparent); transition:background .2s, color .2s; }
.report-grid { display:flex; flex-direction:column; gap:12px; }
.report-card { background:linear-gradient(180deg, var(--surface2) 0%, #17203450 100%); border:1px solid var(--border);
    border-radius:12px; padding:14px 16px; }
.report-card.dim { opacity:.4; }
.card-header { display:flex; align-items:center; gap:10px; padding-left:12px; margin-bottom:10px; }
.card-key { font-family:var(--fm); font-weight:700; font-size:.98rem; }
.card-label { font-size:.69rem; color:var(--text-mid) !important; font-family:var(--fm); margin-left:4px; }
.card-text { display:flex; flex-direction:column; gap:6px; }
.tu { font-size:.84rem; line-height:1.65; padding:5px 10px 5px 12px; transition:opacity .2s; }
.ts { --lc:#38BDF8; position:relative; cursor:pointer; border:1px solid transparent; border-left:3px solid var(--lc);
    border-radius:9px; padding:7px 10px 7px 11px; background:color-mix(in srgb, var(--lc) 5%, transparent);
    transition:opacity .2s, background .2s, border-color .2s, box-shadow .2s; }
.ts-main { font-size:.84rem; line-height:1.65; }
.ts-drawer { display:grid; grid-template-rows:0fr; opacity:0; transition:grid-template-rows .22s ease, opacity .18s ease, margin-top .22s ease; }
.ts-drawer-in { overflow:hidden; min-height:0; border-top:1px dashed color-mix(in srgb, var(--lc) 45%, transparent); }
.ts:hover .ts-drawer { grid-template-rows:1fr; opacity:1; margin-top:9px; }
.ts:hover .ts-drawer-in { padding-top:9px; }
.dr-row { display:flex; gap:12px; margin-bottom:7px; align-items:flex-start; }
.dr-k { flex:0 0 78px; font-family:var(--fm); font-size:.64rem; color:var(--text-mid) !important; padding-top:2px; }
.dr-v { font-size:.78rem; line-height:1.5; white-space:pre-wrap; display:flex; flex-direction:column; gap:4px; }
.dr-concept { font-family:var(--fm); color:var(--lc) !important; font-weight:500; }
.dr-onto { font-family:var(--fm); font-size:.73rem; color:#CBD5E1 !important; }
.dr-none { color:var(--text-mid) !important; font-style:italic; }
.dr-img { width:100%; max-width:260px; border-radius:8px; border:1px solid var(--border); display:block; }
.dr-cap { font-family:var(--fm); font-size:.66rem; color:var(--text-mid) !important; }
.dr-foot { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-top:4px; }
.dr-pill { font-family:var(--fm); font-size:.66rem; padding:2px 10px; border:1px solid; border-radius:20px; }
.dr-conf { display:flex; align-items:center; gap:8px; font-family:var(--fm); font-size:.66rem; color:var(--text-mid) !important; }
.dr-bar { width:70px; height:4px; border-radius:3px; background:var(--surface3); overflow:hidden; display:inline-block; }
.dr-bar span { display:block; height:100%; border-radius:3px; background:linear-gradient(90deg, var(--accent), var(--success)); }
@media (max-width: 900px) { .studio-cols { flex-direction:column; } .studio-link { width:100%; height:26px; }
    .studio-link span { width:2px; height:100%; background:repeating-linear-gradient(180deg, var(--accent) 0 5px, transparent 5px 9px); } }
@media (prefers-reduced-motion: reduce) { .ss, .su, .ts, .tu, .ts-drawer { transition:none !important; } }

.onto-list { display:flex; flex-direction:column; gap:10px; }
.onto-row { background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:12px 16px; }
.onto-concept { font-family:var(--fm); font-size:.8rem; color:var(--accent2) !important; margin-bottom:6px; font-weight:500; }
.onto-info { font-size:.78rem; white-space:pre-wrap; line-height:1.55; }
.cat-list { display:flex; flex-direction:column; gap:10px; }
.cat-row { display:flex; align-items:flex-start; gap:12px; flex-wrap:wrap; }
.cat-badge { padding:4px 12px; border-radius:20px; font-family:var(--fm); font-size:.73rem; white-space:nowrap; flex-shrink:0; }
.cat-chips { display:flex; flex-wrap:wrap; gap:6px; }
.cat-chip { padding:3px 10px; border-radius:6px; border:1px solid; font-size:.72rem; font-family:var(--fm); }
.full-width-divider { border-top:1px solid var(--border); margin-top:28px; padding-top:24px; }
.empty-msg { color:var(--text-mid) !important; font-family:var(--fm); font-size:.8rem; padding:20px; text-align:center;
    border:1px dashed var(--border); border-radius:10px; }
::-webkit-scrollbar { width:5px; height:5px; } ::-webkit-scrollbar-track { background:var(--surface); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
""" + _link_css()


# ── processing ───────────────────────────────────────────────────────────────
def _empty_outputs(steps, status):
    return (_pipeline_html(steps), status, _vision_html(None, 0), _studio_html("", {}), _concepts_html({}),
            _ontology_html({}), _categories_html({}), [], "")


def analyze(image_path, mode, min_prob, concepts_only, model_path):
    """Generator: runs the pipeline in a worker thread and streams the UI state after every agent step."""
    steps = MODE_STEPS[mode] if not concepts_only else ["vision"]
    if not image_path:
        yield _empty_outputs(steps, _status_html("error", "Upload a chest X-ray first."))
        return

    events: "queue.Queue" = queue.Queue()

    def worker():
        with _lock:
            try:
                result = mp.run_inference(image_path, mode, _get_predictor(model_path), float(min_prob),
                                          MAX_OVERLAYS, bool(concepts_only), progress=events.put)
                events.put({"type": "result", "result": result})
            except Exception as e:   # show the failing step in the UI, keep the full trace in the console
                traceback.print_exc()
                events.put({"type": "error", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})

    t0 = time.time()
    threading.Thread(target=worker, daemon=True).start()

    acc = {"report": "", "concepts": {}, "ontology": {}, "categories": {}, "structured_report": {}}
    prediction = image = None
    overlays = []
    done, times, active = set(), {}, "vision"

    def render(trace_now=False, status=None, all_done=False, total=None, details=""):
        trace = build_trace(acc, prediction, image, float(min_prob)) if trace_now else {}
        return (_pipeline_html(steps, active=None if all_done else active, done_steps=done, times=times,
                               total=total, all_done=all_done),
                status or _status_html("running", STEP_STATUS.get(active, "Working…")),
                _vision_html(prediction, float(min_prob)),
                _studio_html(acc["report"], acc["structured_report"], trace),
                _concepts_html(acc["concepts"]), _ontology_html(acc["ontology"]), _categories_html(acc["categories"]),
                overlays, details)

    yield render()
    while True:
        ev = events.get()
        kind = ev["type"]
        if kind == "vision":
            r = ev["result"]
            prediction, image, overlays = r["prediction"], r["image"], r["overlays"]
            times["vision"] = r["timings"]["vision_time_sec"]
            done.add("vision")
            active = steps[1] if len(steps) > 1 else None
            if concepts_only:
                continue                       # the final "result" event renders the finished state
            yield render()
        elif kind == "start":
            active = ev["step"]
            yield render()
        elif kind == "end":
            step, res = ev["step"], ev["result"]
            times[step] = times.get(step, 0.0) + ev["elapsed"]
            done.add(step)
            if step in ("grounded", "correction") and isinstance(res, str):
                acc["report"] = res                         # a correction pass returns the cumulative report
            elif step in ("concepts", "correction") and isinstance(res, dict) and res:
                acc["concepts"] = res
            elif step == "ontology" and isinstance(res, list):
                for item in res:
                    if isinstance(item, dict):
                        desc = item.get("preferred_desc", "")
                        acc["ontology"][item.get("query", "")] = f'{desc} [{item.get("concept_id", "")}]'.strip()
            elif step == "categorize" and isinstance(res, dict):
                acc["categories"].update(res)
            elif step == "structured" and isinstance(res, dict):
                acc["structured_report"] = res
            yield render(trace_now=(step == "structured"))
        elif kind == "error":
            yield render(status=_status_html("error", f'Error: {_esc(ev["error"])}'), details=ev["trace"])
            return
        elif kind == "result":
            r = ev["result"]
            agent = r["agent"]
            total = time.time() - t0
            details = {"vision_concepts": r["vision_concepts"], "timings": r["timings"]}
            if agent:
                acc["report"] = final_report_text(agent)
                acc["concepts"] = agent.get("extracted_concepts") or acc["concepts"]
                acc["structured_report"] = agent.get("structured_report") or acc["structured_report"]
                for key in ("missing_concepts", "iterations", "correction_log", "cache_hit", "cache_miss"):
                    if key in agent:
                        details[key] = agent[key]
                details["agent_timings"] = agent.get("timings")
                details["grounded_report"] = agent.get("grounded_report")
                msg = _status_html("done", f"Pipeline complete &nbsp;·&nbsp; {len(acc['concepts'])} concepts &nbsp;·&nbsp; "
                                           f"total: <strong>{_fmt(total)}</strong>")
            else:
                msg = _status_html("done", f"Vision step complete &nbsp;·&nbsp; total: <strong>{_fmt(total)}</strong>")
            yield render(trace_now=bool(agent), status=msg, all_done=True, total=total,
                         details=json.dumps(details, indent=2, default=str))
            return


def clear_all():
    return (None,) + _empty_outputs(MODE_STEPS[DEFAULT_MODE], _status_html("idle", "Awaiting a chest X-ray."))


# ── layout ───────────────────────────────────────────────────────────────────
def build_demo(model_path: str = VISION_MODEL) -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE, css=CSS) as demo:
        gr.HTML(f"""
        <div class="app-header"><div class="header-logo">🫁</div>
            <div class="header-text"><h1>{_esc(APP_TITLE)}</h1>
                <p>CHEST X-RAY · CONCEPT BOTTLENECK · PLAN · ACT · OBSERVE · ABCDEF PROTOCOL</p></div>
            <div class="header-badge">MedSigLIP slots · MedGemma · SNOMED CT</div></div>
        <div class="disclaimer">{_esc(DISCLAIMER)}</div>""")

        pipeline = gr.HTML(_pipeline_html(MODE_STEPS[DEFAULT_MODE]))
        status = gr.HTML(_status_html("idle", "Awaiting a chest X-ray."))

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                gr.HTML('<div class="section-label">Chest X-ray</div>')
                image = gr.Image(type="filepath", show_label=False, height=380)
            with gr.Column(scale=2):
                gr.HTML('<div class="section-label">Settings</div>')
                mode = gr.Dropdown(list(mp.AGENT_MODULES), value=DEFAULT_MODE, label="Agent mode",
                                   info=MODE_HELP[DEFAULT_MODE])
                min_prob = gr.Slider(0.0, 0.95, value=MIN_CONCEPT_PROB, step=0.05, label="Minimum concept probability",
                                     info="Present/absent concepts below this confidence are dropped before report generation.")
                concepts_only = gr.Checkbox(False, label="Vision step only (no report; no LLMs needed)")
                with gr.Row():
                    run = gr.Button("▶  Generate report", variant="primary")
                    clear = gr.Button("✕  Clear", variant="secondary")
            with gr.Column(scale=3):
                gr.HTML('<div class="section-label">Vision model concepts</div>')
                vision = gr.HTML(_vision_html(None, 0))

        gr.HTML('<div class="full-width-divider"><div class="section-label">Transformation Studio</div></div>')
        gr.HTML('<div class="hint-banner"><span>🔗</span><span>Hover a sentence on <b>either side</b>. Its counterpart lights up '
                'in the same colour and link number and everything else dims. Hovering a structured sentence opens the whole '
                'chain: <b>free-text source → concept → SNOMED CT → attention map</b>, with the match confidence.</span></div>')
        studio = gr.HTML(_studio_html("", {}))

        gr.HTML('<div class="full-width-divider"><div class="section-label">Analysis Details</div></div>')
        with gr.Tabs():
            with gr.Tab("Concepts"):
                concepts = gr.HTML(_concepts_html({}))
            with gr.Tab("Ontology"):
                ontology = gr.HTML(_ontology_html({}))
            with gr.Tab("Categories"):
                categories = gr.HTML(_categories_html({}))
            with gr.Tab("Attention maps"):
                gallery = gr.Gallery(label="Most confident present concepts", columns=3, height="auto")
            with gr.Tab("Debug"):
                debug = gr.Code(language="json", label="", lines=18)

        outputs = [pipeline, status, vision, studio, concepts, ontology, categories, gallery, debug]
        mode.change(lambda m: (gr.update(info=MODE_HELP[m]), _pipeline_html(MODE_STEPS[m])), mode, [mode, pipeline],
                    api_name=False)
        run.click(lambda im, m, p, c: analyze(im, m, p, c, model_path), [image, mode, min_prob, concepts_only], outputs,
                  api_name="generate")
        clear.click(clear_all, None, [image] + outputs, api_name=False)
    return demo


def main(argv=None):
    p = argparse.ArgumentParser(description="MedPAO-VL Gradio demo (research use only).")
    p.add_argument("--vision-model", default=VISION_MODEL, help="local folder or Hub repo id of the vision model")
    p.add_argument("--host", default=SERVER_NAME)
    p.add_argument("--port", type=int, default=SERVER_PORT)
    p.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    p.add_argument("--no-preload", action="store_true", help="load the models on the first request instead of at start-up")
    llm_backend.add_cli_args(p)
    a = p.parse_args(argv)

    llm_backend.configure_from_args(a)
    if not a.no_preload:
        _get_predictor(a.vision_model)
        llm_backend.preload()      # local backend: load every LLM now (no-op for the server backend)
    demo = build_demo(a.vision_model)
    demo.queue(default_concurrency_limit=1).launch(server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
