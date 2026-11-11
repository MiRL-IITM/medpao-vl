"""Where the agent's LLM calls are answered: HTTP servers (default) or models loaded in this process.

server  The tools call the TGI servers configured at the top of each tool file (unchanged behaviour).
local   The tools call models loaded here from local paths / Hub ids. Three roles are needed:

        vlm        image + text -> text   grounded report and missing-concept lines
                   (a base model, a merged model, or a base model + LoRA adapter)
        text       ONE base LLM (TEXT_BASE_MODEL, Qwen3-4B-Instruct-2507) loaded once, shared by three steps:
          extractor    raw text completion    concept extraction        -> its own LoRA (EXTRACTOR_ADAPTER)
          categorizer  chat messages          concept categorization    -> its own LoRA (CATEGORIZER_ADAPTER)
          chat         chat messages          structured report         -> the plain base model (no adapter)

        The two LoRAs are attached to the same base weights and switched per call, so the memory cost is one 4B model.

Select the backend with command-line flags (inference.py, app.py), with `configure(...)`, or with environment
variables (also used by the batch scripts):

    MEDPAO_BACKEND=local  MEDPAO_VLM_MODEL=...  MEDPAO_VLM_ADAPTER=...  MEDPAO_TEXT_MODEL=...
    MEDPAO_EXTRACTOR_ADAPTER=...  MEDPAO_CATEGORIZER_ADAPTER=...

`preload()` loads every configured model up front, so the first request does not pay the loading time.
Heavy imports (torch, transformers, peft) happen only in local mode.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Sequence

# ───────────────────────────── CONFIG (edit here) ─────────────────────────────
BACKEND      = os.environ.get("MEDPAO_BACKEND", "server")        # "server" | "local"
TEXT_BASE_MODEL    = os.environ.get("MEDPAO_TEXT_MODEL", "Qwen/Qwen3-4B-Instruct-2507")   # shared base for all text steps
EXTRACTOR_ADAPTER  = os.environ.get("MEDPAO_EXTRACTOR_ADAPTER", "/path/to/extractor_lora")       # LoRA: concept extraction
CATEGORIZER_ADAPTER = os.environ.get("MEDPAO_CATEGORIZER_ADAPTER", "/path/to/categorizer_lora")  # LoRA: concept categorization
LOCAL_MODELS = {                                                  # local paths or Hugging Face ids
    "vlm":         os.environ.get("MEDPAO_VLM_MODEL"),
    "vlm_adapter": os.environ.get("MEDPAO_VLM_ADAPTER"),         # optional LoRA adapter for the vlm
}
DTYPE        = os.environ.get("MEDPAO_DTYPE", "auto")             # auto | bfloat16 | float16 | float32
DEVICE_MAP   = os.environ.get("MEDPAO_DEVICE_MAP", "auto")        # "auto" spreads big models over all GPUs
LOAD_IN_4BIT = os.environ.get("MEDPAO_4BIT", "0") == "1"          # needs bitsandbytes; saves GPU memory
DEFAULT_MAX_NEW_TOKENS = 1024     # used when a caller does not give a limit (e.g. the structured-report retry)
SAMPLING_MIN_TEMPERATURE = 0.05   # below this the call is treated as greedy (the tools use 0.01 for "almost greedy")
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_ROLES = ("vlm", "text")
TEXT_ADAPTER_NAMES = {"extractor": "extractor", "categorizer": "categorizer"}   # role -> peft adapter name
_CACHE: Dict[tuple, object] = {}
_LOAD_LOCK = threading.Lock()
_TEXT_LOCK = threading.Lock()   # adapters are switched on one shared model, so text generations run one at a time


# ── configuration ────────────────────────────────────────────────────────────
def is_local() -> bool:
    return BACKEND == "local"


def configure(backend: Optional[str] = None, vlm: Optional[str] = None, vlm_adapter: Optional[str] = None,
              text_model: Optional[str] = None, extractor_adapter: Optional[str] = None,
              categorizer_adapter: Optional[str] = None, dtype: Optional[str] = None,
              device_map: Optional[str] = None, load_in_4bit: Optional[bool] = None) -> None:
    """Set the backend and the model paths. Only the arguments that are given are changed. Giving any model path
    without an explicit backend switches to the local backend."""
    global BACKEND, DTYPE, DEVICE_MAP, LOAD_IN_4BIT, TEXT_BASE_MODEL, EXTRACTOR_ADAPTER, CATEGORIZER_ADAPTER
    given = {"vlm": vlm, "vlm_adapter": vlm_adapter, "text": text_model, "extractor_adapter": extractor_adapter,
             "categorizer_adapter": categorizer_adapter}
    for role in ("vlm", "vlm_adapter"):
        if given[role]:
            LOCAL_MODELS[role] = given[role]
    TEXT_BASE_MODEL = text_model or TEXT_BASE_MODEL
    EXTRACTOR_ADAPTER = extractor_adapter or EXTRACTOR_ADAPTER
    CATEGORIZER_ADAPTER = categorizer_adapter or CATEGORIZER_ADAPTER
    if backend is not None:
        if backend not in ("server", "local"):
            raise ValueError(f"backend must be 'server' or 'local', got {backend!r}")
        BACKEND = backend
    elif any(given.values()):
        BACKEND = "local"
    if dtype is not None:
        DTYPE = dtype
    if device_map is not None:
        DEVICE_MAP = device_map
    if load_in_4bit is not None:
        LOAD_IN_4BIT = load_in_4bit


def add_cli_args(parser) -> None:
    g = parser.add_argument_group("LLM backend (default: the HTTP servers configured in tools/*.py)")
    g.add_argument("--backend", choices=["server", "local"], default=None,
                   help="server: call the TGI servers | local: load the models below in this process "
                        "(implied when a --*-model path is given)")
    g.add_argument("--vlm-model", help="report-generation VLM: local path or Hub id (base or merged MedGemma)")
    g.add_argument("--vlm-adapter", help="optional LoRA adapter folder/Hub id applied on top of --vlm-model")
    g.add_argument("--text-model", help=f"shared base LLM for the text steps (default: {TEXT_BASE_MODEL})")
    g.add_argument("--extractor-adapter", help="LoRA for concept extraction (default: EXTRACTOR_ADAPTER in tools/llm_backend.py)")
    g.add_argument("--categorizer-adapter", help="LoRA for concept categorization (default: CATEGORIZER_ADAPTER in tools/llm_backend.py)")
    g.add_argument("--llm-dtype", choices=["auto", "bfloat16", "float16", "float32"], default=None)
    g.add_argument("--device-map", default=None, help="transformers device_map for the LLMs (default: auto)")
    g.add_argument("--load-in-4bit", action="store_true", default=None, help="4-bit quantization (bitsandbytes)")


def configure_from_args(args) -> None:
    configure(backend=args.backend, vlm=args.vlm_model, vlm_adapter=args.vlm_adapter,
              text_model=args.text_model, extractor_adapter=args.extractor_adapter,
              categorizer_adapter=args.categorizer_adapter, dtype=args.llm_dtype,
              device_map=args.device_map, load_in_4bit=args.load_in_4bit)


# ── model loading ────────────────────────────────────────────────────────────
def _dtype(torch):
    if DTYPE != "auto":
        return getattr(torch, DTYPE)
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16   # on CPU, float32 would double the memory of a multi-billion-parameter model


def _model_kwargs() -> dict:
    import torch
    dtype = _dtype(torch)
    kwargs = {"torch_dtype": dtype, "device_map": DEVICE_MAP}
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    return kwargs


def _require(role: str) -> str:
    path = LOCAL_MODELS.get(role)
    if not path:
        raise RuntimeError(f"Local backend: no model configured for '{role}'. Set --{role.replace('_', '-')}-model "
                           f"(or MEDPAO_{role.upper()}_MODEL).")
    return path


def _load_vlm():
    key = ("vlm", _require("vlm"), LOCAL_MODELS.get("vlm_adapter"))
    with _LOAD_LOCK:
        if key not in _CACHE:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            path, adapter = key[1], key[2]
            print(f"[llm_backend] loading VLM {path}" + (f" + adapter {adapter}" if adapter else ""))
            model = AutoModelForImageTextToText.from_pretrained(path, **_model_kwargs())
            if adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter)
            _CACHE[key] = (model.eval(), AutoProcessor.from_pretrained(path))
    return _CACHE[key]


def _text_adapters() -> Dict[str, str]:
    """role -> adapter path for the text steps that have a LoRA. Unusable placeholders are an error, not a silent skip."""
    adapters = {"extractor": EXTRACTOR_ADAPTER, "categorizer": CATEGORIZER_ADAPTER}
    for role, path in adapters.items():
        if not path or (path.startswith("/path/to/")):
            raise RuntimeError(f"Local backend: set the {role} LoRA path (--{role}-adapter, MEDPAO_{role.upper()}_ADAPTER, "
                               f"or {role.upper()}_ADAPTER at the top of tools/llm_backend.py).")
    return adapters


def _load_text():
    adapters = _text_adapters()
    key = ("text", TEXT_BASE_MODEL, tuple(sorted(adapters.items())))
    with _LOAD_LOCK:
        if key not in _CACHE:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[llm_backend] loading text base model {TEXT_BASE_MODEL}")
            model = AutoModelForCausalLM.from_pretrained(TEXT_BASE_MODEL, **_model_kwargs())
            first = True
            for role, path in adapters.items():
                print(f"[llm_backend]   + {role} LoRA {path}")
                if first:
                    model = PeftModel.from_pretrained(model, path, adapter_name=TEXT_ADAPTER_NAMES[role])
                    first = False
                else:
                    model.load_adapter(path, adapter_name=TEXT_ADAPTER_NAMES[role])
            _CACHE[key] = (model.eval(), AutoTokenizer.from_pretrained(TEXT_BASE_MODEL))
    return _CACHE[key]


def preload(roles: Sequence[str] = REQUIRED_ROLES) -> None:
    """Load all configured models now (local backend only; a no-op for the server backend). Fails early, with the
    list of missing settings, instead of deep inside the first request."""
    if not is_local():
        return
    if "vlm" in roles and not LOCAL_MODELS.get("vlm"):
        raise RuntimeError("Local backend: missing --vlm-model (or MEDPAO_VLM_MODEL).")
    if "text" in roles:
        _text_adapters()
    if "vlm" in roles:
        _load_vlm()
    if "text" in roles:
        _load_text()
    print("[llm_backend] all models loaded")


# ── generation ───────────────────────────────────────────────────────────────
def _frequency_penalty_processor(penalty: float, prompt_len: int):
    """OpenAI-style frequency penalty (logit -= penalty * count of the token in the generated text so far)."""
    import torch
    from transformers import LogitsProcessor

    class FrequencyPenalty(LogitsProcessor):
        def __call__(self, input_ids, scores):
            generated = input_ids[:, prompt_len:]
            if generated.shape[1] == 0:
                return scores
            counts = torch.zeros_like(scores).scatter_add_(1, generated, torch.ones_like(generated, dtype=scores.dtype))
            return scores - penalty * counts

    return FrequencyPenalty()


def _cut_at_stop(text: str, stop: Optional[Sequence[str]]) -> str:
    """Servers return the text before the stop sequence; transformers keeps it, so cut it off."""
    if stop:
        cuts = [text.find(s) for s in stop if s in text]
        if cuts:
            text = text[:min(cuts)]
    return text


def _generate(model, tokenizer, inputs, *, max_new_tokens, temperature=None, repetition_penalty=None,
              frequency_penalty=None, stop=None) -> str:
    import torch
    from transformers import LogitsProcessorList

    prompt_len = inputs["input_ids"].shape[-1]
    kwargs = {"max_new_tokens": max_new_tokens or DEFAULT_MAX_NEW_TOKENS, "do_sample": False}
    if temperature is not None and temperature > SAMPLING_MIN_TEMPERATURE:
        kwargs.update(do_sample=True, temperature=temperature)
    if repetition_penalty:
        kwargs["repetition_penalty"] = repetition_penalty
    if stop:
        kwargs.update(stop_strings=list(stop), tokenizer=tokenizer)
    if frequency_penalty:
        kwargs["logits_processor"] = LogitsProcessorList([_frequency_penalty_processor(frequency_penalty, prompt_len)])
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    return _cut_at_stop(text, stop)


def vlm_generate(image, prompt: str, *, max_new_tokens: int, temperature: Optional[float] = None,
                 repetition_penalty: Optional[float] = None, frequency_penalty: Optional[float] = None,
                 stop: Optional[Sequence[str]] = None) -> str:
    """Image + prompt -> text with the local VLM (same chat format as report_generator/finetune.py)."""
    model, processor = _load_vlm()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=[[image.convert("RGB")]], return_tensors="pt")
    import torch
    inputs = inputs.to(model.device, dtype=_dtype(torch))   # casts only the floating-point tensors (pixel values)
    return _generate(model, processor.tokenizer, inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                     repetition_penalty=repetition_penalty, frequency_penalty=frequency_penalty, stop=stop)


def _text_generate(role: str, inputs_fn, *, max_new_tokens, stop=None) -> str:
    """Run the shared text model with the LoRA of `role` active ('chat' = plain base model, adapters disabled)."""
    model, tokenizer = _load_text()
    with _TEXT_LOCK:
        inputs = inputs_fn(tokenizer).to(model.device)
        inputs.pop("token_type_ids", None)
        if role == "chat":
            with model.disable_adapter():
                return _generate(model, tokenizer, inputs, max_new_tokens=max_new_tokens, stop=stop)
        model.set_adapter(TEXT_ADAPTER_NAMES[role])
        return _generate(model, tokenizer, inputs, max_new_tokens=max_new_tokens, stop=stop)


def text_complete(prompt: str, *, max_new_tokens: int, stop: Optional[Sequence[str]] = None) -> str:
    """Raw text completion (no chat template), like the TGI text-generation endpoint. Uses the extractor LoRA."""
    return _text_generate("extractor", lambda tok: tok(prompt, return_tensors="pt"),
                          max_new_tokens=max_new_tokens, stop=stop)


def chat(messages: List[dict], *, max_new_tokens: Optional[int] = None, role: str = "chat") -> str:
    """Chat messages ([{'role', 'content'}, ...]) -> assistant reply. role='categorizer' uses the categorization LoRA,
    role='chat' (default, structured report) the plain base model."""
    if role not in ("chat", "categorizer"):
        raise ValueError(f"unknown chat role {role!r}")

    def build(tok):
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return tok(text, return_tensors="pt", add_special_tokens=False)

    return _text_generate(role, build, max_new_tokens=max_new_tokens)
