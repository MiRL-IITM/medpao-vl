import pandas as pd
import json
import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import InferenceClient
import base64
import io
import requests

# ───────────────────────────── CONFIG (edit here) ─────────────────────────────
TGI_URL              = "http://localhost:8082"   # TGI multimodal endpoint (no /v1/ suffix)
MODEL_NAME           = "tgi"                      # model name sent to the server
REQUEST_TIMEOUT      = 300                        # seconds, for every request
IMAGE_FORMAT         = "JPEG"                     # encoding used for the base64 image (JPEG or PNG)
STOP_SEQUENCES       = ["\n\n\n", "---", "Impression:", "IMPRESSION:"]

# Full grounded report (first pass)
REPORT_MAX_NEW_TOKENS       = 300
REPORT_REPETITION_PENALTY   = 1.5
REPORT_FREQUENCY_PENALTY    = 1.5
COMPLETIONS_MAX_TOKENS      = 80     # last-resort /v1/completions fallback only

# Focused "missing concepts" report (re-check passes)
MISSING_MAX_NEW_TOKENS      = 80
MISSING_TEMPERATURE         = 0.01
MISSING_REPETITION_PENALTY  = 1.3
MISSING_FREQUENCY_PENALTY   = 1.2

SAVINGS_DIR_NAME     = "savings"                          # default: <cwd>/savings
GROUNDED_REPORT_FILE = "generated_grounded_report.json"   # output (also updated by missing-concept passes)
# ──────────────────────────────────────────────────────────────────────────────

# --- Client Setup ---
client = InferenceClient(base_url=TGI_URL + "/v1/", timeout=REQUEST_TIMEOUT)


def image_to_base64_url(pil_image):
    """Converts a PIL Image to a base64 data URL."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format=IMAGE_FORMAT)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{IMAGE_FORMAT.lower()};base64,{img_str}"


def parse_vision_concepts(vision_concepts):
    """Parse vision concepts from string or dict into a dict with 'present' and 'absent' keys."""
    if isinstance(vision_concepts, dict) and 'present' in vision_concepts:
        return vision_concepts
    
    if isinstance(vision_concepts, str):
        present = []
        absent = []
        concepts = [c.strip() for c in vision_concepts.split(',')]
        for concept in concepts:
            if concept.endswith('(1.0)'):
                present.append(concept[:-5].strip())
            elif concept.endswith('(-1.0)'):
                absent.append(concept[:-6].strip())
            else:
                # No marker — treat as present
                present.append(concept.strip())
        return {'present': present, 'absent': absent}
    
    return {'present': [], 'absent': []}


def _remove_repeated_sentences(text: str) -> str:
    """Remove duplicate sentences from generated text."""
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
    seen = set()
    unique = []
    for s in sentences:
        normalized = s.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(s)
    new_report = '. '.join(unique) + '.' if unique else text
    new_report = new_report.split('Final Answer:')[0]
    return new_report.strip()


def generate_the_grounded_report(img_path, vision_concepts, saving_dir=os.path.join(os.getcwd(), SAVINGS_DIR_NAME), missing_prompt=None, id=None):
    """
    Processes a single image: loads it, calls the TGI VLM server,
    and returns the prediction.
    """
    print("\n===========================================Generating grounded report=========================")
    # Parse vision_concepts into dict with present/absent keys
    parsed_concepts = parse_vision_concepts(vision_concepts)
    
    if missing_prompt is None:
        try:
            # 1. Load the image
            if not os.path.exists(img_path):
                print(f"Image not found at {img_path}")
                return {'prediction': "Image not found"}
            
            image = Image.open(img_path).convert("RGB")
            
            image_url = image_to_base64_url(image)

            prompt_text = (
                f"You are an expert radiologist analyzing a chest X-ray. "
                f"Based on the image and observed clinical concepts, generate a concise radiology report. "
                f"Write only the findings section. Do not repeat any sentence.\n"
                f"Observed Concepts:\n"
                f"Present: {', '.join(parsed_concepts['present'])}\n"
                f"Absent: {', '.join(parsed_concepts['absent'])}\n\n"
                f"Findings:"
            )

            # Try chat completions first, fall back to generate endpoint
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ]
                output = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=REPORT_MAX_NEW_TOKENS,
                    stream=False,
                    frequency_penalty=REPORT_FREQUENCY_PENALTY,
                    repetition_penalty=REPORT_REPETITION_PENALTY,
                    stop=STOP_SEQUENCES,
                )
                report = output.choices[0].message.content
            except Exception as chat_err:
                print(f"⚠️ Chat completions failed ({chat_err}), falling back to /generate endpoint …")
                # Fallback: use the raw /generate endpoint
                payload = {
                    "inputs": prompt_text,
                    "parameters": {
                        "max_new_tokens": REPORT_MAX_NEW_TOKENS,
                        "do_sample": False,
                        "repetition_penalty": REPORT_REPETITION_PENALTY,
                        "stop": STOP_SEQUENCES,
                    },
                }
                response = requests.post(
                    f"{TGI_URL}/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code == 200:
                    result = response.json()
                    report = result.get("generated_text", str(result))
                else:
                    # Final fallback: use the /v1/completions endpoint (non-chat)
                    payload_v1 = {
                        "model": MODEL_NAME,
                        "prompt": prompt_text,
                        "max_tokens": COMPLETIONS_MAX_TOKENS,
                        "stop": STOP_SEQUENCES,
                    }
                    response_v1 = requests.post(
                        f"{TGI_URL}/v1/completions",
                        json=payload_v1,
                        headers={"Content-Type": "application/json"},
                        timeout=REQUEST_TIMEOUT,
                    )
                    if response_v1.status_code == 200:
                        result_v1 = response_v1.json()
                        report = result_v1["choices"][0]["text"]

                        
                    else:
                        raise Exception(
                            f"All endpoints failed. "
                            f"/generate: {response.status_code} {response.text[:200]}, "
                            f"/v1/completions: {response_v1.status_code} {response_v1.text[:200]}"
                        )

            # Post-process: remove repeated sentences
            report = _remove_repeated_sentences(report)

            llm_out = {'image_path':img_path, 'report': report, 'vision_concepts': vision_concepts if isinstance(vision_concepts, str) else str(vision_concepts)}
            os.makedirs(saving_dir, exist_ok=True)
            with open(os.path.join(saving_dir, GROUNDED_REPORT_FILE), 'w') as f:
                json.dump(llm_out, f, indent=4)
            return report

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            return f"Error generating report: {e}"

    else:
        try:
            # Load image
            if not os.path.exists(img_path):
                print(f"Image not found at {img_path}")
                return {'prediction': "Image not found"}

            image = Image.open(img_path).convert("RGB")
            image_url = image_to_base64_url(image)

            # Prompt focused ONLY on missing concepts
            prompt_text = (
                "You are an expert chest radiologist.\n"
                "A previous radiology report missed some findings.\n"
                "Carefully re-check the chest X-ray ONLY for the concepts listed below. and generate a missing part of report\n\n"
                "Rules:\n"
                "- Mention ONLY the listed concepts.\n"
                "- If a concept is visible, describe it clearly.\n"
                "- If absent findings are truly absent, explicitly state their absence.\n"
                "- Do not generate a full report.\n"
                "- Do not hallucinate unrelated findings.\n"
                "- Return concise findings only.\n"
                "- Do not repeat sentences.\n\n"
                "- Write it professionally.\n"
                "- Do not write 1. or 2. sections, report must be completely free style\n"
                "- DO NOT BOLD OR HIGHLIGHT ANYTHING, OUTPUT SHOULD BE SIMPLE TEXT."
                f"Missing Concepts:\n{', '.join(missing_prompt)}\n\n"
                "Focused Findings:"
            )

            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt_text
                            },
                        ],
                    },
                ]

                output = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=MISSING_MAX_NEW_TOKENS,
                    stream=False,
                    temperature=MISSING_TEMPERATURE,
                    frequency_penalty=MISSING_FREQUENCY_PENALTY,
                    repetition_penalty=MISSING_REPETITION_PENALTY,
                    stop=STOP_SEQUENCES,
                )

                missing_report = output.choices[0].message.content.strip()

            except Exception as chat_err:
                print(f"⚠️ Chat completion failed: {chat_err}")

                payload = {
                    "inputs": prompt_text,
                    "parameters": {
                        "max_new_tokens": MISSING_MAX_NEW_TOKENS,
                        "do_sample": False,
                        "temperature": MISSING_TEMPERATURE,
                        "repetition_penalty": MISSING_REPETITION_PENALTY,
                        "stop": STOP_SEQUENCES,
                    },
                }

                response = requests.post(
                    f"{TGI_URL}/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 200:
                    result = response.json()
                    missing_report = result.get("generated_text", "").strip()
                else:
                    raise Exception(
                        f"/generate endpoint failed: "
                        f"{response.status_code} {response.text[:200]}"
                    )

            # Clean report
            missing_report = _remove_repeated_sentences(missing_report)

            grounded_report_path = os.path.join(
                                            saving_dir,
                                            GROUNDED_REPORT_FILE
                                        )

            with open(grounded_report_path, 'r') as f:
                llm_out = json.load(f)
            
            llm_out[f"missing_report{id}"] =  missing_report
     
            with open(grounded_report_path, 'w') as f:
                json.dump(llm_out, f, indent=4)


            full_report = llm_out['report']
            for i in range(id+1):
                full_report+=' '+llm_out[f"missing_report{i}"]
            
            print("MISSING CONCEPT:")
            print(missing_prompt)

            print("FULL REPORT:")
            print(full_report)
            return full_report

        except Exception as e:
            print(f"❌ Error generating missing report: {e}")
            return f"Error generating missing report: {e}"