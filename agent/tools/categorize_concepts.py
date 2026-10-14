import json
from huggingface_hub import InferenceClient
import re
import os
from tqdm import tqdm

# ───────────────────────────── CONFIG (edit here) ─────────────────────────────
LLM_BASE_URL        = "http://localhost:8080/v1/"   # OpenAI-compatible (TGI) chat endpoint
LLM_MODEL           = "tgi"                          # model name sent to the server
MAX_NEW_TOKENS      = 200                            # generation budget for the categorization reply

SAVINGS_DIR_NAME    = "savings"                      # default: <cwd>/savings
CACHE_DIR_SUFFIX    = "_cache"                       # cache dir = <saving_dir> + this suffix
SEGREGATED_FILE     = "segregated.json"              # input  (written by ontology_map_sapbert.py)
CONCEPTS_FILE       = "concepts_output.json"         # input  (written by get_concept.py)
NEW_CONCEPTS_FILE   = "new_concepts_output.json"     # present when only cache-miss concepts were processed
CATEGORIZED_FILE    = "categorized_concepts.json"    # output
CACHED_VOCAB_FILE   = "cached_vocab.json"            # running vocabulary inside the cache dir
# ──────────────────────────────────────────────────────────────────────────────

def load_json_file(file_path):
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json_file(file_path, data, indent=4):
    """Save data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=indent)

def update_and_save_json(file_path, new_data):
    """Update existing JSON file with new data or create new file."""
    existing_data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    
    else:
        existing_data={}
        
    existing_data.update(new_data)
    save_json_file(file_path, existing_data)
    print("Vocab saved to ", file_path)



def create_categorization_prompt2(concepts_report, ontology_data):
    """Create system message and prompt for categorizing concepts."""
    sys_msg = '''You are radiology expert with the following knowledge:
            Chest x-ray review is a key competency for medical students, junior doctors and other allied health professionals. Using P, A, B, C, D, E, F is a helpful and systematic method for chest x-ray review:
            
            P: projection views(PA, AP, lateral views)

            A: airways (intraluminal mass, narrowing, splayed carina)

            B: breathing (lungs, pulmonary vessels, pleural spaces)

            C: circulation (cardiomediastinal contour, great vessels)

            D: diaphragm and below (diaphragmatic paresis, pneumoperitoneum, gaseous distension, splenomegaly, calculi)

            E: external e.g. chest wall (ribs, shoulder girdles, fractures), soft tissues

            F: foreign material (devices, foreign bodies, gossypibomas)

            '''
            
    prompt = f"""
        You will be given:

        1. A concept-source sentence extracted from a radiology report.
        2. A list of medical concepts with their ontology information.
        3. A protocol category.

        Your task is to categorize each medical concept into one of the protocol categories: P, A, B, C, D, E, or F.

        Use BOTH:
        - the concept-source sentence for contextual understanding, and
        - the ontology information (description and ancestors) for semantic understanding.

        Ontology information follows the structure:

        {{
            "concept": "<medical concept>",
            "description": "<ontology description>",
            "ancestors": [
                "<ancestor 1>",
                "<ancestor 2>",
                ...
            ]
        }}

        The description represents the normalized ontology meaning of the concept.
        The ancestors represent progressively broader ontology categories and may provide important clues about the anatomical structure, pathology, severity, procedure, device, or clinical finding represented by the concept.

        Carefully analyze the concept, its ontology description, its ancestors, and the report context before assigning a category.

        Example:

        INPUT:

        Concept-source sentence:
        "The heart is mildly enlarged. Small left pleural effusion is present."

        Ontology data:

        [
            {{
                "concept": "cardiomegaly",
                "description": "Cardiomegaly (disorder)",
                "ancestors": [
                    "Structural disorder of heart",
                    "Heart disease",
                    "Cardiac finding",
                    "Cardiovascular finding",
                    "Clinical finding"
                ]
            }},
            {{
                "concept": "pleural effusion",
                "description": "Pleural effusion (disorder)",
                "ancestors": [
                    "Disorder of pleura and pleural cavity",
                    "Respiratory finding",
                    "Clinical finding"
                ]
            }}
        ]

        OUTPUT:

        {{
            "cardiomegaly": "C",
            "pleural effusion": "B"
        }}

        Now categorize all concepts.

        Concept-source sentence:
        {concepts_report}

        Ontology data:
        {ontology_data}

        Return ONLY a valid JSON object in the format:

        {{
            "<concept_1>": "<category>",
            "<concept_2>": "<category>"
        }}

        Do not provide explanations, reasoning, markdown, or additional text.
        """
    
    return sys_msg, prompt





def parse_model_response(output):
    """Extract and parse JSON from model response."""
    pattern = r'\{.*?\}'
    try:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        return {}
    except Exception as e:
        print(f"Error parsing model response: {e}")
        return {}

def categorize_the_concepts(start, saving_dir = os.path.join(os.getcwd(), SAVINGS_DIR_NAME)):
    """
    Categorize medical concepts into P/A/B/C/D/E/F categories using an LLM.
    
    Args:
        start: A trigger parameter (not functionally used)
        
    Returns:
        dict: Categorized medical concepts
    """
    # Setup paths
    print("============================================Entered concept categorization===============================================")
    base_dir = os.getcwd()
    segregated_file = os.path.join(saving_dir, SEGREGATED_FILE)
    concepts_output_file = os.path.join(saving_dir, CONCEPTS_FILE)
    categorized_file = os.path.join(saving_dir, CATEGORIZED_FILE)
    vocab_file = os.path.join(saving_dir+CACHE_DIR_SUFFIX, CACHED_VOCAB_FILE)
    os.makedirs(saving_dir + CACHE_DIR_SUFFIX, exist_ok=True)
    
    # Load input data
    ontology_data = load_json_file(segregated_file)
    if ontology_data is None:
        print(f"File {segregated_file} does not exist. Please check the path.")
        return {'concepts already in vocabulary': 'skipped'}
    
    concepts_report = load_json_file(concepts_output_file)
    if concepts_report is None:
        print(f"File {concepts_output_file} does not exist. Please check the path.")
        return {'concepts already in vocabulary': 'skipped'}
    
    # Setup inference client
    client = InferenceClient(base_url=LLM_BASE_URL)
    
    # Create prompt and get model response
    sys_msg, prompt = create_categorization_prompt2(concepts_report, ontology_data)
    
    try:

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            max_tokens=MAX_NEW_TOKENS
        )
        output = response.choices[0].message.content.strip()
        # print(output)
        # Parse response
        categorized_concepts = parse_model_response(output)
        
        # Save results
        new_path = os.path.join(saving_dir, NEW_CONCEPTS_FILE)
        if os.path.exists(new_path):
            update_and_save_json(categorized_file, categorized_concepts)
        else:
            save_json_file(categorized_file, categorized_concepts)
        update_and_save_json(vocab_file, categorized_concepts)
        
        return categorized_concepts
        
    except Exception as e:
        print(f"Error during model inference: {e}")
        return {}





