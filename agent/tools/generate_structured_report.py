from huggingface_hub import InferenceClient
import re
import json
import os
from tqdm import tqdm

# ───────────────────────────── CONFIG (edit here) ─────────────────────────────
LLM_BASE_URL        = "http://localhost:8080/v1/"   # OpenAI-compatible (TGI) chat endpoint
LLM_MODEL           = "tgi"                          # model name sent to the server
SYSTEM_MESSAGE      = "You are a helpful clinical assistant for writing medical reports."
MAX_NEW_TOKENS      = 200                            # first attempt; retry uses the server default
THINK_END_TAG       = "</think>"                     # text before this tag is dropped from the reply

SAVINGS_DIR_NAME    = "savings"                      # default: <cwd>/savings
CATEGORIZED_FILE    = "categorized_concepts.json"    # input  (written by categorize_concepts.py)
CONCEPTS_FILE       = "concepts_output.json"         # input  (written by get_concept.py)
REPORT_FILE         = "categorized_report.json"      # output
# ──────────────────────────────────────────────────────────────────────────────


def get_prompt(concepts, concepts_report):
    """
        Constructs a prompt for the language model to generate a structured medical report.
        
        Args:
            concepts (dict): Categorized medical concepts (e.g., {"finding": "category"})
            concepts_report (dict): Mapping of concepts to their source sentences from the text report
            
        Returns:
            str: A formatted prompt instructing the LLM how to generate the report
    """
    # Define the P.A.B.C.D.E.F protocol for chest x-ray review
    protocol='''

            Chest x-ray review is a key competency for medical students, junior doctors and other allied health professionals. Using P, A, B, C, D, E, F is a helpful and systematic method for chest x-ray review:

                P: projection views(PA, AP, lateral views or no views specified)
                        
                A: airways (intraluminal mass, narrowing, splayed carina)

                B: breathing (lungs, pulmonary vessels, pleural spaces)

                C: circulation (cardiomediastinal contour, great vessels)

                D: diaphragm and below (diaphragmatic paresis, pneumoperitoneum, gaseous distension, splenomegaly, calculi)

                E: external e.g. chest wall (ribs, shoulder girdles, fractures), soft tissues

                F: foreign material (devices, foreign bodies, gossypibomas)

             '''

    # Construct the full prompt with instructions and input data
    prompt = f'''
                You are given with the categorized medical concepts according to the following protocol: {protocol} and each concept mapped to its source sentence from the text report\n
                Your task is to generate structured medical reports according to the categories mentioned above in the protocol, given the structured concepts.\n
                Your generated report must have a json format like: {{"P": "Mention views information if no views specified then write 'No views'", "A": "Findings of concepts belonging to A", "B": "Findings of concepts belonging to B", "C":"Findings of concepts belonging to C", "D":"Findings of concepts belonging to D", "E":"Findings of concepts belonging to E", "F":"Findings of concepts belonging to F"}}.\n
                If from the given concepts any of the categories are missing have its report as "No findings", for example if there are no concepts from "A" then have the findings as "No Findings".\n
                You must write report in a medical radiologist style, in a descriptive way, DONT JUST AGGREGATE THE CONCEPTS!, DONT BRING IN ANY EXTRA MEDICAL FINDINGS.\n
                So when the concept-source sentence mapping is:{concepts_report} and the concept categories are: {concepts}, the generated structured report is:
            '''
    return prompt


def get_llm_response(prompt, max_new_tokens=None):
    """
    Get the response from LLM based on the prompt.
    
    Args:
        prompt (str): The formatted prompt for the language model
        
    Returns:
        str: The LLM's response containing the generated medical report
    """
    # Initialize the local inference client
    client = InferenceClient(base_url=LLM_BASE_URL)
    
    if max_new_tokens==None:
        response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_MESSAGE},
                  {"role": "user", "content": prompt}],
        stream=False
    )
    else:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}],
            stream=False,
            max_tokens=max_new_tokens
        )
    
    # Extract the response content, removing any thinking section
    if THINK_END_TAG in response.choices[0].message.content:
        output = response.choices[0].message.content.split(THINK_END_TAG)[1]
    else:
        output = response.choices[0].message.content
    return output

def generate_the_report(concepts, saving_dir=os.path.join(os.getcwd(), SAVINGS_DIR_NAME)):
    """
    Generate a structured medical report from categorized concepts.
    
    Args:
        concepts: Input parameter to initiate the report generation process
        
    Returns:
        dict: The generated structured report in JSON format
    """
    # Load previously saved categorized concepts and concept-report mappings
    with open(os.path.join(saving_dir, CATEGORIZED_FILE)) as f:
        categorized_concepts = json.load(f)
    with open(os.path.join(saving_dir, CONCEPTS_FILE)) as f:
        concepts_report = json.load(f)
    
    # Generate the prompt and get LLM response
    user = get_prompt(categorized_concepts, concepts_report)
    outputs = get_llm_response(user, max_new_tokens=MAX_NEW_TOKENS)

    # Extract JSON data from the LLM response
    pattern = r'\{.*?\}'
    match = re.search(pattern, outputs, re.DOTALL)

    try:
        json_str = match.group(0) 
        texts = json.loads(json_str)
    except Exception as e:
        outputs = get_llm_response(user)
        match = re.search(pattern, outputs, re.DOTALL)
        print(outputs)
        json_str = match.group(0) 
        texts = json.loads(json_str)
        print(outputs)

    # Save the structured report to a file
    file_path = os.path.join(saving_dir, REPORT_FILE)
    with open(file_path, 'w') as json_file:
        json.dump(texts, json_file, indent=4)
        
    return texts


