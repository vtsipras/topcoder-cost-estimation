import pandas as pd
import requests
import json
import time
import re

# Configuration
MODELS = {
    'Llama': 'llama3.2'
}

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """
You are an expert Software Engineering Project Estimator. 
Read the following raw project description from a crowdsourcing platform. Ignore any navigational menus, footers, or unrelated text.
Focus only on the technical requirements and extract the following features.

CRITICAL: You MUST output ONLY a valid JSON object. No explanations, no markdown (do not use ```json). Just the raw JSON format exactly like this:
{
  "Complexity": 3,
  "Tech_Count": 4,
  "QA_Effort": 2,
  "Deliverables_Count": 3,
  "Duration_Days": 7
}

Rules for values:
- Complexity: Integer from 1 (Very Simple) to 5 (Extremely Complex).
- Tech_Count: Integer count of distinct programming languages/frameworks required.
- QA_Effort: Integer from 1 (Minimal testing) to 5 (Heavy QA/Security).
- Deliverables_Count: Integer count of distinct deliverables expected.
- Duration_Days: Integer representing the estimated number of days to complete the project based on the text. If no specific deadline is mentioned, make a logical estimation based on the project size.
"""

def extract_json(text, model_name):
    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(clean_text)
    except Exception:
        print(f"Error: JSON parsing failed for model {model_name}.")
        return {"Complexity": 0, "Tech_Count": 0, "QA_Effort": 0, "Deliverables_Count": 0, "Duration_Days": 0}

def query_local_ollama(model_name, description):
    prompt = f"{SYSTEM_PROMPT}\n\nPROJECT DESCRIPTION:\n{description[:3000]}"
    payload = {"model": model_name, "prompt": prompt, "stream": False, "format": "json"}
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return extract_json(response.json().get("response", ""), model_name)
    except requests.exceptions.ConnectionError:
         print("Error: Ollama service is not responding.")
    return {"Complexity": 0, "Tech_Count": 0, "QA_Effort": 0, "Deliverables_Count": 0, "Duration_Days": 0}

if __name__ == "__main__":
    print("Starting LLM Feature Extraction...")
    
    # Load dataset and handle resume state
    try:
        df = pd.read_csv("topcoder_features_extracted_LOCAL.csv")
        print("Found existing output file. Resuming process.")
    except FileNotFoundError:
        try:
            df = pd.read_csv("topcoder_dataset.csv")
            print("Created new extraction file.")
        except FileNotFoundError:
            print("Error: topcoder_dataset.csv not found.")
            exit()
    
    # df = df.head(50) 
    
    features = ['Complexity', 'Tech_Count', 'QA_Effort', 'Deliverables_Count', 'Duration_Days']
    
    # Initialize missing columns
    for llm_display_name in MODELS.keys():
        for feat in features:
            if f'{feat}_{llm_display_name}' not in df.columns:
                df[f'{feat}_{llm_display_name}'] = 0

    output_filename = "topcoder_features_extracted_LOCAL.csv"

    # Main execution loop
    for display_name, ollama_model_name in MODELS.items():
        print(f"Loading model: {display_name}...")
        
        for index, row in df.iterrows():
            # Skip rows that have already been evaluated
            if df.at[index, f'Complexity_{display_name}'] != 0:
                continue

            description = str(row['Description'])
            if len(description) < 50:
                continue
                
            estimations = query_local_ollama(ollama_model_name, description)
            
            for feat in features:
                val = estimations.get(feat, 0)
                df.at[index, f'{feat}_{display_name}'] = val
                
            c_val = df.at[index, f'Complexity_{display_name}']
            print(f"[{index+1}/{len(df)}] {display_name} -> Evaluated task: {str(row['Title'])[:30]}... (Complexity: {c_val})")

            # Auto-save checkpoint
            df.to_csv(output_filename, index=False, encoding='utf-8')

    print(f"Process completed successfully. Data saved to {output_filename}")