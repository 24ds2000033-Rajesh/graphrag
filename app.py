import os
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI

app = FastAPI(title="GraphMind Systems - GraphRAG API")

# Read token and base URL
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN") or os.environ.get("AIPIPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openrouter/v1")

client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url=BASE_URL
)

MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-4.1-nano")

def safe_parse_json(content: str) -> dict:
    """Robustly parse JSON responses from proxy endpoints."""
    if not content:
        raise ValueError("LLM returned an empty response.")
    
    # Clean markdown fences if present
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\s*```$", "", content, flags=re.MULTILINE)
    
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        raise ValueError(f"Model response was not valid JSON. Raw output: {content[:200]}")

# --- Endpoints ---

@app.post("/extract-graph")
def extract_graph(req: ExtractRequest):
    system_prompt = """
    You are an expert data extractor. Extract entities and relationships from the provided text.
    Allowed Entity Types: Person, Organization, Product, Framework.
    Allowed Relationship Types: FOUNDED, DEVELOPED, INTEGRATED_INTO, HIRED, AUTHORED, CREATED.
    
    Output strictly valid JSON with no markdown formatting:
    {
      "entities": [{"name": "...", "type": "..."}],
      "relationships": [{"source": "...", "target": "...", "relation": "..."}]
    }
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.text}
            ],
            temperature=0
        )
        content = response.choices[0].message.content
        return safe_parse_json(content)
    except Exception as e:
        # Return exact error string to identify broken proxy paths
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")
