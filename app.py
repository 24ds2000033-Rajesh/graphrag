import os
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI

app = FastAPI(title="GraphMind Systems - GraphRAG API")

# Configure AI Pipe client
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN") or os.environ.get("AIPIPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openrouter/v1")

client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url=BASE_URL
)

# Use free router by default if not explicitly set
MODEL_NAME = os.environ.get("MODEL_NAME", "openrouter/free")

def clean_json_response(content: str) -> dict:
    """Helper to safely parse JSON output even if model wraps it in markdown fences."""
    content = content.strip()
    # Remove markdown ```json and ``` fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\s*```$", "", content, flags=re.MULTILINE)
    return json.loads(content.strip())

# --- Request Models ---

class ExtractRequest(BaseModel):
    chunk_id: str
    text: str

class GraphQueryRequest(BaseModel):
    question: str
    graph: Dict[str, Any]

class CommunitySummaryRequest(BaseModel):
    community_id: str
    entities: List[str]
    relationships: List[Dict[str, str]]

# --- Endpoints ---

@app.post("/extract-graph")
def extract_graph(req: ExtractRequest):
    system_prompt = """
    You are an expert data extractor. Extract entities and relationships from the provided text.
    Allowed Entity Types: Person, Organization, Product, Framework.
    Allowed Relationship Types: FOUNDED, DEVELOPED, INTEGRATED_INTO, HIRED, AUTHORED, CREATED.
    
    Output strictly VALID JSON only with no conversational text or markdown:
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
        return clean_json_response(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph-query")
def graph_query(req: GraphQueryRequest):
    system_prompt = """
    You are a graph reasoning agent. Analyze the provided knowledge graph JSON and answer the question.
    
    Output strictly VALID JSON only with no markdown or extra commentary:
    {
      "answer": "...",
      "reasoning_path": ["Entity1", "Entity2"],
      "hops": 2
    }
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Graph: {json.dumps(req.graph)}\nQuestion: {req.question}"}
            ],
            temperature=0
        )
        return clean_json_response(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/community-summary")
def community_summary(req: CommunitySummaryRequest):
    system_prompt = """
    Summarize the sub-community of the knowledge graph in 1-2 concise sentences.
    
    Output strictly VALID JSON only with no extra commentary:
    {
      "community_id": "...",
      "summary": "..."
    }
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Community ID: {req.community_id}\nEntities: {json.dumps(req.entities)}\nRelationships: {json.dumps(req.relationships)}"}
            ],
            temperature=0.2
        )
        return clean_json_response(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "GraphRAG Server active", "model": MODEL_NAME}
