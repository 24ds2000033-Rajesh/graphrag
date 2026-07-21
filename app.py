import os
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI

app = FastAPI(title="GraphMind Systems - GraphRAG API")

# Configure AI Pipe / OpenAI client
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN") or os.environ.get("AIPIPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openrouter/v1")

client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url=BASE_URL
)

MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-4.1-nano")

def safe_parse_json(content: str) -> dict:
    """Safely parse JSON responses from proxy endpoints."""
    if not content:
        raise ValueError("LLM returned an empty response.")
    
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\s*```$", "", content, flags=re.MULTILINE)
    
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        raise ValueError(f"Model response was not valid JSON: {content[:200]}")

# ==========================================
# PYDANTIC MODELS (MUST BE DEFINED BEFORE ENDPOINTS)
# ==========================================

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

# ==========================================
# ENDPOINTS
# ==========================================

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
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@app.post("/graph-query")
def graph_query(req: GraphQueryRequest):
    system_prompt = """
    You are a graph reasoning agent. Analyze the provided knowledge graph JSON and answer the question.
    
    Output strictly valid JSON only:
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
        content = response.choices[0].message.content
        return safe_parse_json(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@app.post("/community-summary")
def community_summary(req: CommunitySummaryRequest):
    system_prompt = """
    Summarize the sub-community of the knowledge graph in 1-2 concise sentences.
    
    Output strictly valid JSON only:
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
        content = response.choices[0].message.content
        return safe_parse_json(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "GraphRAG Server active", "model": MODEL_NAME}
