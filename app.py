import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI

app = FastAPI(title="GraphMind Systems - GraphRAG API")

# Configure OpenAI client to route requests through AI Pipe
# Default base_url points to AI Pipe's OpenRouter proxy endpoint
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN") or os.environ.get("AIPIPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openrouter/v1")

client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url=BASE_URL
)

# Model to use via AI Pipe (e.g. "openai/gpt-4o-mini" or "gpt-4o-mini")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-4o-mini")

# --- Pydantic Models ---

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
    
    Output strictly in JSON format with two keys:
    - "entities": list of objects with "name" and "type".
    - "relationships": list of objects with "source", "target", and "relation".
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.text}
            ],
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph-query")
def graph_query(req: GraphQueryRequest):
    system_prompt = """
    You are a graph reasoning agent. You will be provided with a JSON knowledge graph (entities and relationships) and a question.
    Perform multi-hop reasoning to find the answer.
    
    Output strictly in JSON format with three keys:
    - "answer": The direct answer to the question.
    - "reasoning_path": A list of entity names tracing the path from the starting entity to the answer.
    - "hops": Integer representing the number of relationship edges traversed.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Graph: {json.dumps(req.graph)}\nQuestion: {req.question}"}
            ],
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/community-summary")
def community_summary(req: CommunitySummaryRequest):
    system_prompt = """
    You are an AI tasked with summarizing a specific sub-community of a knowledge graph.
    You will receive a community ID, a list of entities, and a list of relationships.
    Write a concise, 1-2 sentence summary explaining how these entities are connected.
    
    Output strictly in JSON format with two keys:
    - "community_id": The exact ID provided.
    - "summary": The generated summary string.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Community ID: {req.community_id}\nEntities: {json.dumps(req.entities)}\nRelationships: {json.dumps(req.relationships)}"}
            ],
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "GraphRAG Server is running via AI Pipe"}
