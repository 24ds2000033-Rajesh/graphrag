from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import re
import networkx as nx

app = FastAPI(title="GraphRAG API")


###########################################################
# Models
###########################################################

class ExtractRequest(BaseModel):
    chunk_id: str
    text: str


class GraphRequest(BaseModel):
    question: str
    graph: Dict


class CommunityRequest(BaseModel):
    community_id: str
    entities: List[str]
    relationships: List[Dict]


###########################################################
# Dictionaries
###########################################################

ENTITY_TYPES = {
    "OpenAI": "Organization",
    "Google": "Organization",
    "Microsoft": "Organization",
    "Meta": "Organization",
    "Anthropic": "Organization",

    "LangChain": "Framework",
    "LlamaIndex": "Framework",
    "PyTorch": "Framework",
    "TensorFlow": "Framework",

    "GPT-4": "Product",
    "GPT-3.5": "Product",
    "Claude": "Product",
    "Gemini": "Product"
}

PERSON_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b"
)

###########################################################
# Extraction
###########################################################

RELATION_PATTERNS = [
    (r"(.+?) was created by (.+)", "CREATED"),
    (r"(.+?) was founded by (.+)", "FOUNDED"),
    (r"(.+?) developed (.+)", "DEVELOPED"),
    (r"(.+?) integrates with (.+)", "INTEGRATED_INTO"),
    (r"(.+?) integrated into (.+)", "INTEGRATED_INTO"),
    (r"(.+?) hired (.+)", "HIRED"),
    (r"(.+?) authored (.+)", "AUTHORED"),
]


def normalize(x):
    return x.strip(" .,;")


@app.post("/extract-graph")
def extract(req: ExtractRequest):

    text = req.text

    entities = []

    seen = set()

    # predefined entities
    for name, typ in ENTITY_TYPES.items():
        if name.lower() in text.lower():
            entities.append({"name": name, "type": typ})
            seen.add(name)

    # people
    for p in PERSON_PATTERN.findall(text):
        if p not in seen:
            entities.append({"name": p, "type": "Person"})
            seen.add(p)

    relationships = []

    lower = text

    # Created
    m = re.search(r"(.+?) was created by (.+?)(\.|$)", lower, re.I)
    if m:
        framework = normalize(m.group(1))
        person = normalize(m.group(2))
        relationships.append({
            "source": person,
            "target": framework,
            "relation": "CREATED"
        })

    m = re.search(r"(.+?) was founded by (.+?)(\.|$)", lower, re.I)
    if m:
        org = normalize(m.group(1))
        person = normalize(m.group(2))
        relationships.append({
            "source": person,
            "target": org,
            "relation": "FOUNDED"
        })

    for m in re.finditer(r"(.+?) integrates with (.+?)(\.|$)", lower, re.I):
        relationships.append({
            "source": normalize(m.group(1)),
            "target": normalize(m.group(2)),
            "relation": "INTEGRATED_INTO"
        })

    for m in re.finditer(r"(.+?) developed (.+?)(\.|$)", lower, re.I):
        relationships.append({
            "source": normalize(m.group(1)),
            "target": normalize(m.group(2)),
            "relation": "DEVELOPED"
        })

    for m in re.finditer(r"(.+?) hired (.+?)(\.|$)", lower, re.I):
        relationships.append({
            "source": normalize(m.group(1)),
            "target": normalize(m.group(2)),
            "relation": "HIRED"
        })

    for m in re.finditer(r"(.+?) authored (.+?)(\.|$)", lower, re.I):
        relationships.append({
            "source": normalize(m.group(1)),
            "target": normalize(m.group(2)),
            "relation": "AUTHORED"
        })

    return {
        "entities": entities,
        "relationships": relationships
    }


###########################################################
# Graph Query
###########################################################

@app.post("/graph-query")
def graph_query(req: GraphRequest):

    entities = req.graph.get("entities", [])
    rels = req.graph.get("relationships", [])

    G = nx.Graph()

    for r in rels:
        G.add_edge(r["source"], r["target"], relation=r["relation"])

    q = req.question.lower()

    answer = None
    reasoning = []

    # locate target mentioned in question
    target = None

    for e in entities:
        if e["name"].lower() in q:
            target = e["name"]
            break

    if target is None:
        return {
            "answer": "",
            "reasoning_path": [],
            "hops": 0
        }

    # Find creator/founder via multi-hop

    for person in G.nodes:

        if person == target:
            continue

        try:
            path = nx.shortest_path(G, target, person)

            if len(path) >= 2:

                last_edge = G[path[-2]][path[-1]]

                if last_edge["relation"] in [
                    "CREATED",
                    "FOUNDED",
                    "DEVELOPED",
                    "AUTHORED"
                ]:
                    answer = path[-1]
                    reasoning = path
                    break

        except:
            pass

    return {
        "answer": answer,
        "reasoning_path": reasoning,
        "hops": max(len(reasoning)-1,0)
    }


###########################################################
# Community Summary
###########################################################

@app.post("/community-summary")
def community(req: CommunityRequest):

    summary = []

    summary.append(
        f"This community contains {len(req.entities)} entities."
    )

    if req.relationships:

        rels = []

        for r in req.relationships:
            rels.append(
                f'{r["source"]} {r["relation"].lower().replace("_"," ")} {r["target"]}'
            )

        summary.append("Key relationships include: " + "; ".join(rels) + ".")

    return {
        "community_id": req.community_id,
        "summary": " ".join(summary)
    }


@app.get("/")
def home():
    return {"status":"ok"}
