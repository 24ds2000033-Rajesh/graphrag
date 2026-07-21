from fastapi import FastAPI
from pydantic import BaseModel
import networkx as nx
import re

app = FastAPI()

########################################################################
# MODELS
########################################################################

class ExtractRequest(BaseModel):
    chunk_id: str
    text: str

class GraphQueryRequest(BaseModel):
    question: str
    graph: dict

class CommunityRequest(BaseModel):
    community_id: str
    entities: list
    relationships: list

########################################################################
# ENTITY HELPERS
########################################################################

ENTITY_TYPES = {
    "Framework": [
        "LangChain",
        "LlamaIndex",
        "Haystack",
        "AutoGen",
        "CrewAI",
        "Semantic Kernel",
        "DSPy"
    ],
    "Organization": [
        "OpenAI",
        "Microsoft",
        "Google",
        "Meta",
        "Anthropic",
        "NVIDIA",
        "Apple",
        "Amazon"
    ],
    "Product": [
        "ChatGPT",
        "GPT-4",
        "GPT-3",
        "Claude",
        "Gemini",
        "Copilot"
    ]
}

def detect_entities(text):
    entities = []

    # predefined entities
    for typ, values in ENTITY_TYPES.items():
        for v in values:
            if re.search(r"\b"+re.escape(v)+r"\b", text, re.I):
                entities.append({"name":v,"type":typ})

    # person detection
    persons = re.findall(
        r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b",
        text
    )

    blacklist = set(sum(ENTITY_TYPES.values(), []))

    for p in persons:
        if p not in blacklist:
            entities.append({"name":p,"type":"Person"})

    # deduplicate

    seen=set()
    final=[]

    for e in entities:
        key=(e["name"],e["type"])
        if key not in seen:
            seen.add(key)
            final.append(e)

    return final

########################################################################
# RELATIONSHIP EXTRACTION
########################################################################

PATTERNS = [

(
r"(.+?) was created by (.+)",
"CREATED"
),

(
r"(.+?) was developed by (.+)",
"DEVELOPED"
),

(
r"(.+?) was founded by (.+)",
"FOUNDED"
),

(
r"(.+?) integrates with (.+)",
"INTEGRATED_INTO"
),

(
r"(.+?) integrated into (.+)",
"INTEGRATED_INTO"
),

(
r"(.+?) hired (.+)",
"HIRED"
),

(
r"(.+?) authored (.+)",
"AUTHORED"
),

(
r"(.+?) created (.+)",
"CREATED"
),

(
r"(.+?) developed (.+)",
"DEVELOPED"
),

(
r"(.+?) founded (.+)",
"FOUNDED"
)

]

def relationship_extract(text):

    rels=[]

    for pattern,rel in PATTERNS:

        m=re.search(pattern,text,re.I)

        if not m:
            continue

        left=m.group(1).strip(" .,")

        right=m.group(2).strip(" .,")

        if rel in ["CREATED","DEVELOPED","FOUNDED"]:

            rels.append({
                "source":right,
                "target":left,
                "relation":rel
            })

        elif rel=="HIRED":

            rels.append({
                "source":left,
                "target":right,
                "relation":rel
            })

        elif rel=="AUTHORED":

            rels.append({
                "source":left,
                "target":right,
                "relation":rel
            })

        else:

            rels.append({
                "source":left,
                "target":right,
                "relation":rel
            })

    return rels

########################################################################
# ENDPOINT 1
########################################################################

@app.post("/extract-graph")
def extract_graph(req:ExtractRequest):

    entities=detect_entities(req.text)

    rels=relationship_extract(req.text)

    return {
        "entities":entities,
        "relationships":rels
    }

########################################################################
# GRAPH QUERY
########################################################################

@app.post("/graph-query")
def graph_query(req:GraphQueryRequest):

    G=nx.Graph()

    for e in req.graph["entities"]:
        G.add_node(e["name"])

    for r in req.graph["relationships"]:
        G.add_edge(
            r["source"],
            r["target"],
            relation=r["relation"]
        )

    q=req.question.lower()

    if "who" in q:

        framework=None

        org=None

        for e in req.graph["entities"]:

            if e["type"]=="Framework":
                framework=e["name"]

            if e["type"]=="Organization":
                org=e["name"]

        if framework and org:

            try:
                path=nx.shortest_path(G,org,framework)

                creator=None

                for r in req.graph["relationships"]:

                    if r["target"]==framework and r["relation"] in [
                        "CREATED",
                        "DEVELOPED",
                        "FOUNDED"
                    ]:

                        creator=r["source"]

                        return {
                            "answer":creator,
                            "reasoning_path":[org,framework,creator],
                            "hops":2
                        }

            except:
                pass

    return {
        "answer":"Unknown",
        "reasoning_path":[],
        "hops":0
    }

########################################################################
# COMMUNITY SUMMARY
########################################################################

@app.post("/community-summary")
def community(req:CommunityRequest):

    names=", ".join(req.entities)

    rels=[]

    for r in req.relationships:
        rels.append(
            f'{r["source"]} {r["relation"]} {r["target"]}'
        )

    summary=(
        f"This community contains entities {names}. "
        f"Key relationships include: "
        + "; ".join(rels)
        + "."
    )

    return {
        "community_id":req.community_id,
        "summary":summary
    }
