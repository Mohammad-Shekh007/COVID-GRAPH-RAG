import json
import re
import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI

# =========================
# CONFIG
# =========================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
NEO4J_DATABASE =st.secrets["NEO4J_DATABASE"]

ENTITY_INDEX_NAME = "entity_embedding_index"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

TOP_K_VECTOR = 8
TOP_K_KEYWORD = 8
MAX_FACTS_FOR_CONTEXT = 15
MAX_PATHS_FOR_CONTEXT = 10

client = OpenAI(api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

embedding_cache = {}


# =========================
# HELPERS
# =========================
def get_embedding(text: str):
    text = (text or "").strip()
    if text in embedding_cache:
        return embedding_cache[text]

    emb = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL
    ).data[0].embedding

    embedding_cache[text] = emb
    return emb


def extract_keywords(text: str):
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]+\b", text.lower())
    stop = {
        "the", "is", "are", "was", "were", "what", "why", "how", "and", "or", "for",
        "of", "to", "in", "on", "with", "by", "from", "that", "this", "these", "those",
        "a", "an", "at", "as", "be", "it", "its", "their", "there", "about", "does",
        "did", "do", "can", "could", "should", "would", "will", "latest", "please"
    }
    words = [w for w in words if w not in stop and len(w) > 2]
    return list(dict.fromkeys(words))[:8]


def format_fact(record):
    s = record.get("subject", "")
    p = record.get("predicate", "")
    o = record.get("object", "")

    extras = []
    for k in ["value", "unit", "date", "location", "condition", "source"]:
        v = record.get(k)
        if v not in [None, "", "unknown"]:
            extras.append(f"{k}: {v}")

    meta = " | ".join(extras)
    if meta:
        return f"FACT: {s} {p} {o} | {meta}"
    return f"FACT: {s} {p} {o}"


def format_path(record):
    n1 = record.get("n1", "")
    r1 = record.get("r1", "")
    n2 = record.get("n2", "")
    r2 = record.get("r2", "")
    n3 = record.get("n3", "")
    return f"PATH: {n1} -[{r1}]-> {n2} -[{r2}]-> {n3}"


# =========================
# STEP 1: QUERY REWRITE
# =========================
def rewrite_query(user_query: str):
    prompt = f"""
You are a retrieval query optimizer.

Rewrite the user question into one clearer, more retrieval-friendly version.
Keep the meaning exactly the same.
Do not answer the question.

User query:
{user_query}

Return only the rewritten query.
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


# =========================
# STEP 2: QUERY EXPANSION
# =========================
def expand_query(rewritten_query: str):
    prompt = f"""
Generate 3 short alternate search phrasings for this query.
Use synonyms and related wording.
Do not answer the query.

Query:
{rewritten_query}

Return valid JSON only:
{{
  "expansions": ["...", "...", "..."]
}}
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("expansions", [])


# =========================
# STEP 3: HYBRID RETRIEVAL
# =========================
def vector_retrieve_entities(query_text: str, top_k=TOP_K_VECTOR):
    query_embedding = get_embedding(query_text)

    cypher = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node, score
    RETURN node.name AS name,
           node.type AS type,
           score
    ORDER BY score DESC
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(
            cypher,
            index_name=ENTITY_INDEX_NAME,
            top_k=top_k,
            embedding=query_embedding
        )
        return [dict(r) for r in result]


def keyword_retrieve_entities(query_text: str, top_k=TOP_K_KEYWORD):
    keywords = extract_keywords(query_text)
    if not keywords:
        return []

    conditions = []
    params = {"top_k": top_k}

    for i, kw in enumerate(keywords):
        key = f"kw{i}"
        params[key] = kw
        conditions.append(f"toLower(n.name) CONTAINS ${key} OR toLower(n.type) CONTAINS ${key}")

    where_clause = " OR ".join(conditions)

    cypher = f"""
    MATCH (n:Entity)
    WHERE {where_clause}
    RETURN n.name AS name,
           n.type AS type,
           1.0 AS score
    LIMIT $top_k
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(cypher, **params)
        return [dict(r) for r in result]


# =========================
# STEP 4: GRAPH FACT RETRIEVAL
# =========================
def get_neighbor_facts(entity_names):
    if not entity_names:
        return []

    cypher = """
    MATCH (s:Entity)-[r]->(o:Entity)
    WHERE s.name IN $names OR o.name IN $names
    RETURN s.name AS subject,
           type(r) AS predicate,
           o.name AS object,
           r.value AS value,
           r.unit AS unit,
           r.date AS date,
           r.location AS location,
           r.condition AS condition,
           r.source AS source,
           r.text AS rel_text
    LIMIT 100
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(cypher, names=entity_names)
        return [dict(r) for r in result]


# =========================
# STEP 5: GRAPH REASONING (2-HOP PATHS)
# =========================
def get_reasoning_paths(entity_names):
    if not entity_names:
        return []

    cypher = """
    MATCH path = (a:Entity)-[r1]->(b:Entity)-[r2]->(c:Entity)
    WHERE a.name IN $names OR b.name IN $names OR c.name IN $names
    RETURN a.name AS n1,
           type(r1) AS r1,
           b.name AS n2,
           type(r2) AS r2,
           c.name AS n3
    LIMIT 100
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(cypher, names=entity_names)
        return [dict(r) for r in result]


# =========================
# STEP 6: RERANK FACTS + PATHS
# =========================
def rerank_context(user_query: str, facts, paths):
    fact_lines = [format_fact(f) for f in facts]
    path_lines = [format_path(p) for p in paths]

    prompt = f"""
You are a graph relevance reranker.

Given a user query, a list of factual triples, and a list of graph reasoning paths,
select the most useful evidence.

Rules:
- Prefer items that directly answer the question
- Prefer causal or multi-step chains for "why", "how", "start", "lead to", "result" questions
- Use the paths when they connect relevant entities
- Do not invent information

User query:
{user_query}

Fact candidates:
{json.dumps(fact_lines, indent=2)}

Path candidates:
{json.dumps(path_lines, indent=2)}

Return valid JSON only:
{{
  "selected_facts": ["..."],
  "selected_paths": ["..."]
}}

Select at most {MAX_FACTS_FOR_CONTEXT} facts and {MAX_PATHS_FOR_CONTEXT} paths.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    return (
        data.get("selected_facts", [])[:MAX_FACTS_FOR_CONTEXT],
        data.get("selected_paths", [])[:MAX_PATHS_FOR_CONTEXT]
    )


# =========================
# STEP 7: GROUNDED GENERATION WITH PATH REASONING
# =========================
def generate_answer(user_query: str, rewritten_query: str, expansions, selected_facts, selected_paths):
    facts_block = "\n".join(selected_facts) if selected_facts else "No facts found."
    paths_block = "\n".join(selected_paths) if selected_paths else "No paths found."

    prompt = f"""
You are a grounded Knowledge Graph assistant.

You must answer ONLY from:
1. FACT CONTEXT
2. PATH CONTEXT

Important:
- PATH CONTEXT represents connected graph reasoning chains
- If a path logically answers the question, use it explicitly
- You MAY connect the dots using the path structure
- Do NOT invent facts beyond the provided paths and facts
- If the answer is not supported, say so clearly

Rules:
1. Be concise and direct
2. Use the paths when the question asks "how", "why", "start", "lead to", "cause", "sequence"
3. If a connected chain exists, explain it as a reasoning path
4. Do not hallucinate

Original user query:
{user_query}

Rewritten retrieval query:
{rewritten_query}

Expanded search queries:
{json.dumps(expansions)}

FACT CONTEXT:
{facts_block}

PATH CONTEXT:
{paths_block}

Return your answer in this format:

Answer:
<final answer>

Grounded Evidence:
- <fact or path 1>
- <fact or path 2>
- <fact or path 3>
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


# =========================
# MAIN PIPELINE
# =========================
def run_chatbot(user_query: str):
    rewritten = rewrite_query(user_query)
    expansions = expand_query(rewritten)

    vector_hits = vector_retrieve_entities(rewritten, TOP_K_VECTOR)
    keyword_hits = keyword_retrieve_entities(rewritten, TOP_K_KEYWORD)

    for exp in expansions:
        vector_hits.extend(vector_retrieve_entities(exp, max(3, TOP_K_VECTOR // 2)))
        keyword_hits.extend(keyword_retrieve_entities(exp, max(3, TOP_K_KEYWORD // 2)))

    combined_entities = []
    seen = set()

    for hit in vector_hits + keyword_hits:
        name = hit.get("name", "")
        if name and name not in seen:
            seen.add(name)
            combined_entities.append(name)

    combined_entities = combined_entities[:20]

    facts = get_neighbor_facts(combined_entities)
    paths = get_reasoning_paths(combined_entities)

    selected_facts, selected_paths = rerank_context(user_query, facts, paths)

    answer = generate_answer(
        user_query,
        rewritten,
        expansions,
        selected_facts,
        selected_paths
    )

    debug_info = {
        "rewritten_query": rewritten,
        "expansions": expansions,
        "retrieved_entities": combined_entities,
        "raw_fact_count": len(facts),
        "raw_path_count": len(paths),
        "selected_facts": selected_facts,
        "selected_paths": selected_paths
    }

    return answer, debug_info


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="KG + Advanced RAG Chatbot", layout="wide")
st.title("KG + Advanced RAG Chatbot")
st.markdown("Ask questions over your Neo4j knowledge graph with query rewriting, expansion, hybrid retrieval, graph reasoning, reranking, and grounded generation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

with st.sidebar:
    st.header("Settings")
    st.session_state.debug_mode = st.checkbox("Show debug info", value=False)
    st.markdown("**Database:** Neo4j local")
    st.markdown("**Models:**")
    st.markdown(f"- Embeddings: `{EMBED_MODEL}`")
    st.markdown(f"- Chat: `{CHAT_MODEL}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("debug") and st.session_state.debug_mode:
            with st.expander("Debug Details"):
                st.json(message["debug"])

user_query = st.chat_input("Ask a question about the graph...")

if user_query:
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, debug_info = run_chatbot(user_query)
                st.markdown(answer)

                if st.session_state.debug_mode:
                    with st.expander("Debug Details"):
                        st.json(debug_info)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "debug": debug_info
                })

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
