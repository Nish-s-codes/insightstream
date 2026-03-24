import requests
import os
from dotenv import load_dotenv
from app.services.embed import get_embedding
from app.db.vector_store import query_embeddings
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

JUNK_PATTERNS = [
    "you may also like",
    "notes for professionals",
    "goalKicker",
    "www.",
    "http",
    "......",
    "chapter ",
    "section ",
]
def is_junk_chunk(text: str) -> bool:
    t = text.lower().strip()
    # too many dots = table of contents line
    if t.count(".") > 10:
        return True
    for pattern in JUNK_PATTERNS:
        if pattern in t and len(t) < 300:
            return True
    return False

def expand_query(query: str) -> str:
    expansions = {
        "disk space": "disk space df du filesystem storage usage",
        "cpu": "cpu processor mpstat top usage",
        "memory": "memory ram free top usage",
        "network": "network ifconfig ip netstat",
        "check": "check monitor status view display",
        "process": "process ps kill top running",
        "file": "file ls cat find grep directory",
        "permission": "permission chmod chown access rights",
    }
    expanded = query
    for keyword, terms in expansions.items():
        if keyword in query.lower():
            expanded += " " + terms
    return expanded

def answer_question(query: str):
    broad = any(kw in query.lower() for kw in ["all", "explain", "describe", "overview", "summary"])
    top_k = 12 if broad else 6

    query_embedding = get_embedding(expand_query(query))
    results = query_embeddings(query_embedding, n_results=20)

    docs = results["documents"]
    distances = results["distances"]

    if not docs:
        return {"answer": "No relevant data found."}

    MIN_SIMILARITY = 0.42
    MIN_LENGTH = 80

    paired = []
    for doc, dist in zip(docs, distances):
        similarity = 1 - dist
        if len(doc.strip()) < MIN_LENGTH:
            continue
        if is_junk_chunk(doc):
            continue
        if similarity < MIN_SIMILARITY:
            continue
        paired.append((doc, similarity))

    # off-topic detection — if best match is still weak, reject
    if not paired or max(s for _, s in paired) < 0.42:
        return {
            "answer": "I can only answer questions related to the uploaded documents. This question doesn't appear to be relevant to the content.",
            "sources": [],
            "best_source": "N/A",
            "confidence": "none"
        }

    paired.sort(key=lambda x: x[1], reverse=True)

    top_docs = [p[0] for p in paired[:top_k]]
    top_scores = [p[1] for p in paired[:top_k]]

    context = "\n\n---\n\n".join(top_docs)

    prompt = f"""You are a helpful assistant answering questions based on documentation excerpts.

Use the provided context as your primary source. You may use your general knowledge to clarify technical terms or fill minor gaps, but make clear what comes from the docs vs general knowledge.

If the context is irrelevant or doesn't answer the question, just say: 'I don't know based on the uploaded documents.' Do not explain what the context contains.

Context from documentation:
{context}

Question: {query}

Answer:"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1500
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"Error generating response: {e}"

    if not top_scores:
        confidence = "low"
    elif max(top_scores) > 0.6:
        confidence = "high"
    elif max(top_scores) > 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    clean_sources = [f"{i+1}. {' '.join(s.split())[:150]}" for i, s in enumerate(top_docs)]

    # hide sources if LLM couldn't answer from context
    if answer.strip().lower().startswith("i don't know"):
        clean_sources = []
        best_source = "N/A"
    else:
        best_source = top_docs[0][:200] if top_docs else "Not found"

    return {
        "answer": answer,
        "sources": clean_sources,
        "best_source": best_source,
        "confidence": confidence
    }