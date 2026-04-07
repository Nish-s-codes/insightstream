# app/services/rag.py
import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from groq import AsyncGroq
from app.services.embed import get_embedding
from app.db.vector_store import query_embeddings

load_dotenv()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

JUNK_PATTERNS = [
    "you may also like", "notes for professionals", "goalKicker",
    "www.", "http", "......", "chapter ", "section ",
]

def is_junk_chunk(text: str) -> bool:
    t = text.lower().strip()
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

def retrieve_context(query: str):
    """Pure retrieval logic — returns docs, scores, and confidence. Sync is fine here."""
    broad = any(kw in query.lower() for kw in ["all", "explain", "describe", "overview", "summary"])
    top_k = 12 if broad else 6

    query_embedding = get_embedding(expand_query(query))
    results = query_embeddings(query_embedding, n_results=20)

    docs = results["documents"]
    distances = results["distances"]

    if not docs:
        return None, [], "none"

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

    if not paired or max(s for _, s in paired) < 0.42:
        return None, [], "none"

    paired.sort(key=lambda x: x[1], reverse=True)
    top_docs = [p[0] for p in paired[:top_k]]
    top_scores = [p[1] for p in paired[:top_k]]

    if max(top_scores) > 0.6:
        confidence = "high"
    elif max(top_scores) > 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    return top_docs, top_scores, confidence


async def answer_question_stream(query: str) -> AsyncGenerator[str, None]:
    """Real streaming — yields tokens as the LLM generates them."""
    top_docs, top_scores, confidence = retrieve_context(query)

    # Send confidence metadata first
    yield f"[CONFIDENCE: {confidence}]\n\n"

    if top_docs is None:
        yield "I can only answer questions related to the uploaded documents. This question doesn't appear to be relevant to the content."
        return

    context = "\n\n---\n\n".join(top_docs)
    prompt = f"""You are a helpful assistant answering questions based on documentation excerpts.
Use the provided context as your primary source. You may use your general knowledge to clarify technical terms or fill minor gaps, but make clear what comes from the docs vs general knowledge.
If the context is irrelevant or doesn't answer the question, say: 'I don't know based on the uploaded documents.'

Context from documentation:
{context}

Question: {query}
Answer:"""

    stream = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500,
        stream=True,
    )

    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


async def answer_question(query: str) -> dict:
    """Non-streaming version — kept for the /query route."""
    top_docs, top_scores, confidence = retrieve_context(query)

    if top_docs is None:
        return {
            "answer": "I can only answer questions related to the uploaded documents.",
            "sources": [],
            "best_source": "N/A",
            "confidence": "none"
        }

    context = "\n\n---\n\n".join(top_docs)
    prompt = f"""You are a helpful assistant answering questions based on documentation excerpts.
Context:
{context}

Question: {query}
Answer:"""

    # For non-streaming, collect the full response
    full_response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500,
        stream=False,
    )

    answer = full_response.choices[0].message.content
    clean_sources = [f"{i+1}. {' '.join(s.split())[:150]}" for i, s in enumerate(top_docs)]

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