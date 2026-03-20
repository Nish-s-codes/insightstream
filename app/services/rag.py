import requests
import os
from dotenv import load_dotenv
import numpy as np
from app.services.embed import get_embedding
from app.db.vector_store import query_embeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def answer_question(query: str):
    # 🔹 Detect broad query
    broad_query = False
    if "all" in query.lower() or "explain" in query.lower():
        broad_query = True

    # 1. Embed query
    query_embedding = get_embedding(query)

    # 2. Retrieve relevant chunks
    results = query_embeddings(query_embedding)

    docs = results["documents"]
    distances = results["distances"]

    if not docs:
        return {"answer": "No relevant data found"}

    # 3. Filter + clean chunks
    filtered_docs = []
    filtered_scores = []

    for doc, dist in zip(docs, distances):
        similarity = 1 - dist

        # ❌ remove junk chunks
        if len(doc.strip()) < 100:
            continue

        if "section" in doc.lower() and len(doc) < 200:
            continue

        if similarity > 0.4:
            filtered_docs.append(doc)
            filtered_scores.append(similarity)

    # 4. Fallback
    if not filtered_docs:
        filtered_docs = docs[:10]
        filtered_scores = [1 - d for d in distances[:10]]

    # 5. Sort by similarity
    paired = list(zip(filtered_docs, filtered_scores))
    paired.sort(key=lambda x: x[1], reverse=True)

    # 🔹 Dynamic chunk selection
    if broad_query:
        top_k = 10
    else:
        top_k = 5

    top_chunks = [p[0] for p in paired[:top_k]]
    top_scores = [p[1] for p in paired[:top_k]]

    context = "\n\n".join(top_chunks)

    # 6. Prompt
    prompt = f"""
Answer the question using the provided context.

Instructions:
- Do NOT use outside knowledge
- Extract ALL relevant points from context
- Combine information from multiple chunks
- If question asks for "all", cover all categories found
- If answer is incomplete, say what is missing

Context:
{context}

Question:
{query}
"""

    # 7. Groq API call
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    try:
        answer = result["choices"][0]["message"]["content"]
    except:
        answer = "Error generating response"

    # 8. Confidence calculation
    if top_scores:
        avg_similarity = sum(top_scores) / len(top_scores)
    else:
        avg_similarity = 0

    if avg_similarity > 0.75:
        confidence = "high"
    elif avg_similarity > 0.5:
        confidence = "medium"
    else:
        confidence = "low"

    # 9. Best source
    best_source = top_chunks[0] if top_chunks else "Not found"

    # 10. Clean sources
    clean_sources = []
    for i, s in enumerate(top_chunks):
        cleaned = " ".join(s.split())
        clean_sources.append(f"{i+1}. {cleaned[:150]}")

    return {
        "answer": answer,
        "sources": clean_sources,
        "best_source": best_source[:200] if best_source != "Not found" else "Not found",
        "confidence": confidence
    }