import requests
import os
from dotenv import load_dotenv
import numpy as np
from app.services.embed import get_embedding
from app.db.vector_store import query_embeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def answer_question(query: str):
    # 1. embed query
    enhanced_query = f"""
    Find exact section or explanation for:
    {query}
     """

    query_embedding = get_embedding(enhanced_query)

    # 2. retrieve relevant chunks
    docs = query_embeddings(query_embedding)

    if not docs or not docs[0]:
        return {"answer": "No relevant data found"}

    context = " ".join(docs[0])

    # 3. prompt
    prompt = f"""
    You are a helpful assistant.

    Answer ONLY using the context below.
    Do NOT guess.
    If answer is partially present, try to answer using available context.
    Combine information from multiple parts of context if needed.
    If the answer is not in the context, say "Not found in document."

    Context:
    {context}

    Question:
    {query}
    """

    # 4. Groq API call
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

    # Optional debug (remove later)
    print(result)

    try:
        answer = result["choices"][0]["message"]["content"]
    except:
        answer = "Error generating response"

    # 🔥 Best source detection (FIXED INDENT)
    highlighted_source = None
    for s in docs[0]:
        if answer[:100].lower() in s.lower():
            highlighted_source = s
            break

    # 🔥 Confidence
    confidence = "high" if highlighted_source else "medium"

    # 🔥 Clean sources
    clean_sources = []
    for i, s in enumerate(docs[0]):
        cleaned = " ".join(s.split())
        clean_sources.append(f"{i+1}. {cleaned[:150]}")

    return {
        "answer": answer,
        "sources": clean_sources,
        "best_source": highlighted_source[:200] if highlighted_source else "Not found",
        "confidence": confidence
    }