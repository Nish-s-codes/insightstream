# app/db/vector_store.py
import hashlib
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def get_text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def store_embeddings(chunks, embeddings, source_file):
    if not chunks:
        return 0

    ids = [get_text_hash(c["text"]) for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "source": source_file,
            "page": c.get("page", 0),
            "hash": get_text_hash(c["text"])
        } for c in chunks]
    )
    return 0  # upsert handles duplicates automatically

def query_embeddings(query_embedding, n_results=20):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )
    return {
        "documents": results["documents"][0],
        "distances": results["distances"][0],
        "metadatas": results["metadatas"][0]
    }