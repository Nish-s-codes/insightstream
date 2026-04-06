import chromadb
import uuid
import hashlib

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def get_text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def is_duplicate_chunk(text: str) -> bool:
    text_hash = get_text_hash(text)
    results = collection.get(where={"hash": text_hash})
    return len(results["ids"]) > 0

def store_embeddings(chunks, embeddings, source_file):
    skipped = 0
    
    docs_to_add = []
    embs_to_add = []
    ids_to_add = []
    metas_to_add = []
    
    for i, chunk in enumerate(chunks):
        if is_duplicate_chunk(chunk["text"]):
            skipped += 1
            continue
            
        docs_to_add.append(chunk["text"])
        embs_to_add.append(embeddings[i])
        ids_to_add.append(str(uuid.uuid4()))
        metas_to_add.append({
            "source": source_file,
            "chunk_id": i,
            "page": chunk.get("page", 0),
            "hash": get_text_hash(chunk["text"])
        })
        
    if docs_to_add:
        collection.add(
            documents=docs_to_add,
            embeddings=embs_to_add,
            ids=ids_to_add,
            metadatas=metas_to_add
        )
        
    return skipped

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