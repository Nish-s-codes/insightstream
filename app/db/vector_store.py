import chromadb
import uuid

client = chromadb.Client()
collection = client.get_or_create_collection(name="documents")


def store_embeddings(chunks, embeddings):
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],  # ✅ FIX: correct embedding per chunk
            ids=[str(uuid.uuid4())],     # ✅ FIX: unique id
            metadatas=[{"source": "pdf"}]
        )


def query_embeddings(query_embedding, n_results=8):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"]