import chromadb
import uuid

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")

def store_embeddings(chunks, embeddings, source_file):
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[str(uuid.uuid4())],
            metadatas=[{
                "source": source_file,
                "chunk_id": i
            }]
        )

def query_embeddings(query_embedding, n_results=9):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return {
        "documents": results["documents"][0],
        "distances": results["distances"][0]
    }