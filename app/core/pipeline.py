from app.services.ingest import extract_text, chunk_text
from app.services.embed import get_embedding
from app.db.vector_store import store_embeddings

def process_document(file_path):
    text = extract_text(file_path)

    if not text:
        return {"error": "No text extracted"}

    chunks = chunk_text(text)

    embeddings = [get_embedding(chunk) for chunk in chunks]

    store_embeddings(chunks, embeddings)

    return {
        "num_chunks": len(chunks)
    }