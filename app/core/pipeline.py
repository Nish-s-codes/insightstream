from app.services.ingest import extract_text, chunk_text
from app.services.embed import get_embeddings_batch
from app.db.vector_store import store_embeddings

def process_document(file_path):
    pages = extract_text(file_path)

    if not pages or not any(pages):
        return {"error": "No text extracted"}

    chunks = chunk_text(pages)
    texts = [c["text"] for c in chunks]

    # batch encode all at once instead of one by one
    embeddings = get_embeddings_batch(texts)

    skipped = store_embeddings(chunks, embeddings, file_path)

    return {
        "num_chunks": len(chunks) - skipped,
        "skipped_duplicates": skipped
    }