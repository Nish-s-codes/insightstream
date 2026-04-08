# mcp/mcp_rag.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.server.fastmcp import FastMCP
from app.services.embed import get_embedding
from app.db.vector_store import query_embeddings
from app.services.rag import is_junk_chunk

mcp = FastMCP("PDF-Knowledge-Base")


@mcp.tool()
def search_pdfs(query: str) -> str:
    """Search through uploaded PDF documents to find relevant technical information."""

    query_emb = get_embedding(query)
    results = query_embeddings(query_emb, n_results=8)

    docs = results.get("documents", [])
    distances = results.get("distances", [])

    if not docs:
        return "[INFO] No relevant information found in PDFs."

    MIN_SIMILARITY = 0.30
    MIN_LENGTH = 80
    MAX_RESULTS = 3  # 🔴 prevent overload

    filtered = []

    for doc, dist in zip(docs, distances):
        similarity = 1 - dist

        if (
            similarity >= MIN_SIMILARITY
            and len(doc.strip()) >= MIN_LENGTH
            and not is_junk_chunk(doc)
        ):
            # -------- CLEAN TEXT --------
            clean_doc = doc.replace("\x00", "").strip()
            filtered.append(clean_doc)

        if len(filtered) >= MAX_RESULTS:
            break

    if not filtered:
        return "[INFO] No sufficiently relevant information found for this query."

    # -------- FORMAT OUTPUT --------
    formatted = []
    for i, chunk in enumerate(filtered, 1):
        formatted.append(f"[Result {i}]\n{chunk}")

    return "\n\n---\n\n".join(formatted)

if __name__ == "__main__":
    mcp.run()