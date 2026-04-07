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
    results = query_embeddings(query_emb, n_results=8)  # increased from 5

    docs = results["documents"]
    distances = results["distances"]

    if not docs:
        return "No relevant information found in PDFs."

    MIN_SIMILARITY = 0.30  # lowered from 0.42 — MCP passes raw chunks to LLM anyway
    MIN_LENGTH = 80

    filtered = [
        doc for doc, dist in zip(docs, distances)
        if (1 - dist) >= MIN_SIMILARITY
        and len(doc.strip()) >= MIN_LENGTH
        and not is_junk_chunk(doc)
    ]

    if not filtered:
        return "No sufficiently relevant information found for this query."

    return "\n\n---\n\n".join(filtered)


if __name__ == "__main__":
    mcp.run()