# app/services/mcp_client.py

# 8000 chars ≈ 2000 tokens — large enough for a 90-line requirements.txt
# (~5000 chars) without choking Groq's context window.
MAX_OUTPUT_CHARS = 8000

# Tools that live on the RAG server
RAG_TOOLS = {"search_pdfs"}

# Tools that live on the custom GitHub server (mcp/mcp_github.py)
CUSTOM_GITHUB_TOOLS = {
    "list_repos",
    "list_files",
    "read_file",
    "get_readme",
    "commit_file",
    "delete_file",
}

# Everything else goes to the official GitHub MCP (npx server).


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(result) -> str:
    """Pull text out of an MCP result object and truncate if needed."""
    if not result or not hasattr(result, "content"):
        return "[INFO] No result returned."

    parts = []
    for item in result.content:
        if hasattr(item, "text") and item.text:
            parts.append(item.text)
        elif isinstance(item, dict) and "text" in item:
            parts.append(item["text"])
        else:
            parts.append(str(item))

    text = "\n".join(parts) if parts else "[INFO] No readable output."

    if len(text) > MAX_OUTPUT_CHARS:
        return (
            text[:MAX_OUTPUT_CHARS]
            + f"\n\n... [output truncated at {MAX_OUTPUT_CHARS} chars] ..."
        )
    return text


# ── Main entry point ──────────────────────────────────────────────────────────

async def call_tool(tool_name: str, arguments: dict, app) -> str:
    """Route a tool call to the correct MCP session."""
    if tool_name in RAG_TOOLS:
        session = getattr(app.state, "mcp_rag_session", None)
    elif tool_name in CUSTOM_GITHUB_TOOLS:
        session = getattr(app.state, "mcp_github_session", None)
    else:
        session = getattr(app.state, "mcp_github_official", None)
        if session is None:
            return (
                f"[ERROR] Official GitHub MCP is not running, "
                f"so '{tool_name}' is unavailable. "
                f"Check that Node.js/npx is installed."
            )

    if session is None:
        return (
            f"[ERROR] No active MCP session for '{tool_name}'. "
            "Check startup logs and /debug/mcp."
        )

    try:
        result = await session.call_tool(tool_name, arguments=arguments)
        return extract_text(result)
    except Exception as e:
        return f"[ERROR] Tool '{tool_name}' failed: {e}"