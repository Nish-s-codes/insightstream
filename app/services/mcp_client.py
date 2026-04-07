# app/services/mcp_client.py

RAG_TOOLS = {"search_pdfs"}

GITHUB_TOOLS = {
    "list_repos", "list_files", "read_file",
    "get_readme", "read_repo_for_summary",
    "commit_file", "delete_file"
}


async def call_tool(tool_name: str, arguments: dict, app) -> str:
    # -------- ROUTING --------
    if tool_name in RAG_TOOLS:
        session = app.state.mcp_rag_session
        label = "RAG"
    elif tool_name in GITHUB_TOOLS:
        session = app.state.mcp_github_session
        label = "GitHub"
    else:
        return f"[ERROR] Unknown tool: {tool_name}"

    # -------- SESSION CHECK --------
    if session is None:
        return f"[ERROR] {label} MCP session not initialized."

    try:
        # -------- TOOL CALL --------
        result = await session.call_tool(tool_name, arguments=arguments)

        # -------- SAFE PARSING --------
        if not result or not result.content:
            return f"[INFO] {tool_name} executed but returned no results."

        # Extract all readable text chunks
        text_parts = [
            item.text for item in result.content
            if hasattr(item, "text") and item.text
        ]

        if text_parts:
            return "\n".join(text_parts)

        return f"[INFO] {tool_name} returned data, but no readable text found."

    except Exception as e:
        return f"[ERROR] Tool '{tool_name}' failed: {str(e)}"