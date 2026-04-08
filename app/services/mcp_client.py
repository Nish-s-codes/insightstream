# app/services/mcp_client.py

RAG_TOOLS = {"search_pdfs"}

GITHUB_TOOLS = {
    "list_repos",
    "list_files",
    "read_file",
    "get_readme",
    "read_repo_for_summary",
    "commit_file",
    "delete_file",
}


async def call_tool(tool_name: str, arguments: dict, app) -> str:
    """
    Routes tool calls to the correct MCP server and safely parses output.
    """

    # -------- ROUTING --------
    if tool_name in RAG_TOOLS:
        session = getattr(app.state, "mcp_rag_session", None)
        label = "RAG"

    elif tool_name in GITHUB_TOOLS:
        session = getattr(app.state, "mcp_github_session", None)
        label = "GitHub"

    else:
        return f"[ERROR] Unknown tool: {tool_name}"

    # -------- SESSION CHECK --------
    if session is None:
        return f"[ERROR] {label} MCP session not initialized."

    try:
        # -------- TOOL EXECUTION --------
        result = await session.call_tool(tool_name, arguments=arguments)

        # -------- EMPTY CHECK --------
        if not result or not hasattr(result, "content") or not result.content:
            return f"[INFO] {tool_name} executed but returned no data."

        # -------- PARSE TEXT SAFELY --------
        text_parts = []

        for item in result.content:
            if hasattr(item, "text") and item.text:
                text_parts.append(item.text)

            # fallback for unexpected formats
            elif isinstance(item, dict):
                text_parts.append(str(item))

        if text_parts:
            return "\n".join(text_parts)

        return f"[INFO] {tool_name} returned data but no readable text found."

    except Exception as e:
        return f"[ERROR] Tool '{tool_name}' failed: {str(e)}"