# app/services/mcp_client.py
import app.main as main_app

async def call_search_pdfs(query: str) -> str:
    if main_app.mcp_session is None:
        return "MCP session not initialized. Server may still be starting."

    result = await main_app.mcp_session.call_tool(
        "search_pdfs",
        arguments={"query": query}
    )

    if result.content:
        return result.content[0].text
    return "No relevant information found."