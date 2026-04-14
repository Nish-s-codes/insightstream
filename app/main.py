# app/main.py
#
# Official GitHub MCP (npx @modelcontextprotocol/server-github) is DISABLED.
# It requires Node.js + a Copilot-compatible PAT.
# To re-enable: uncomment the block marked [OFFICIAL MCP - DISABLED] below.
#
# Active MCP servers:
#   1. Custom GitHub MCP  — list_repos, list_files, read_file, get_readme, commit_file, delete_file
#   2. RAG MCP            — search_pdfs

import os
import sys
from contextlib import asynccontextmanager, AsyncExitStack

from fastapi import FastAPI, Request
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

from app.api.routes import router as main_router
from app.core.streamer import router as stream_router
from app.api.ask import router as ask_router
from app.core.tool_loader import invalidate_tool_cache


@asynccontextmanager
async def lifespan(app: FastAPI):

    async with AsyncExitStack() as stack:

        # ── RAG MCP (local stdio) ─────────────────────────────────────────────
        print("Starting RAG MCP server...")
        try:
            rag_transport = await stack.enter_async_context(
                stdio_client(StdioServerParameters(
                    command=sys.executable,
                    args=[os.path.abspath("mcp/mcp_rag.py")],
                    env=os.environ.copy(),
                ))
            )
            rag_session = await stack.enter_async_context(
                ClientSession(rag_transport[0], rag_transport[1])
            )
            await rag_session.initialize()
            app.state.mcp_rag_session = rag_session
            print("RAG MCP ready.")
        except Exception as e:
            print(f"RAG MCP failed: {e}")
            app.state.mcp_rag_session = None

        # ── Custom GitHub MCP (local stdio) ───────────────────────────────────
        print("Starting custom GitHub MCP...")
        try:
            gh_custom_transport = await stack.enter_async_context(
                stdio_client(StdioServerParameters(
                    command=sys.executable,
                    args=[os.path.abspath("mcp/mcp_github.py")],
                    env=os.environ.copy(),
                ))
            )
            gh_custom_session = await stack.enter_async_context(
                ClientSession(gh_custom_transport[0], gh_custom_transport[1])
            )
            await gh_custom_session.initialize()
            app.state.mcp_github_session = gh_custom_session
            print("Custom GitHub MCP ready.")
        except Exception as e:
            print(f"Custom GitHub MCP failed: {e}")
            app.state.mcp_github_session = None

        # ── [OFFICIAL MCP - DISABLED] ─────────────────────────────────────────
        # Uncomment this entire block to re-enable the official GitHub MCP.
        # Requires: Node.js + npx on PATH, GITHUB_TOKEN with repo + read:org scopes.
        #
        # def npx_command() -> str:
        #     return "npx.cmd" if sys.platform == "win32" else "npx"
        #
        # print("Starting official GitHub MCP (npx)...")
        # try:
        #     token = os.getenv("GITHUB_TOKEN")
        #     if not token:
        #         raise RuntimeError("GITHUB_TOKEN not set in .env")
        #     gh_official_transport = await stack.enter_async_context(
        #         stdio_client(StdioServerParameters(
        #             command=npx_command(),
        #             args=["-y", "@modelcontextprotocol/server-github"],
        #             env={
        #                 **os.environ,
        #                 "GITHUB_PERSONAL_ACCESS_TOKEN": token,
        #                 "GITHUB_TOOLSETS": "repos,issues,pull_requests,git,search",
        #             },
        #         ))
        #     )
        #     gh_official_session = await stack.enter_async_context(
        #         ClientSession(gh_official_transport[0], gh_official_transport[1])
        #     )
        #     await gh_official_session.initialize()
        #     app.state.mcp_github_official = gh_official_session
        #     print("Official GitHub MCP ready.")
        # except Exception as e:
        #     print(f"Official GitHub MCP failed: {e}")
        #     app.state.mcp_github_official = None

        # Always set to None while disabled so tool_loader/mcp_client don't crash
        app.state.mcp_github_official = None

        yield   # ── app is running ────────────────────────────────────────────

    invalidate_tool_cache()
    print("All MCP servers shut down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)

app.include_router(main_router)
app.include_router(stream_router)
app.include_router(ask_router)


# ── Debug endpoints ────────────────────────────────────────────────────────────

@app.get("/debug/mcp")
async def debug_mcp(request: Request):
    """Check active MCP session statuses and available tools."""
    result = {}
    sessions = {
        "mcp_github_session":  getattr(request.app.state, "mcp_github_session",  None),
        "mcp_github_official": getattr(request.app.state, "mcp_github_official", None),  # always None when disabled
        "mcp_rag_session":     getattr(request.app.state, "mcp_rag_session",     None),
    }
    for name, session in sessions.items():
        if session is None:
            result[name] = {"status": "MISSING / DISABLED"}
            continue
        try:
            tools = (await session.list_tools()).tools
            result[name] = {
                "status": "OK",
                "tool_count": len(tools),
                "tools": [t.name for t in tools],
            }
        except Exception as e:
            result[name] = {"status": "ERROR", "error": str(e)}
    return result


@app.post("/debug/reset-tools")
async def reset_tools_post():
    """Force the tool cache to refresh on the next /ask request."""
    invalidate_tool_cache()
    return {"status": "Tool cache cleared. Will reload on next /ask request."}


@app.get("/debug/reset-tools")
async def reset_tools_get():
    """Same as POST — allows triggering cache reset directly from a browser."""
    invalidate_tool_cache()
    return {"status": "Tool cache cleared. Will reload on next /ask request."}