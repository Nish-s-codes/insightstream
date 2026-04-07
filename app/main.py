# app/main.py
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.api.routes import router as main_router
from app.core.streamer import router as stream_router
from app.api.ask import router as ask_router


async def start_mcp_server(script_path: str):
    params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.abspath(script_path)],
    )
    cleanup = stdio_client(params)
    read, write = await cleanup.__aenter__()
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()
    return cleanup, session


async def stop_mcp_server(cleanup, session):
    try:
        await session.__aexit__(None, None, None)
        await cleanup.__aexit__(None, None, None)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting RAG MCP server...")
    cleanup_rag, session_rag = await start_mcp_server("mcp/mcp_rag.py")
    app.state.mcp_rag_session = session_rag
    print("RAG MCP server ready.")

    print("Starting GitHub MCP server...")
    cleanup_gh, session_gh = await start_mcp_server("mcp/mcp_github.py")
    app.state.mcp_github_session = session_gh
    print("GitHub MCP server ready.")

    yield

    print("Shutting down MCP servers...")
    await stop_mcp_server(cleanup_rag, session_rag)
    await stop_mcp_server(cleanup_gh, session_gh)
    print("Done.")


app = FastAPI(lifespan=lifespan)

app.include_router(main_router)
app.include_router(stream_router)
app.include_router(ask_router)