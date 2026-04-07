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

mcp_cleanup = None
mcp_session = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_cleanup, mcp_session

    print("Starting MCP server process...")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.abspath("mcp/mcp_rag.py")],
    )

    mcp_cleanup = stdio_client(server_params)
    read, write = await mcp_cleanup.__aenter__()

    mcp_session = ClientSession(read, write)
    await mcp_session.__aenter__()
    await mcp_session.initialize()

    print("MCP server ready and warm!")

    yield

    print("Shutting down MCP server...")
    await mcp_session.__aexit__(None, None, None)
    await mcp_cleanup.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)

app.include_router(main_router)
app.include_router(stream_router)
app.include_router(ask_router)