from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
from typing import AsyncGenerator

from app.services.rag import answer_question

router = APIRouter()

def log_step(func):
    async def wrapper(*args, **kwargs):
        print(f"Running {func.__name__}")
        return await func(*args, **kwargs)
    return wrapper
# -------- CONTEXT MANAGER -------- #

class RequestContext:
    def __enter__(self):
        print("Starting request")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Ending request")
# -------- STREAMING FUNCTION (GENERATOR + ASYNC + TYPE HINT) -------- #

@log_step
async def stream_response(query: str) -> AsyncGenerator[str, None]:
    with RequestContext():

        yield "[Processing Query]\n"
        await asyncio.sleep(0.5)

        yield "[Fetching Answer...]\n"
        await asyncio.sleep(0.5)

        # blocking RAG call
        result = answer_question(query)

        # format result
        if isinstance(result, dict):
            answer = result.get("answer", "")
            confidence = result.get("confidence", "")
            formatted = f"{answer}\n\nConfidence: {confidence}"
        else:
            formatted = str(result)
        # chunk streaming
        for word in formatted.split():
            yield word + " "
            await asyncio.sleep(0.9)

        yield "\n\n[Done]\n"
# -------- ROUTE -------- #

@router.get("/stream")
async def stream(query: str):
    return StreamingResponse(
        stream_response(query),
        media_type="text/event-stream"
    )