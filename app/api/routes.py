from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse
import os
import asyncio
from typing import AsyncGenerator
from app.services.rag import answer_question
from app.core.pipeline import process_document

router = APIRouter()

UPLOAD_DIR = "data/uploads"
# -------------------- UPLOAD -------------------- #

@router.post("/upload")
async def upload_file(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = process_document(file_path)

    return {
        "message": "Processed",
        "result": result
    }
# -------------------- QUERY -------------------- #

@router.get("/query")
def query(q: str):
    return answer_question(q)
# -------------------- STREAMING -------------------- #

async def stream_response(query: str) -> AsyncGenerator[str, None]:

    yield "[Processing Query]\n"
    await asyncio.sleep(0.5)

    yield "[Fetching Answer...]\n"
    await asyncio.sleep(0.5)

    # RAG call
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
        await asyncio.sleep(0.05)

    yield "\n\n[Completed]\n"

@router.get("/stream")
async def stream(query: str):
    return StreamingResponse(
        stream_response(query),
        media_type="text/event-stream"
    )

# -------------------- ROOT -------------------- #
@router.get("/")
def root():
    return {"message": "Streaming RAG system running"}