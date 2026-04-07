# app/api/streaming.py  — clean version
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.rag import answer_question_stream

router = APIRouter()

@router.get("/stream")
async def stream(query: str):
    return StreamingResponse(
        answer_question_stream(query),
        media_type="text/event-stream"
    )