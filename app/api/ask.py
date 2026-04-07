# app/api/ask.py
import os
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from dotenv import load_dotenv
from app.services.mcp_client import call_search_pdfs

load_dotenv()
router = APIRouter()
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

@router.get("/ask")
async def ask(q: str, request: Request):

    # check if client wants SSE (browser) or plain stream (curl/terminal)
    accept = request.headers.get("accept", "")
    use_sse = "text/event-stream" in accept

    def format_chunk(text: str) -> str:
        if use_sse:
            return f"data: {text}\n\n"
        return text

    async def generate():
        yield format_chunk("[Searching documents...]\n\n")

        chunks = await call_search_pdfs(q)

        if "No relevant" in chunks or "not initialized" in chunks:
            yield format_chunk(chunks)
            return

        prompt = f"""You are a helpful assistant. Answer the question using only the context below.
If the context doesn't answer the question, say: 'I don't know based on the uploaded documents.'

Context:
{chunks}

Question: {q}
Answer:"""

        stream = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
            stream=True,
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield format_chunk(content)

    media_type = "text/event-stream" if use_sse else "text/plain"
    return StreamingResponse(generate(), media_type=media_type)