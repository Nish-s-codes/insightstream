# app/api/routes.py

from fastapi import APIRouter, UploadFile, Request
from fastapi.responses import StreamingResponse
import os
import json
from dotenv import load_dotenv
from groq import AsyncGroq

from app.services.rag import answer_question
from app.core.pipeline import process_document
from app.services.mcp_client import call_tool

load_dotenv()

router = APIRouter()
UPLOAD_DIR = "data/uploads"

# ---------------- EXISTING ENDPOINTS (UNCHANGED) ----------------

@router.post("/upload")
async def upload_file(file: UploadFile):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = process_document(file_path)
    return {"message": "Processed", "result": result}


@router.get("/query")
async def query(q: str):
    return await answer_question(q)


@router.get("/")
def root():
    return {"message": "Streaming RAG system running"}


# ---------------- NEW: AGENT + MEMORY ----------------

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.3-70b-versatile"

# 🔴 SIMPLE MEMORY STORE (can replace with Redis later)
CHAT_MEMORY = {}

SYSTEM_PROMPT = """You are an AI assistant.

Rules:
- Use GitHub tools only when needed
- Use search_pdfs only for document queries
- Do not call unnecessary tools
- Be concise
"""

# 🔴 TOOL SCHEMA (SYNCED WITH MCP)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_pdfs",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_repos",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "path": {"type": "string"}
                },
                "required": ["repo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "file_path": {"type": "string"}
                },
                "required": ["repo", "file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_readme",
            "parameters": {
                "type": "object",
                "properties": {"repo": {"type": "string"}},
                "required": ["repo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "commit_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                    "commit_message": {"type": "string"}
                },
                "required": ["repo", "file_path", "content", "commit_message"]
            }
        }
    }
]


@router.get("/ask")
async def ask(q: str, request: Request, session_id: str = "default"):

    use_sse = "text/event-stream" in request.headers.get("accept", "")

    def fmt(text: str):
        return f"data: {json.dumps({'text': text})}\n\n" if use_sse else text

    # 🔴 LOAD OR INIT MEMORY
    if session_id not in CHAT_MEMORY:
        CHAT_MEMORY[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    messages = CHAT_MEMORY[session_id]

    # add user message
    messages.append({"role": "user", "content": q})

    async def generate():
        try:
            for _ in range(8):
                response = await groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0
                )

                msg = response.choices[0].message

                # ---- TOOL CALL ----
                if msg.tool_calls:
                    messages.append(msg)

                    for tc in msg.tool_calls:
                        name = tc.function.name
                        args = json.loads(tc.function.arguments)

                        yield fmt(f"[Calling: {name}...]")

                        result = await call_tool(name, args, request.app)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result)
                        })

                    continue

                # ---- FINAL RESPONSE ----
                if msg.content:
                    messages.append({
                        "role": "assistant",
                        "content": msg.content
                    })

                    yield fmt(msg.content)
                    break

        except Exception as e:
            yield fmt(f"[ERROR] {str(e)}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream" if use_sse else "text/plain"
    )