# app/api/ask.py

import os
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from dotenv import load_dotenv
from app.services.mcp_client import call_tool

load_dotenv()

router = APIRouter()
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "llama-3.3-70b-versatile"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_pdfs",
            "description": "Search uploaded PDF documents for technical or user-specific information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_repos",
            "description": "List GitHub repositories.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a repository.",
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
            "description": "Read file contents from a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "file_path": {"type": "string"}
                },
                "required": ["repo", "file_path"]
            }
        }
    }
]


SYSTEM_PROMPT = """You are an intelligent assistant with access to external tools.

You can choose to call tools when they help answer the question.

Use search_pdfs when:
- The user refers to PDFs or uploaded documents

Use GitHub tools when:
- The user asks about repositories or files

Guidelines:
- Answer directly for general knowledge
- Use tools only when necessary
- After calling a tool, produce a final answer
- Avoid unnecessary repeated tool calls
"""


@router.get("/ask")
async def ask(q: str, request: Request):
    accept = request.headers.get("accept", "")
    use_sse = "text/event-stream" in accept

    def fmt(text: str) -> str:
        return f"data: {json.dumps({'text': text})}\n\n" if use_sse else text

    async def generate():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q}
        ]

        try:
            for _ in range(10):  # ✅ increased from 3 → 10
                response = await groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=1500,
                )

                msg = response.choices[0].message

                if msg.tool_calls:
                    msg.content = None
                    messages.append(msg)

                    for tc in msg.tool_calls:
                        tool_name = tc.function.name

                        try:
                            args = json.loads(tc.function.arguments)
                        except:
                            args = {}

                        yield fmt(f"[Calling {tool_name}...]")

                        result = await call_tool(tool_name, args, request.app)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result)
                        })

                    continue

                if msg.content:
                    yield fmt(msg.content)
                    break

        except Exception as e:
            yield fmt(f"[ERROR] {str(e)}")

    media_type = "text/event-stream" if use_sse else "text/plain"
    return StreamingResponse(generate(), media_type=media_type)