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

# Note: If you still get 429 Rate Limit errors, consider using llama-3.1-70b-versatile
MODEL_NAME = "llama-3.3-70b-versatile"

# ---------------- TOOLS (Fixed Schema) ----------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_pdfs",
            "description": "Search through uploaded PDF documents for technical or specific information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search keywords"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_repos",
            "description": "List all GitHub repositories for the authenticated user.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files and directories in a specific GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "The name of the repository"}
                },
                "required": ["repo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text content of a specific file from a GitHub repository.",
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
            "name": "commit_file",
            "description": "Create or update a file in a GitHub repository.",
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

# ---------------- AGENT SYSTEM PROMPT (Pure Logic) ----------------
SYSTEM_PROMPT = """You are a highly efficient AI Assistant with access to GitHub and PDF tools.

DECISION LOGIC:
1. GREETINGS & SMALL TALK: If the user says 'hi', 'hola', 'who are you', or asks general questions (e.g., 'what is the capital of France'), respond DIRECTLY using your internal knowledge. Do NOT call any tools.
2. GITHUB TASKS: Use GitHub tools ONLY if the user explicitly mentions repositories, files, or commits.
3. PDF TASKS: Use 'search_pdfs' ONLY if the user asks about content from uploaded documents or technical documentation.

RULES:
- Be concise.
- Never call more than one tool at a time unless the task requires multiple steps (e.g., list files then read one).
- If you call a tool, do not explain what you are doing first. Just call it.
- If you have the answer, stop and provide it.
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
            for _ in range(5):  # Safety limit for tool turns
                response = await groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=1500,
                    parallel_tool_calls=False
                )

                msg = response.choices[0].message

                # 1. Handle Tool Calls
                if msg.tool_calls:
                    msg.content = None # Required for API consistency
                    messages.append(msg)

                    for tc in msg.tool_calls:
                        t_name = tc.function.name
                        t_args = json.loads(tc.function.arguments)

                        yield fmt(f"[Calling: {t_name}...]")

                        result = await call_tool(t_name, t_args, request.app)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result)
                        })
                    continue  # Let the LLM process the tool result

                # 2. Handle Final Text Response
                if msg.content:
                    yield fmt(msg.content)
                    break

        except Exception as e:
            yield fmt(f"[ERROR] {str(e)}")

    media_type = "text/event-stream" if use_sse else "text/plain"
    return StreamingResponse(generate(), media_type=media_type)