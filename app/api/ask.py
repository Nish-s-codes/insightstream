# app/api/ask.py

import os
import re
import json
from collections import Counter
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from dotenv import load_dotenv

from app.services.mcp_client import call_tool
from app.core.tool_loader import load_all_tools

load_dotenv()

router = APIRouter()
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# llama-3.3-70b-versatile: 128k context window, high token throughput on Groq
MODEL = "llama-3.3-70b-versatile"

CHAT_MEMORY: dict[str, list] = {}
PENDING_DELETES: dict[str, dict] = {}
_GITHUB_USER = os.getenv("GITHUB_USERNAME", "")

SYSTEM_PROMPT = f"""You are InsightStream, a GitHub assistant for user '{_GITHUB_USER}'.

- Never ask for credentials. Never pretend to be a generic AI.
- Always use the provided tools to fetch real data. Never guess or fabricate.
- For file contents: use read_file with the repo and file_path.
- For file listing: use list_files with path="". Ignore venv/, .git/, node_modules/, __pycache__/.
- For repo listing: use list_repos.
- For README: use get_readme.
- After getting tool results, respond in plain conversational text. No code blocks unless showing actual file content.
- delete_file: always confirm with user first."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_parse_args(raw: str) -> dict:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def is_delete_confirmation(q: str) -> bool:
    return any(x in q.lower().strip() for x in ["yes", "confirm", "delete it"])


def get_or_create_session(session_id: str) -> list:
    if session_id not in CHAT_MEMORY:
        CHAT_MEMORY[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return CHAT_MEMORY[session_id]


def trim_memory(messages: list) -> None:
    if len(messages) > 9:
        messages[:] = [messages[0]] + messages[-8:]


def fmt(text: str, use_sse: bool) -> str:
    return f"data: {json.dumps({'text': text})}\n\n" if use_sse else text


def detect_repetition(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 5]
    if len(lines) < 8:
        return False
    return any(c > 3 for c in Counter(lines).values())


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("/ask")
async def ask(q: str, request: Request, session_id: str = "default"):

    use_sse  = "text/event-stream" in request.headers.get("accept", "")
    messages = get_or_create_session(session_id)

    # ── Pending delete confirmation ───────────────────────────────────────────
    if session_id in PENDING_DELETES and is_delete_confirmation(q):
        pending = PENDING_DELETES.pop(session_id)

        async def do_delete():
            result = await call_tool(pending["tool"], pending["args"], request.app)
            messages.append({"role": "assistant", "content": result})
            trim_memory(messages)
            yield fmt(result, use_sse)

        return StreamingResponse(
            do_delete(),
            media_type="text/event-stream" if use_sse else "text/plain",
        )

    messages.append({"role": "user", "content": q})

    async def generate():
        try:
            available_tools = await load_all_tools(request.app)

            # Agentic loop — max 5 iterations to prevent infinite loops
            for _ in range(5):
                response = await groq_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=available_tools if available_tools else None,
                    tool_choice="auto" if available_tools else None,
                    temperature=0,
                    max_tokens=2048,
                )

                msg = response.choices[0].message

                # No tool call — stream the final answer
                if not msg.tool_calls:
                    reply = msg.content or ""
                    messages.append({"role": "assistant", "content": reply})
                    trim_memory(messages)
                    yield fmt(reply, use_sse)
                    return

                # Tool call(s) — execute and loop back
                messages.append(msg)

                for tc in msg.tool_calls:
                    args = safe_parse_args(tc.function.arguments)
                    tool_name = tc.function.name

                    # Guard: delete needs confirmation
                    if tool_name == "delete_file":
                        PENDING_DELETES[session_id] = {"tool": tool_name, "args": args}
                        confirm_msg = (
                            f"Are you sure you want to delete "
                            f"'{args.get('file_path', '?')}' "
                            f"from '{args.get('repo', '?')}'? "
                            f"Reply 'yes' to confirm."
                        )
                        messages.append({"role": "assistant", "content": confirm_msg})
                        trim_memory(messages)
                        yield fmt(confirm_msg, use_sse)
                        return

                    result = await call_tool(tool_name, args, request.app)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": result,
                    })

            # Fallback if loop limit hit
            yield fmt("I wasn't able to complete that in time. Please try rephrasing.", use_sse)

        except Exception as e:
            yield fmt(f"[ERROR] {e}", use_sse)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream" if use_sse else "text/plain",
    )