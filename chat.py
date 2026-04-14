# chat.py
# python chat.py          → text mode
# python chat.py --voice  → voice mode

import sys
import httpx
import json
import uuid

BASE_URL   = "http://localhost:8000"
SESSION_ID = str(uuid.uuid4())


def run_text_mode():
    print(f"InsightStream | Session: {SESSION_ID[:8]}…\nType 'exit' to quit.\n")
    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            print("Bot: ", end="", flush=True)
            with httpx.stream(
                "GET", f"{BASE_URL}/ask",
                params={"q": query, "session_id": SESSION_ID},
                headers={"accept": "text/event-stream"},
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    decoded = line.decode() if isinstance(line, bytes) else line
                    if decoded.startswith("data: "):
                        try:
                            print(json.loads(decoded[6:]).get("text", ""), end="", flush=True)
                        except json.JSONDecodeError:
                            pass
            print("\n")

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            print(f"[ERROR] {e}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def run_voice_mode():
    import asyncio
    from app.voice.voice_client import run
    print(f"InsightStream — Voice Mode | Session: {SESSION_ID[:8]}…\n")
    asyncio.run(run(SESSION_ID))


if __name__ == "__main__":
    if "--voice" in sys.argv:
        run_voice_mode()
    else:
        run_text_mode()