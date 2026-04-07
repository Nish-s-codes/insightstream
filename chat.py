# chat.py

import httpx
import json

BASE_URL = "http://localhost:8000"


def ask(query: str):
    print("Bot: ", end="", flush=True)

    with httpx.stream(
        "GET",
        f"{BASE_URL}/ask",
        params={"q": query},
        headers={"accept": "text/event-stream"},
        timeout=60
    ) as response:

        for line in response.iter_lines():
            if not line:
                continue

            decoded = line  # ✅ FIXED

            if decoded.startswith("data: "):
                payload = decoded[6:]

                try:
                    data = json.loads(payload)
                    text = data.get("text", "")
                    print(text, end="", flush=True)
                except:
                    print(payload, end="", flush=True)

    print("\n")


def main():
    print("InsightStream Terminal")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            ask(query)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()