# terminal_chat.py
import httpx

BASE_URL = "http://localhost:8000"

def ask(query: str):
    print(f"Bot: ", end="", flush=True)

    with httpx.stream(
        "GET",
        f"{BASE_URL}/ask",
        params={"q": query},
        timeout=60,
        headers={"accept": "text/plain"}
    ) as response:
        for chunk in response.iter_text():
            if "[Searching documents...]" in chunk:
                print("(searching...) ", end="", flush=True)
                continue
            print(chunk, end="", flush=True)

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