# app/core/tool_loader.py
#
# Loads tools from all active MCP sessions once, then caches them.
# Only exposes the tools in ALLOWED_TOOLS to the LLM — keeps the
# tool list small without a second LLM routing call.

ALLOWED_TOOLS: set[str] = {
    # ── Custom GitHub (always available) ──────────────────────────────────────
    "list_repos",
    "list_files",
    "read_file",
    "get_readme",
    "commit_file",
    "delete_file",
    # ── Official GitHub ────────────────────────────────────────────────────────
    "get_file_contents",
    "create_or_update_file",
    "push_files",
    "create_branch",
    "list_branches",
    "list_commits",
    "list_issues",
    "get_issue",
    "create_issue",
    "update_issue",
    "add_issue_comment",
    "search_issues",
    "list_pull_requests",
    "get_pull_request",
    "create_pull_request",
    "search_code",
    # ── RAG ───────────────────────────────────────────────────────────────────
    "search_pdfs",
}

FULL_TOOL_MAP: dict[str, dict] = {}


def slim_schema(schema: dict) -> dict:
    """
    Strip verbose 'description' fields from parameter schemas.
    Keeps type/required/properties structure so the LLM still knows
    what args to pass — just removes the wordy explanations that
    bloat token usage.
    """
    if not isinstance(schema, dict):
        return schema

    result = {}
    for key, val in schema.items():
        if key == "description":
            continue  # drop parameter-level descriptions
        elif isinstance(val, dict):
            result[key] = slim_schema(val)
        elif isinstance(val, list):
            result[key] = [slim_schema(i) if isinstance(i, dict) else i for i in val]
        else:
            result[key] = val
    return result


async def load_all_tools(app) -> list[dict]:
    global FULL_TOOL_MAP

    if FULL_TOOL_MAP:
        return list(FULL_TOOL_MAP.values())

    sessions = {
        "mcp_github_session":  getattr(app.state, "mcp_github_session",  None),
        "mcp_github_official": getattr(app.state, "mcp_github_official", None),
        "mcp_rag_session":     getattr(app.state, "mcp_rag_session",     None),
    }

    for label, session in sessions.items():
        if session is None:
            print(f"[WARN] Session '{label}' not available — skipping.")
            continue

        try:
            tools = (await session.list_tools()).tools
        except Exception as e:
            print(f"[ERROR] Could not list tools from '{label}': {e}")
            continue

        loaded = 0
        for t in tools:
            if t.name not in ALLOWED_TOOLS:
                continue

            FULL_TOOL_MAP[t.name] = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,       # keep tool-level description
                    "parameters": slim_schema(t.inputSchema),  # strip param descriptions
                },
            }
            loaded += 1

        print(f"[OK] Loaded {loaded} allowed tools from '{label}'.")

    return list(FULL_TOOL_MAP.values())


def invalidate_tool_cache() -> None:
    global FULL_TOOL_MAP
    FULL_TOOL_MAP.clear()
    print("[INFO] Tool cache cleared.")