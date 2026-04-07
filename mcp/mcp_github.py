# mcp/mcp_github.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import base64
import json
import requests
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("GitHub-Tool")

GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept":        "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"   # fixed
}

def gh_get(url: str) -> dict:
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def gh_put(url: str, payload: dict) -> dict:
    r = requests.put(url, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()

def gh_delete(url: str, payload: dict) -> dict:
    r = requests.delete(url, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()

# ─────────────────────────────────────────────
# READ TOOLS
# ─────────────────────────────────────────────

@mcp.tool()
def list_repos() -> str:
    """List all repositories for the authenticated GitHub user."""
    data = gh_get(f"https://api.github.com/user/repos?per_page=50&visibility=all")
    repos = [f"- {r['name']} ({'private' if r['private'] else 'public'})" for r in data]
    return "\n".join(repos) if repos else "No repositories found."


@mcp.tool()
def list_files(repo: str, path: str = "") -> str:
    """List files and folders in a GitHub repository at a given path."""
    url  = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{path}"
    data = gh_get(url)
    if isinstance(data, list):
        items = [f"{'[DIR]' if i['type']=='dir' else '[FILE]'} {i['name']}" for i in data]
        return "\n".join(items)
    return "Path not found or is a file."


@mcp.tool()
def read_file(repo: str, file_path: str) -> str:
    """Read the contents of a file from a GitHub repository."""
    url  = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{file_path}"
    data = gh_get(url)
    if "content" not in data:
        return "Could not read file content."
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return f"File: {file_path}\n\n{content}"


@mcp.tool()
def get_readme(repo: str) -> str:
    """Fetch the README of a GitHub repository."""
    url  = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/readme"
    data = gh_get(url)
    if "content" not in data:
        return "No README found."
    return base64.b64decode(data["content"]).decode("utf-8", errors="replace")


@mcp.tool()
def read_repo_for_summary(repo: str, max_files: int = 8) -> str:
    """
    Read key source files from a repo so the LLM can understand what it does
    and write a README for it. Reads up to max_files code files.
    """
    try:
        root_items = gh_get(
            f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/"
        )
    except Exception as e:
        return f"Could not access repo: {e}"

    # collect files recursively up to max_files
    code_extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
        ".rs", ".cpp", ".c", ".cs", ".rb", ".php", ".html", ".css"
    }

    collected = []

    def collect(items, depth=0):
        if len(collected) >= max_files:
            return
        for item in items:
            if len(collected) >= max_files:
                return
            if item["type"] == "file":
                ext = os.path.splitext(item["name"])[1].lower()
                if ext in code_extensions:
                    try:
                        data    = gh_get(item["url"])
                        content = base64.b64decode(
                            data["content"]
                        ).decode("utf-8", errors="replace")
                        collected.append(f"### {item['path']}\n{content[:2000]}")
                    except Exception:
                        pass
            elif item["type"] == "dir" and depth < 2:
                try:
                    sub = gh_get(item["url"])
                    collect(sub, depth + 1)
                except Exception:
                    pass

    collect(root_items)

    if not collected:
        return "No readable source files found in this repository."

    return "\n\n---\n\n".join(collected)


# ─────────────────────────────────────────────
# WRITE TOOLS
# ─────────────────────────────────────────────

@mcp.tool()
def commit_file(repo: str, file_path: str, content: str, commit_message: str) -> str:
    """
    Create or update a file in a GitHub repository.
    Use this to commit a generated README or any file changes.
    """
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{file_path}"

    # check if file already exists (need its SHA to update)
    sha = None
    try:
        existing = gh_get(url)
        sha = existing.get("sha")
    except Exception:
        pass  # file doesn't exist yet, that's fine

    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    payload = {
        "message": commit_message,
        "content": encoded,
    }
    if sha:
        payload["sha"] = sha

    try:
        gh_put(url, payload)
        action = "Updated" if sha else "Created"
        return f"{action} '{file_path}' in '{repo}' successfully."
    except Exception as e:
        return f"Failed to commit file: {e}"


# ─────────────────────────────────────────────
# DELETE TOOL (double confirmation enforced)
# ─────────────────────────────────────────────

@mcp.tool()
def delete_file(
    repo: str,
    file_path: str,
    commit_message: str,
    confirmed_once: bool = False,
    confirmed_twice: bool = False
) -> str:
    """
    Delete a file from a GitHub repository.
    IMPORTANT: Both confirmed_once AND confirmed_twice must be True to proceed.
    If either is False, do NOT delete — instead ask the user to confirm again.
    This tool must never be called with both set to True without explicit user approval.
    """
    if not confirmed_once:
        return (
            f"CONFIRMATION REQUIRED (1/2): You are about to delete '{file_path}' "
            f"from '{repo}'. Are you sure? Please confirm to proceed."
        )

    if not confirmed_twice:
        return (
            f"CONFIRMATION REQUIRED (2/2): This will PERMANENTLY delete '{file_path}' "
            f"from '{repo}'. This cannot be undone. Please confirm a second time."
        )

    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{file_path}"

    try:
        existing = gh_get(url)
        sha      = existing.get("sha")
    except Exception as e:
        return f"Could not find file to delete: {e}"

    try:
        gh_delete(url, {"message": commit_message, "sha": sha})
        return f"Deleted '{file_path}' from '{repo}' successfully."
    except Exception as e:
        return f"Failed to delete file: {e}"


if __name__ == "__main__":
    mcp.run()