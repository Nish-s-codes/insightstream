# mcp/mcp_github.py
# Uses httpx (async-safe) instead of requests.
# This is your reliable custom GitHub layer.

import sys
import os
import base64
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

mcp = FastMCP("GitHub-Tool")

GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


async def gh_get(url: str) -> dict | list:
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=HEADERS)
        if r.status_code >= 400:
            return {"error": r.text, "status_code": r.status_code}
        return r.json()


async def gh_put(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.put(url, headers=HEADERS, json=payload)
        r.raise_for_status()
        return r.json()


async def gh_delete(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.request("DELETE", url, headers=HEADERS, json=payload)
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def list_repos() -> str:
    """List all GitHub repositories belonging to the authenticated user."""
    data = await gh_get("https://api.github.com/user/repos?per_page=100")
    if isinstance(data, dict) and "error" in data:
        return f"Error fetching repos: {data['error']}"
    return "\n".join(f"- {r['name']}" for r in data)


@mcp.tool()
async def list_files(repo: str, path: str = "") -> str:
    """
    List files and folders inside a GitHub repository directory.

    :param repo: Repository name (e.g. 'my-project').
    :param path: Subdirectory path (e.g. 'src/components'). Leave empty for root.
    """
    path = path.strip("/")
    url  = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{path}"
    data = await gh_get(url)

    if isinstance(data, dict) and "error" in data:
        return f"Error: Could not find path '{path}' in repo '{repo}'. {data['error']}"
    if isinstance(data, list):
        return "\n".join(
            f"{'[DIR] ' if i['type'] == 'dir' else '[FILE]'} {i['name']}"
            for i in data
        )
    if isinstance(data, dict):
        return f"[FILE] {data['name']}"
    return "Not found."


@mcp.tool()
async def read_file(repo: str, file_path: str) -> str:
    """
    Read the text content of a file from a GitHub repository.

    :param repo: Repository name (e.g. 'my-project').
    :param file_path: Full path to the file (e.g. 'app/main.py').
    """
    url  = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{file_path}"
    data = await gh_get(url)

    if isinstance(data, dict) and "error" in data:
        return f"Error: Could not read '{file_path}' in '{repo}'. {data['error']}"

    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return f"File: {file_path}\n\n{content}"


@mcp.tool()
async def get_readme(repo: str) -> str:
    """
    Fetch the README from a GitHub repository.

    :param repo: Repository name (e.g. 'my-project').
    """
    url  = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/readme"
    data = await gh_get(url)

    if isinstance(data, dict) and "error" in data:
        return f"Error: No README found for '{repo}'. {data['error']}"

    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return f"README.md — {repo}\n\n{content}"


@mcp.tool()
async def commit_file(
    repo: str,
    file_path: str,
    content: str,
    commit_message: str = "Update file",
) -> str:
    """
    Create or update a file in a GitHub repository.

    :param repo: Repository name.
    :param file_path: Path where the file should be saved (e.g. 'docs/guide.md').
    :param content: Full text content to write into the file.
    :param commit_message: Short description for the git commit.
    """
    url      = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{file_path}"
    existing = await gh_get(url)
    sha      = None
    if isinstance(existing, dict) and "error" not in existing:
        sha = existing.get("sha")

    payload = {
        "message": commit_message,
        "content": base64.b64encode(content.encode()).decode(),
    }
    if sha:
        payload["sha"] = sha

    try:
        await gh_put(url, payload)
    except httpx.HTTPStatusError as e:
        return f"Error committing file: {e}"

    action = "Updated" if sha else "Created"
    return f"{action} '{file_path}' in '{repo}' — commit: \"{commit_message}\""


@mcp.tool()
async def delete_file(
    repo: str,
    file_path: str,
    commit_message: str = "Delete file",
) -> str:
    """
    Delete a file from a GitHub repository.
    Only call this after the user has explicitly confirmed the deletion.

    :param repo: Repository name.
    :param file_path: Path of the file to delete (e.g. 'README.md').
    :param commit_message: Short description for the git commit.
    """
    url      = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{file_path}"
    existing = await gh_get(url)

    if isinstance(existing, dict) and "error" in existing:
        return f"Error: Could not find '{file_path}' in '{repo}'. {existing['error']}"

    sha = existing.get("sha")
    if not sha:
        return f"Error: Could not retrieve SHA for '{file_path}' — cannot delete."

    try:
        await gh_delete(url, {"message": commit_message, "sha": sha})
    except httpx.HTTPStatusError as e:
        return f"Error deleting file: {e}"

    return f"Deleted '{file_path}' from '{repo}'."


if __name__ == "__main__":
    mcp.run()