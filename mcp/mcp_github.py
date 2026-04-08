# mcp/mcp_github.py
import sys
import os
import base64
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# FIXED PATH BUG
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

mcp = FastMCP("GitHub-Tool")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
# ---------------- HELPERS ----------------

def gh_get(url: str):
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def gh_put(url: str, payload: dict):
    r = requests.put(url, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()

def gh_delete(url: str, payload: dict):
    r = requests.delete(url, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()

# ---------------- READ TOOLS ----------------

@mcp.tool()
def list_repos() -> str:
    """List all repositories"""
    try:
        data = gh_get("https://api.github.com/user/repos?per_page=50&visibility=all")

        if not data:
            return "No repositories found."

        return "\n".join(
            f"- {r['name']} ({'private' if r['private'] else 'public'})"
            for r in data
        )

    except Exception as e:
        return f"Error fetching repos: {str(e)}"

@mcp.tool()
def list_files(repo: str, path: str = "") -> str:
    """List files and folders in repo"""

    path = path.strip("/")
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{path}"

    try:
        data = gh_get(url)

        if isinstance(data, list):
            return "\n".join(
                f"{'[DIR]' if i['type']=='dir' else '[FILE]'} {i['name']}"
                for i in data
            )

        elif isinstance(data, dict):
            return f"[FILE] {data['name']} (size: {data.get('size', 0)} bytes)"

        return "Path not found."

    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def read_file(repo: str, file_path: str) -> str:
    """Read file content"""

    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{file_path}"

    try:
        data = gh_get(url)

        if "content" not in data:
            return "Could not read file."

        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")

        return f"{file_path}\n\n{content}"

    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
def get_readme(repo: str) -> str:
    """Get README content"""

    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/readme"

    try:
        data = gh_get(url)

        if "content" not in data:
            return "No README found."

        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")

    except Exception as e:
        return f"Error fetching README: {str(e)}"

@mcp.tool()
def read_repo_for_summary(repo: str, max_files: int = 8) -> str:
    """Read important repo files"""

    try:
        root = gh_get(f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/")
    except Exception as e:
        return f"Error: {str(e)}"

    code_ext = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".go", ".rs", ".cpp", ".c",
        ".cs", ".rb", ".php", ".html", ".css"
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

                if ext in code_ext:
                    try:
                        data = gh_get(item["url"])
                        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                        collected.append(f"{item['path']}\n{content[:1500]}")
                    except:
                        pass

            elif item["type"] == "dir" and depth < 2:
                try:
                    sub = gh_get(item["url"])
                    collect(sub, depth + 1)
                except:
                    pass

    collect(root)

    if not collected:
        return "No readable files found."

    return "\n\n---\n\n".join(collected)

# ---------------- WRITE TOOLS ----------------

@mcp.tool()
def commit_file(repo: str, file_path: str, content: str, commit_message: str) -> str:
    """Create or update file"""

    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{file_path}"

    try:
        existing = None
        try:
            existing = gh_get(url)
        except:
            pass

        sha = existing.get("sha") if existing else None

        encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")

        payload = {
            "message": commit_message,
            "content": encoded,
        }

        if sha:
            payload["sha"] = sha

        gh_put(url, payload)

        return f"{'Updated' if sha else 'Created'} {file_path}"

    except Exception as e:
        return f"Commit failed: {str(e)}"

@mcp.tool()
def delete_file(
    repo: str,
    file_path: str,
    commit_message: str,
    confirmed_once: bool = False,
    confirmed_twice: bool = False,
) -> str:
    """Delete file (double confirmation)"""
    if not confirmed_once:
        return f"Confirm deletion of {file_path} (1/2)"

    if not confirmed_twice:
        return f"Confirm deletion of {file_path} (2/2)"

    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}/contents/{file_path}"

    try:
        existing = gh_get(url)
        sha = existing.get("sha")

        gh_delete(url, {"message": commit_message, "sha": sha})

        return f"Deleted {file_path}"

    except Exception as e:
        return f"Delete failed: {str(e)}"

# ---------------- RUN ----------------
if __name__ == "__main__":
    mcp.run()