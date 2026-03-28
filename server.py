"""Writing KB MCP server — reads Markdown from GitHub, hosted on Render."""

import os
from base64 import b64decode

import httpx
from mcp.server.fastmcp import FastMCP

REPO = os.environ.get("KB_REPO", "yourusername/writing-kb")
BRANCH = os.environ.get("KB_BRANCH", "main")
TOKEN = os.environ.get("KB_GITHUB_TOKEN", "")
PORT = int(os.environ.get("PORT", "10000"))

API = f"https://api.github.com/repos/{REPO}"
HEADERS = {"Accept": "application/vnd.github.v3+json"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

EXCLUDED = {
    "server.py", "pyproject.toml", "uv.lock", "Dockerfile",
    "README.md", "PLAN.md", "LICENSE", ".gitignore",
}

mcp = FastMCP("writing_kb", host="0.0.0.0", port=PORT)


async def _github_get(path: str) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await client.get(f"{API}/{path}", headers=HEADERS, timeout=10)


async def _list_files(subpath: str = "") -> list[dict]:
    """Recursively list .md files from the GitHub repo."""
    resp = await _github_get(f"contents/{subpath}?ref={BRANCH}")
    resp.raise_for_status()
    files: list[dict] = []
    for item in resp.json():
        if item["name"] in EXCLUDED or item["name"].startswith("."):
            continue
        if item["type"] == "file" and item["name"].endswith(".md"):
            files.append(item)
        elif item["type"] == "dir":
            files.extend(await _list_files(item["path"]))
    return files


@mcp.tool(
    name="kb_list_topics",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
)
async def kb_list_topics() -> str:
    """List all knowledge base articles grouped by category."""
    files = await _list_files()
    by_category: dict[str, list[str]] = {}
    for f in files:
        parts = f["path"].split("/")
        category = parts[0] if len(parts) > 1 else "general"
        name = parts[-1].removesuffix(".md")
        by_category.setdefault(category, []).append(name)
    return "\n\n".join(
        f"## {cat}\n" + "\n".join(f"- {t}" for t in sorted(topics))
        for cat, topics in sorted(by_category.items())
    )


@mcp.tool(
    name="kb_read",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
)
async def kb_read(path: str) -> str:
    """Read a knowledge base article by relative path (e.g. 'craft/dialogue')."""
    normalized = path if path.endswith(".md") else f"{path}.md"
    resp = await _github_get(f"contents/{normalized}?ref={BRANCH}")
    if resp.status_code == 404:
        return f"Error: Article not found: {normalized}"
    resp.raise_for_status()
    return b64decode(resp.json()["content"]).decode("utf-8")


@mcp.tool(
    name="kb_search",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": False,
    },
)
async def kb_search(query: str) -> str:
    """Search all articles for a keyword or phrase (case-insensitive)."""
    files = await _list_files()
    q = query.lower()
    results: list[str] = []
    for f in files:
        resp = await _github_get(f"contents/{f['path']}?ref={BRANCH}")
        if resp.status_code != 200:
            continue
        text = b64decode(resp.json()["content"]).decode("utf-8")
        if q in text.lower():
            matches = [
                l.strip() for l in text.splitlines() if q in l.lower()
            ][:3]
            results.append(f"### {f['path']}\n" + "\n".join(matches))
    return "\n\n".join(results) if results else "No results found."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
