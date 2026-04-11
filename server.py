"""Writing KB MCP server — reads Markdown from GitHub, hosted on Render."""

import os
import re
from base64 import b64decode

import httpx
from mcp.server.fastmcp import FastMCP
from rank_bm25 import BM25Plus

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


def parse_sections(content: str, filepath: str) -> list[dict]:
    """Split markdown content by ## and ### headers into indexable sections."""
    sections = []
    current_title = "Introduction"
    current_level = 0
    current_content: list[str] = []

    for line in content.splitlines():
        if line.startswith("## "):
            if current_content:
                sections.append({
                    "filepath": filepath,
                    "section_title": current_title,
                    "level": current_level,
                    "content": "\n".join(current_content).strip(),
                })
            current_title = line[3:].strip()
            current_level = 2
            current_content = []
        elif line.startswith("### "):
            if current_content:
                sections.append({
                    "filepath": filepath,
                    "section_title": current_title,
                    "level": current_level,
                    "content": "\n".join(current_content).strip(),
                })
            current_title = line[4:].strip()
            current_level = 3
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append({
            "filepath": filepath,
            "section_title": current_title,
            "level": current_level,
            "content": "\n".join(current_content).strip(),
        })

    return [s for s in sections if s["content"]]


class SearchIndex:
    """BM25-based search index for KB sections."""

    def __init__(self):
        self.documents: list[dict] = []
        self.bm25: BM25Plus | None = None
        self._indexed = False

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def build(self, sections: list[dict]):
        self.documents = sections
        if not sections:
            self._indexed = True
            return
        corpus = [self._tokenize(doc["section_title"] + " " + doc["content"]) for doc in sections]
        self.bm25 = BM25Plus(corpus)
        self._indexed = True

    def search(self, query: str, top_n: int = 5) -> list[dict]:
        if not self._indexed or self.bm25 is None:
            return []
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        results = []
        for idx, score in ranked:
            if score > 0:
                doc = self.documents[idx].copy()
                doc["score"] = score
                results.append(doc)
        return results


_search_index = SearchIndex()


async def _initialize_search_index():
    """Build the search index from all KB files."""
    files = await _list_files()
    all_sections = []
    for f in files:
        resp = await _github_get(f"contents/{f['path']}?ref={BRANCH}")
        if resp.status_code == 200:
            content = b64decode(resp.json()["content"]).decode("utf-8")
            sections = parse_sections(content, f["path"])
            all_sections.extend(sections)
    _search_index.build(all_sections)


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


def _safe_path(path: str) -> str:
    """Validate and normalize a relative path - prevents traversal attacks."""
    normalized = "/".join(p for p in path.split("/") if p)
    if "\\" in normalized or ".." in normalized:
        raise ValueError(f"Invalid path: {path}")
    if not normalized:
        raise ValueError("Empty path")
    return normalized


async def _fetch_file(path: str) -> str | None:
    """Fetch and decode a file from GitHub. Returns None on 404."""
    resp = await _github_get(f"contents/{path}?ref={BRANCH}")
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return b64decode(resp.json()["content"]).decode("utf-8")


@mcp.tool(
    name="kb_list_topics",
    annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False},
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
    name="kb_outline",
    annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True},
)
async def kb_outline(path: str) -> str:
    """List all section headers in a KB article. Use to navigate a file before reading a specific section.

    Args:
        path: Relative path to the article (e.g. 'craft/dialogue').
    """
    try:
        safe_path = _safe_path(path)
    except ValueError as e:
        return str(e)
    normalized = safe_path if safe_path.endswith(".md") else f"{safe_path}.md"
    content = await _fetch_file(normalized)
    if content is None:
        return f"Error: Article not found: {normalized}"

    headers = []
    for line in content.splitlines():
        if line.startswith("## "):
            headers.append(f"## {line[3:].strip()}")
        elif line.startswith("### "):
            headers.append(f"  ### {line[4:].strip()}")

    if not headers:
        return f"{normalized}: no section headers found."
    return f"# {normalized}\n\n" + "\n".join(headers)


@mcp.tool(
    name="kb_read",
    annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True},
)
async def kb_read(path: str, section: str = "") -> str:
    """Read a knowledge base article, optionally filtered to a specific section.

    Args:
        path: Relative path to the article (e.g. 'craft/dialogue').
        section: Optional section title (from kb_outline) to return just that section's content.
    """
    try:
        safe_path = _safe_path(path)
    except ValueError as e:
        return str(e)
    normalized = safe_path if safe_path.endswith(".md") else f"{safe_path}.md"
    content = await _fetch_file(normalized)
    if content is None:
        return f"Error: Article not found: {normalized}"

    if not section:
        return content

    sections = parse_sections(content, normalized)
    needle = section.strip().lower()
    match = next((s for s in sections if s["section_title"].lower() == needle), None)
    if match is None:
        available = ", ".join(f'"{s["section_title"]}"' for s in sections)
        return f'Section "{section}" not found in {normalized}. Available: {available}'

    prefix = "##" if match["level"] == 2 else "###"
    return f"{prefix} {match['section_title']}\n\n{match['content']}"


@mcp.tool(
    name="kb_search",
    annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False},
)
async def kb_search(query: str) -> str:
    """Search all KB articles using BM25 full-text ranking. Returns the most relevant sections with their content.

    Args:
        query: Natural language query (e.g. 'how do I write dialogue in a combat scene').
    """
    if not _search_index._indexed:
        await _initialize_search_index()

    results = _search_index.search(query, top_n=5)

    if not results:
        return "No results found."

    output = []
    for r in results:
        output.append(
            f"### {r['filepath']} — {r['section_title']}\n\n{r['content']}"
        )

    return "\n\n---\n\n".join(output)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
