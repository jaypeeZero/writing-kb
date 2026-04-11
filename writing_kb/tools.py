from urllib.parse import unquote

from mcp.server.fastmcp.resources.types import FunctionResource

import writing_kb.content as content_module
import writing_kb.search as search_module
from writing_kb.config import mcp
from writing_kb.content import list_md_files, read_file, safe_path, parse_sections


@mcp.resource("writing-kb://{path}")
def kb_resource(path: str) -> str:
    """Read a KB article as an MCP resource (e.g. writing-kb://craft/dialogue.md)."""
    path = unquote(path)
    try:
        normalized = safe_path(path)
    except ValueError as e:
        return str(e)
    normalized = normalized if normalized.endswith(".md") else f"{normalized}.md"
    content = read_file(normalized)
    return content or f"Not found: {normalized}"


def _register_kb_resources():
    """Register each KB file as a concrete resource so resources/list is populated."""
    for p in list_md_files():
        rel = str(p.relative_to(content_module.KB_DIR))
        uri = f"writing-kb://{rel}"

        def _make_reader(path: str):
            def reader() -> str:
                content = read_file(path)
                return content or f"Not found: {path}"
            return reader

        mcp.add_resource(FunctionResource.from_function(
            fn=_make_reader(rel),
            uri=uri,
            name=rel,
            mime_type="text/plain",
        ))


_register_kb_resources()


@mcp.tool(
    name="kb_list_topics",
    annotations={"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True},
)
async def kb_list_topics(path: str = "") -> str:
    """List knowledge base articles or the section outline of a specific article.

    Args:
        path: Optional. If omitted, lists all articles grouped by category.
              If provided (e.g. 'craft/dialogue'), returns the section headers for that article.
    """
    if not path:
        by_category: dict[str, list[str]] = {}
        for p in list_md_files():
            parts = p.relative_to(content_module.KB_DIR).parts
            category = parts[0] if len(parts) > 1 else "general"
            name = parts[-1].removesuffix(".md")
            by_category.setdefault(category, []).append(name)
        return "\n\n".join(
            f"## {cat}\n" + "\n".join(f"- {t}" for t in sorted(topics))
            for cat, topics in sorted(by_category.items())
        )

    try:
        normalized = safe_path(path)
    except ValueError as e:
        return str(e)
    normalized = normalized if normalized.endswith(".md") else f"{normalized}.md"
    content = read_file(normalized)
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
        section: Optional section title (from kb_list_topics) to return just that section's content.
    """
    try:
        normalized = safe_path(path)
    except ValueError as e:
        return str(e)
    normalized = normalized if normalized.endswith(".md") else f"{normalized}.md"
    content = read_file(normalized)
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
    """Search all KB articles and return the most relevant sections.

    Uses BM25 (full text + title), token coverage, and bigram proximity combined
    via Reciprocal Rank Fusion for higher-quality results than any single method.

    Args:
        query: Natural language query (e.g. 'how do I write dialogue in a combat scene').
    """
    if not search_module._search_index._indexed:
        search_module.initialize_search_index()

    results = search_module._search_index.search(query, top_n=5)

    if not results:
        return "No results found."

    return "\n\n---\n\n".join(
        f"### {r['filepath']} — {r['section_title']}\n\n{r['content']}"
        for r in results
    )
