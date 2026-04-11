from pathlib import Path

from writing_kb.config import KB_DIR, CONTENT_DIRS


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


def safe_path(path: str) -> str:
    """Validate and normalize a relative path — prevents traversal attacks."""
    normalized = "/".join(p for p in path.split("/") if p)
    if "\\" in normalized or ".." in normalized:
        raise ValueError(f"Invalid path: {path}")
    if not normalized:
        raise ValueError("Empty path")
    return normalized


def list_md_files() -> list[Path]:
    """List .md files strictly within the allowed content directories."""
    files = []
    for d in CONTENT_DIRS:
        p = KB_DIR / d
        if p.is_dir():
            files.extend(p.rglob("*.md"))
    return files


def read_file(path: str) -> str | None:
    """Read a markdown file from disk. Returns None if not found or outside CONTENT_DIRS."""
    full = (KB_DIR / path).resolve()
    try:
        rel = full.relative_to(KB_DIR.resolve())
    except ValueError:
        return None
    if rel.parts[0] not in CONTENT_DIRS:
        return None
    return full.read_text("utf-8") if full.is_file() else None
