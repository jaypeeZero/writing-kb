# writing-kb

MCP server exposing writing-craft KB content (`craft/`, `style/`, `structure/`, `examples/`).

## Deploy model (read before changing content/config)
The live MCP helper runs as the **Docker container** at `localhost:8880`. How changes propagate:

- **Content edits** (`*.md` in mounted dirs) — live via bind-mount, BUT `kb_search` caches its index at startup → `docker compose restart` to see them in search.
- **New content dir** — must be added in 3 places: `CONTENT_DIRS` (`writing_kb/config.py`), a `COPY` line in `Dockerfile`, and a bind-mount in `docker-compose.yml`. Then **`docker compose up -d --build`**.
- **Code edits** (`writing_kb/`, `server.py`) — baked into the image, NOT mounted → require **`docker compose up -d --build`** (a plain `restart` does nothing).

## examples/ conventions
- One passage per `##` section so search returns each as a first-class hit.
- Embed the **actual verified text** (`>` quote) — never a pointer to a book. Verify wording via web before committing.
- Keep these files purely examples (no meta/notes — they pollute search). Roadmap notes: `~/code/plans/writing-kb/examples-bank/`.

## Tests
`uv run pytest`
