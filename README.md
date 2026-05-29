# Writing KB MCP

An MCP server for creative writing knowledge and learning.

## Purpose

This isn't about having an LLM write for you. It's about **learning to write better** using LLMs as a tool.

Think of it as having instant access to writing craft knowledge—structure, style, dialogue, pacing—so you can:

- Understand *why* something works, not just that it does
- Learn the mechanics behind effective prose
- Get targeted guidance on specific craft challenges
- Build your own writing intuition faster

## What's Here

- `craft/` - Technique-specific guidance (dialogue, voice, POV, etc.)
- `style/` - Prose-level craft (rhythm, word choice, sentence variety)
- `structure/` - Scene and story architecture

## Using This MCP

Connect any MCP-compatible client to query writing knowledge, explore craft concepts, or work through specific writing challenges.

The goal: make you a more skilled, intentional writer—not replace your voice.

## Running the Server

Two entrypoints, both backed by the same tools and resources:

### stdio (local)

Spawned per-client; re-parses all KB files on startup so edits are picked up immediately.

```bash
uv run python server_stdio.py
```

Add to Claude Code:

```bash
claude mcp add -s user writing-kb -- uv run --directory /Users/jw/code/writing-kb python server_stdio.py
```

### Streamable HTTP (Docker container)

Runs as a long-lived container exposing MCP over HTTP on port 8880 — the same
pattern as the Cognee and Viking knowledge-base MCPs.

```bash
docker compose up -d --build   # or: make up
```

- Health check: `curl http://localhost:8880/health`
- MCP endpoint: `http://localhost:8880/mcp`

KB content (`craft/`, `style/`, `structure/`) is bind-mounted, so `kb_read`,
`kb_list_topics`, and resources reflect edits live. `kb_search` builds its index
at startup — run `docker compose restart` (or `make restart`) to pick up edits
in search results.

Register with Claude Code:

```bash
claude mcp add -s user -t http writing-kb http://localhost:8880/mcp
```

### Streamable HTTP (remote)

Used for the hosted deployment on Render.

```bash
uv run python server.py   # listens on $PORT (default 10000)
```
