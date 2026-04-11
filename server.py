"""Writing KB MCP server — reads Markdown from local filesystem, hosted on Render."""

import writing_kb.tools  # noqa: F401 — registers @mcp.tool and @mcp.resource decorators
from writing_kb.config import PORT, mcp

if __name__ == "__main__":
    import uvicorn
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    app = mcp.streamable_http_app()

    async def health(request):
        return JSONResponse({"status": "ok"})

    app.router.routes.append(Route("/health", health))

    uvicorn.run(app, host="0.0.0.0", port=PORT)
