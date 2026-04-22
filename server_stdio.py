"""Writing KB MCP server — stdio transport for local use.

Re-parses all KB files on startup so the latest content is always served.
"""

import writing_kb.tools  # noqa: F401 — registers @mcp.tool and @mcp.resource decorators
from writing_kb.config import mcp
from writing_kb.search import initialize_search_index


if __name__ == "__main__":
    initialize_search_index()
    mcp.run()
