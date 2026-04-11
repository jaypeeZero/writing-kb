import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

PORT = int(os.environ.get("PORT", "10000"))

KB_DIR = Path(__file__).parent.parent

CONTENT_DIRS = {"craft", "style", "structure"}

mcp = FastMCP("writing_kb", host="0.0.0.0", port=PORT)
