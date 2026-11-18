from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from mcp.server.mcpserver import MCPServer

server: Final = MCPServer("vault")


@server.tool()
def read_secret() -> str:
    """Return the current secret value."""
    return Path(os.environ["SECRET_FILE"]).read_text()


if __name__ == "__main__":
    server.run()
