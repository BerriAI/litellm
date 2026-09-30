import os
import shutil
import subprocess
from typing import Final

import pytest

IMAGE: Final = os.getenv("LITELLM_IMAGE")
PROBE: Final = """
import asyncio
import httpx2
from litellm.proxy.proxy_server import app

async def probe():
    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://localhost") as client:
            response = await client.post("/admin/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                        headers={"Accept": "application/json, text/event-stream"})
            print("admin-mcp-status=" + str(response.status_code))

asyncio.run(probe())
"""

pytestmark = [
    pytest.mark.skipif(IMAGE is None, reason="requires a built image (set LITELLM_IMAGE)"),
    pytest.mark.skipif(shutil.which("docker") is None, reason="requires the docker CLI"),
]


@pytest.mark.parametrize("enabled,status", [("false", 404), ("true", 401)])
def test_image_serves_admin_mcp_only_when_enabled(enabled: str, status: int) -> None:
    assert IMAGE is not None
    result: Final = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "12345:0",
            "--env",
            "LITELLM_ENABLE_ADMIN_MCP=" + enabled,
            "--env",
            "LITELLM_LOCAL_MODEL_COST_MAP=true",
            "--env",
            "LITELLM_MASTER_KEY=sk-0123456789abcdef0123456789abcdef",
            "--entrypoint",
            "python",
            IMAGE,
            "-c",
            PROBE,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0 and f"admin-mcp-status={status}" in result.stdout, (
        f"Admin MCP image probe failed with enabled={enabled}\n{result.stdout}\n{result.stderr}"
    )
