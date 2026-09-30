import os
import shutil
import subprocess
from typing import Final

import pytest

IMAGE: Final = os.getenv("LITELLM_IMAGE")
PROBE: Final = """
import asyncio
import base64
import json
import sys
import httpx2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import HTTPException
from litellm.proxy import proxy_server

if sys.argv[1] == "base":
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    message = json.dumps({"expiration_date": "2999-01-01", "user_id": "image-test"}).encode()
    signature = private_key.sign(message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                 salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
    proxy_server._license_check.public_key = private_key.public_key()
    proxy_server._license_check.license_str = base64.b64encode(message + b"." + signature).decode()

async def probe():
    try:
        async with proxy_server.app.router.lifespan_context(proxy_server.app):
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=proxy_server.app),
                                         base_url="http://localhost") as client:
                response = await client.post("/admin/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                            headers={"Accept": "application/json, text/event-stream"})
                print("admin-mcp-status=" + str(response.status_code))
    except HTTPException as exc:
        assert exc.status_code == 403 and "LITELLM_LICENSE" in str(exc.detail)
        assert sys.argv[1] == "none"
        print("admin-mcp-license-required")

asyncio.run(probe())
"""

pytestmark = [
    pytest.mark.skipif(IMAGE is None, reason="requires a built image (set LITELLM_IMAGE)"),
    pytest.mark.skipif(shutil.which("docker") is None, reason="requires the docker CLI"),
]


@pytest.mark.parametrize(
    "enabled,license_mode,expected",
    [("false", "none", "status=404"), ("true", "none", "license-required"), ("true", "base", "status=401")],
)
def test_image_requires_opt_in_and_base_enterprise_license(enabled: str, license_mode: str, expected: str) -> None:
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
            "LITELLM_LICENSE=",
            "--env",
            "LITELLM_LOCAL_MODEL_COST_MAP=true",
            "--env",
            "LITELLM_MASTER_KEY=sk-0123456789abcdef0123456789abcdef",
            "--entrypoint",
            "python",
            IMAGE,
            "-c",
            PROBE,
            license_mode,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0 and f"admin-mcp-{expected}" in result.stdout, (
        f"Admin MCP image probe failed with enabled={enabled}, license={license_mode}\n{result.stdout}\n{result.stderr}"
    )
