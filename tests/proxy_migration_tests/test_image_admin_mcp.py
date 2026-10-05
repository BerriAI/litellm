import os
import shutil
import subprocess
from typing import Final

import pytest
import test_offline_image_migration

offline_postgres: Final = test_offline_image_migration.offline_postgres

IMAGE: Final = os.getenv("LITELLM_IMAGE")
SCHEMA_IMAGE: Final = os.getenv("LITELLM_ADMIN_MCP_SCHEMA_IMAGE")
COMPONENT: Final = os.getenv("LITELLM_IMAGE_COMPONENT", "unified")
PROBE: Final = """
import asyncio
import base64
import importlib
import json
import sys
from typing import Final

import httpx2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import HTTPException
from litellm_admin_mcp.server import create_http_app
from litellm.proxy import proxy_server

module_name: Final = {
    "unified": "litellm.proxy.proxy_server",
    "backend": "backend.main",
    "gateway": "gateway.main",
}[sys.argv[2]]
app: Final = importlib.import_module(module_name).app

if sys.argv[1] == "base":
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    message: Final = json.dumps({"expiration_date": "2999-01-01", "user_id": "image-test"}).encode()
    signature: Final = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    proxy_server._license_check.public_key = private_key.public_key()
    proxy_server._license_check.license_str = base64.b64encode(message + b"." + signature).decode()

async def verify_admin_tools(client: httpx2.AsyncClient) -> None:
    user: Final = await client.post(
        "/user/new",
        headers={"Authorization": "Bearer sk-0123456789abcdef0123456789abcdef"},
        json={"user_id": "image-admin", "user_role": "proxy_admin", "auto_create_key": True},
    )
    assert user.status_code == 200, "Admin provisioning failed: " + str(user.status_code)
    headers: Final = {
        "Authorization": "Bearer " + user.json()["key"],
        "Accept": "application/json, text/event-stream",
    }
    discovery: Final = await client.post(
        "/admin/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )
    assert discovery.status_code == 200, discovery.text
    assert {"create_team", "get_team", "delete_teams"} <= {
        tool["name"] for tool in discovery.json()["result"]["tools"]
    }

    async def call_tool(name: str, arguments: dict[str, object]) -> str:
        response: Final = await client.post(
            "/admin/mcp", headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                  "params": {"name": name, "arguments": arguments}},
        )
        assert response.status_code == 200, response.text
        result: Final = response.json()["result"]
        assert not result.get("isError"), response.text
        return result["content"][0]["text"]

    team_id: Final = "image-admin-mcp-team"
    created: Final = json.loads(await call_tool(
        "create_team", {"body": {"team_id": team_id, "team_alias": team_id, "max_budget": 25}}
    ))
    assert created["team_id"] == team_id and created["max_budget"] == 25, created
    read: Final = json.loads(await call_tool("get_team", {"query": {"team_id": team_id}}))
    assert read["team_info"]["team_id"] == team_id and read["team_info"]["max_budget"] == 25, read
    await call_tool("delete_teams", {"body": {"team_ids": [team_id]}})
    deleted: Final = await client.get("/team/info", headers=headers, params={"team_id": team_id})
    assert deleted.status_code == 404, deleted.text
    print("admin-mcp-tools-ok")

async def probe() -> None:
    try:
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="http://localhost:4000"
            ) as client:
                response: Final = await client.post(
                    "/admin/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    headers={"Accept": "application/json, text/event-stream"},
                )
                print("admin-mcp-status=" + str(response.status_code))
                if sys.argv[3] == "tools":
                    assert response.status_code == 401, response.text
                    await verify_admin_tools(client)
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


def _run_probe(
    enabled: str | None,
    license_mode: str,
    network: str = "none",
    database_url: str | None = None,
) -> subprocess.CompletedProcess[str]:
    assert IMAGE is not None
    enabled_args: Final = () if enabled is None else ("--env", "LITELLM_ENABLE_ADMIN_MCP=" + enabled)
    database_args: Final = () if database_url is None else ("--env", "DATABASE_URL=" + database_url)
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            network,
            "--user",
            "12345:0",
            *enabled_args,
            *database_args,
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
            COMPONENT,
            "tools" if database_url is not None else "visibility",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    "enabled,license_mode,expected",
    [
        (None, "none", "status=404"),
        ("false", "base", "status=404"),
        ("true", "none", "license-required"),
        ("true", "base", "status=404" if COMPONENT == "gateway" else "status=401"),
    ],
)
def test_image_admin_mcp_requires_opt_in_license_and_management_component(
    enabled: str | None, license_mode: str, expected: str
) -> None:
    result: Final = _run_probe(enabled, license_mode)
    assert result.returncode == 0 and f"admin-mcp-{expected}" in result.stdout, (
        f"Admin MCP image probe failed with component={COMPONENT}, enabled={enabled}, license={license_mode}\n"
        f"{result.stdout}\n{result.stderr}"
    )


@pytest.mark.skipif(COMPONENT == "gateway", reason="the gateway excludes management endpoints")
def test_image_admin_mcp_personal_admin_manages_team(offline_postgres: tuple[str, str]) -> None:
    assert SCHEMA_IMAGE is not None, "set LITELLM_ADMIN_MCP_SCHEMA_IMAGE to the matching builder image"
    network, postgres = offline_postgres
    database_url: Final = f"postgresql://postgres:pw@{postgres}:5432/litellm"
    schema: Final = subprocess.run(
        [
            "docker", "run", "--rm", "--network", network,
            "--env", "DATABASE_URL=" + database_url,
            "--env", "HOME=/opt/prisma", "--env", "XDG_CACHE_HOME=/opt/prisma/.cache",
            "--env", "PRISMA_BINARY_CACHE_DIR=/opt/prisma/binaries",
            "--entrypoint", "prisma", SCHEMA_IMAGE,
            "db", "push", "--schema", "/app/schema.prisma", "--skip-generate", "--accept-data-loss",
        ],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert schema.returncode == 0, f"Schema provisioning failed\n{schema.stdout}\n{schema.stderr}"
    result: Final = _run_probe("true", "base", network, database_url)
    assert result.returncode == 0 and "admin-mcp-tools-ok" in result.stdout, (
        f"Admin MCP management failed in {COMPONENT}\n{result.stdout}\n{result.stderr}"
    )
