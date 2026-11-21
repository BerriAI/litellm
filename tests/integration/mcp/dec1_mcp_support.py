"""Shared helpers for the `test_dec1_mcp_*` contracts on the MCP gateway.

These files pin what a caller of the MCP gateway sees today, so the move of MCP serving onto the Rust
MCP host (`litellm/rust/crates/gateway-mcp`, mounted by `litellm/rust/crates/gateway`) keeps it. Two
kinds of tests use this module:

The Python proxy contracts use the shared `gateway` fixture or an `owned_proxy` booted from a config the
test writes (`config_with_servers`), plus the peers in `integration._support.mcp`.

The native host contracts boot the standalone `litellm-gateway` binary through the rig in
`tests/integration/configuration/dec1_gwnative_support.py` (another agent's module, imported, never
edited). That binary is installed only for the CI management group, so in the `mcp` group those tests
skip through `gateway_binary()`. They are written here because they describe the MCP host, and they
become live the day the mcp group installs the binary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import INITIALIZE
from pydantic import JsonValue

ACCEPT: Final = "application/json, text/event-stream"
SPEND_ROWS_FOR_KEY: Final = (
    'SELECT request_id, call_type, model, spend, status, metadata, team_id, "user" AS user_id'
    ' FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "startTime"'
)


def digest(key: str) -> str:
    """The hashed form of a virtual key, as stored in `LiteLLM_SpendLogs.api_key` and Redis limiter keys."""
    return sha256(key.encode()).hexdigest()


def base_config() -> dict[str, JsonValue]:
    """The shared integration proxy config, as the starting point for an owned proxy's config."""
    return JSON_OBJECT.validate_python(
        yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    )


def config_with_servers(
    directory: Path, servers: Mapping[str, Mapping[str, object]], **sections: JsonValue
) -> Path:
    """Write `proxy_config.yaml` plus `mcp_servers` (and any extra top-level sections) and return its path."""
    config: Final = base_config()
    config["mcp_servers"] = json.loads(json.dumps(dict(servers)))
    for name, value in sections.items():
        existing = config.get(name)
        config[name] = {**existing, **value} if isinstance(existing, dict) and isinstance(value, dict) else value
    path: Final = directory / "dec1-mcp-config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def server_ids_by_name(gateway: Gateway) -> dict[str, str]:
    """`server_name -> server_id` from `GET /v1/mcp/server`, for config-declared servers whose id the proxy derives."""
    response: Final = gateway.client.get("/v1/mcp/server", headers={"x-litellm-api-key": gateway.key})
    assert response.status_code == 200, response.text
    listed: Final = response.json()
    assert isinstance(listed, list), response.text
    return {str(server["server_name"]): str(server["server_id"]) for server in listed}


def rpc(
    gateway: Gateway | httpx.Client,
    path: str,
    method: str,
    params: Mapping[str, object] | None = None,
    *,
    key: str | None,
    headers: Mapping[str, str] | None = None,
    identity: int = 1,
    bearer: bool = False,
) -> httpx.Response:
    """POST one JSON-RPC request to a streamable HTTP MCP path.

    The Python proxy reads the key from `x-litellm-api-key`; the native host reads `Authorization: Bearer`
    (`bearer=True`)."""
    client: Final[httpx.Client] = getattr(gateway, "client", gateway)
    credential: Final = (
        {} if key is None else ({"Authorization": f"Bearer {key}"} if bearer else {"x-litellm-api-key": key})
    )
    return client.post(
        path,
        json={"jsonrpc": "2.0", "id": identity, "method": method, "params": dict(params or {})},
        headers={"Accept": ACCEPT, **credential, **(headers or {})},
    )


def rpc_body(response: httpx.Response) -> dict[str, JsonValue]:
    """The JSON-RPC envelope from a JSON or single-event SSE response body."""
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        data: Final = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
        assert data, response.text
        return JSON_OBJECT.validate_json(data[-1])
    return JSON_OBJECT.validate_json(response.content)


def rpc_result(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    body: Final = rpc_body(response)
    assert "error" not in body, body
    result: Final = body["result"]
    assert isinstance(result, dict), body
    return result


def rpc_error(response: httpx.Response) -> dict[str, JsonValue]:
    body: Final = rpc_body(response)
    error: Final = body.get("error")
    assert isinstance(error, dict), body
    return error


def listed_names(response: httpx.Response) -> list[str]:
    tools: Final = rpc_result(response)["tools"]
    assert isinstance(tools, list), response.text
    return [str(tool["name"]) for tool in tools if isinstance(tool, dict)]


def first_text(result: Mapping[str, JsonValue]) -> str | None:
    content: Final = result.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return None
    text: Final = content[0].get("text")
    return text if isinstance(text, str) else None


def open_session(
    gateway: Gateway | httpx.Client, path: str, *, key: str, bearer: bool = False
) -> tuple[str, httpx.Response]:
    """Send `initialize` with no session id and return the `mcp-session-id` the server assigned."""
    response: Final = rpc(gateway, path, "initialize", INITIALIZE, key=key, bearer=bearer)
    assert response.status_code == 200, response.text
    session: Final = response.headers.get("mcp-session-id")
    assert session, dict(response.headers)
    return session, response


def spend_rows(key: str, count: int, *, seconds: float = 70) -> list[dict[str, JsonValue]]:
    """Every spend row of `key`, once at least `count` have landed (spend logs are written async)."""
    return eventually(lambda: read_rows(SPEND_ROWS_FOR_KEY, (digest(key),)), lambda rows: len(rows) >= count, seconds=seconds)


def metadata_of(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    metadata: Final = row["metadata"]
    assert isinstance(metadata, dict), row
    return metadata
