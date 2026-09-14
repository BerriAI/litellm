import json
import queue
import uuid
from urllib.parse import parse_qs, urlsplit
from typing import Final
from pathlib import Path

import pytest

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import McpPeer, register_mcp
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.mcp.oauth.discovery_cannot_erase_configured_authorization_endpoint")
def test_partial_discovery_and_unrelated_edit_keep_actual_authorization_destination(gateway: Gateway, tmp_path: Path) -> None:
    def discovery(request: Request) -> Reply:
        if request.target.startswith("/configured-authorize"):
            return Reply(body=b'{"synthetic_authorization_endpoint":true}')
        if request.target == "/mcp":
            return Reply(body=b'{"synthetic_resource":true}')
        if request.target.startswith("/.well-known/oauth-protected-resource"):
            return Reply(body=json.dumps({"resource": wire.url + "/mcp", "authorization_servers": [wire.url], "scopes_supported": ["tools.read"]}).encode())
        if request.method == "GET":
            return Reply(body=json.dumps({"issuer": wire.url, "token_endpoint": wire.url + "/discovered-token", "scopes_supported": ["tools.read"]}).encode())
        return Reply(status=401, body=b'{"error":"synthetic OAuth requirement"}')

    with wire_server(discovery) as wire, owned_proxy(gateway, tmp_path, {"LITELLM_MCP_OAUTH_DISCOVERY_ON_STARTUP": "true"}) as candidate, candidate.scenario() as scenario:
        gateway = candidate
        alias: Final = "integration" + uuid.uuid4().hex
        endpoint: Final = wire.url + "/configured-authorize"
        identity: Final = register_mcp(scenario, McpPeer(wire.url + "/mcp", queue.Queue()), alias, auth_type="oauth2", authorization_url=endpoint, token_url=wire.url + "/configured-token", oauth2_flow="authorization_code", credentials={"client_id": "synthetic-oauth-client"})
        discovered = []
        def observed() -> tuple[Request, ...]:
            discovered.extend(wire.drain())
            return tuple(item for item in discovered if item.method == "GET" and ".well-known/" in item.target)
        assert eventually(observed, bool, seconds=10)
        for generation in range(2):
            rows: Final = read_rows('SELECT authorization_url FROM "LiteLLM_MCPServerTable" WHERE server_id=%s', (identity,))
            assert rows == [{"authorization_url": endpoint}]
            response: Final = gateway.request("GET", f"/v1/mcp/server/oauth/{identity}/authorize", params={"redirect_uri": "http://127.0.0.1:8765/callback", "state": "synthetic-state", "code_challenge": "A" * 43, "code_challenge_method": "S256", "response_type": "code"})
            assert response.status_code in (302, 307), response.text
            location: Final = urlsplit(response.headers["location"])
            assert location.scheme + "://" + location.netloc + location.path == endpoint
            query: Final = parse_qs(location.query)
            assert query["client_id"] == ["synthetic-oauth-client"]
            assert query["scope"] == ["tools.read"], "Discovery metadata must be applied before checking endpoint preservation"
            assert query["code_challenge"] == ["A" * 43] and query["code_challenge_method"] == ["S256"]
            selected: Final = gateway.client.get(response.headers["location"])
            assert selected.status_code == 200 and selected.json() == {"synthetic_authorization_endpoint": True}
            if generation == 0:
                updated: Final = gateway.request("PUT", "/v1/mcp/server", {"server_id": identity, "server_name": alias + "renamed"})
                assert updated.status_code == 202, updated.text
