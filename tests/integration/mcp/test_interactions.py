import json
import os
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from pydantic import JsonValue

from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


_KEY: Final = "sk-interaction-test"
_PROXY_PYTHONPATH: Final = os.pathsep.join(
    (str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[2]))
)
_META: Final = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": {"name": "continuation-test", "version": "1"},
    "io.modelcontextprotocol/clientCapabilities": {"elicitation": {"form": {}, "url": {}}},
}


def interaction_peer(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(status=405)
    body: Final = json.loads(request.body)
    method: Final = body["method"]
    params: Final = body.get("params", {})
    if method == "server/discover":
        result = {
            "supportedVersions": ["2026-07-28"],
            "capabilities": {"tools": {}, "prompts": {}, "resources": {}},
            "cacheScope": "private",
            "ttlMs": 0,
        }
    elif method == "tools/list":
        result = {
            "tools": [{"name": "confirm", "inputSchema": {"type": "object"}}],
            "cacheScope": "private",
            "ttlMs": 0,
        }
    elif method == "prompts/list":
        result = {"prompts": [{"name": "confirm"}], "cacheScope": "private", "ttlMs": 0}
    elif method == "resources/list":
        result = {"resources": [{"name": "confirm", "uri": "test://confirm"}], "cacheScope": "private", "ttlMs": 0}
    elif method == "resources/templates/list":
        result = {"resourceTemplates": [], "cacheScope": "private", "ttlMs": 0}
    elif not params.get("requestState"):
        return Reply(
            body=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "resultType": "input_required",
                        "requestState": "opaque:" + method,
                        "inputRequests": {
                            "consent": {
                                "method": "elicitation/create",
                                "params": {
                                    "mode": "form",
                                    "message": "Confirm",
                                    "requestedSchema": {"type": "object", "properties": {}},
                                },
                            }
                        },
                    },
                }
            ).encode()
        )
    else:
        assert params["requestState"] == "opaque:" + method
        assert params["inputResponses"] == {"consent": {"action": "accept"}}
        if method == "tools/call":
            result = {"content": [{"type": "text", "text": "confirmed"}], "isError": False}
        elif method == "prompts/get":
            result = {"messages": [{"role": "user", "content": {"type": "text", "text": "confirmed"}}]}
        else:
            assert method == "resources/read"
            result = {"contents": [{"uri": "test://confirm", "text": "confirmed"}]}
    return Reply(
        body=json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"resultType": "complete", **result}}).encode()
    )


def rpc(gateway: Gateway, method: str, params: dict[str, JsonValue], *, status: int = 200) -> dict:
    response: Final = gateway.client.post(
        "/mcp/",
        headers={
            "Authorization": "Bearer " + gateway.key,
            "MCP-Protocol-Version": "2026-07-28",
            "Mcp-Method": method,
            "Mcp-Name": str(params.get("name", params.get("uri", ""))),
            "Accept": "application/json, text/event-stream",
        },
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": {**params, "_meta": _META}},
    )
    assert response.status_code == status, response.text
    return response.json()


@pytest.mark.parametrize("changed_target", [False, True])
def test_continuations_resume_on_another_replica_and_reject_changed_operations(
    tmp_path: Path, changed_target: bool
) -> None:
    with wire_server(interaction_peer) as peer, httpx.Client() as client:
        config: Final = tmp_path / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [],
                    "mcp_servers": {
                        "mrtr": {
                            "url": peer.url,
                            "transport": "http",
                            "protocol_version": "2026-07-28",
                            "allow_elicitation": True,
                        }
                    },
                    "general_settings": {
                        "master_key": _KEY,
                        "store_model_in_db": False,
                        "mcp_advertised_versions": ["2025-11-25", "2026-07-28"],
                    },
                }
            )
        )
        other_config: Final = tmp_path / "other-proxy.yaml"
        other_config.write_text(
            config.read_text().replace(peer.url, peer.url + "/changed-target") if changed_target else config.read_text()
        )
        seed: Final = Gateway(client, _KEY, peer.url)
        environment: Final = {
            "PYTHONPATH": _PROXY_PYTHONPATH,
            "STORE_MODEL_IN_DB": "False",
            "DISABLE_SCHEMA_UPDATE": "true",
            "LITELLM_SALT_KEY": "shared-interaction-test",
        }
        options: Final = {
            "database_setup": (),
            "remove_environment": (
                "DATABASE_URL",
                "DATABASE_URL_READ_REPLICA",
                "LITELLM_LICENSE",
                "LITELLM_LICENSE_PATH",
                "REDIS_URL",
                "REDIS_HOST",
            ),
        }
        with (
            owned_proxy(seed, tmp_path / "a", environment, config=config, **options) as first,
            owned_proxy(seed, tmp_path / "b", environment, config=other_config, **options) as second,
        ):
            for method, params, terminal_field in (
                ("tools/call", {"name": "mrtr-confirm", "arguments": {}}, "content"),
                ("prompts/get", {"name": "mrtr-confirm", "arguments": {}}, "messages"),
                ("resources/read", {"uri": "test://confirm"}, "contents"),
            ):
                initial: Final = rpc(first, method, params)
                assert initial["result"]["resultType"] == "input_required", initial
                state: Final = initial["result"]["requestState"]
                assert state.startswith("mcp_state_v1."), initial
                retry: Final = {**params, "requestState": state, "inputResponses": {"consent": {"action": "accept"}}}
                if changed_target:
                    peer.drain()
                    refused: Final = rpc(second, method, retry, status=400)
                    assert refused["error"]["code"] == -32602, refused
                    assert peer.drain() == (), "Changed target must reject before upstream dispatch"
                    continue
                completed: Final = rpc(second, method, retry)
                assert "confirmed" in json.dumps(completed["result"][terminal_field]), completed
                assert rpc(first, method, retry) == completed
                peer.drain()
                changed: Final = {
                    **retry,
                    **({"uri": "test://other"} if method == "resources/read" else {"name": "mrtr-other"}),
                }
                rejected: Final = rpc(second, method, changed, status=400)
                assert rejected["error"]["code"] == -32602, rejected
                assert peer.drain() == (), "Rejected continuation must not contact the upstream"


def test_continuation_reauthenticates_caller_and_rechecks_revoked_permissions(tmp_path: Path) -> None:
    auth_module: Final = tmp_path / "interaction_auth.py"
    auth_module.write_text(
        "from pathlib import Path\n"
        "from fastapi import HTTPException, Request\n"
        "from litellm.proxy._types import UserAPIKeyAuth\n"
        "async def authenticate(request: Request, api_key: str) -> UserAPIKeyAuth:\n"
        "    if api_key != 'sk-interaction-test':\n"
        "        raise HTTPException(status_code=401, detail='Unknown test caller')\n"
        "    return UserAPIKeyAuth.model_validate_json(Path(__file__).with_suffix('.json').read_text())\n"
    )
    identity: Final = {
        "user_id": "alice",
        "team_id": "team-a",
        "user_role": "internal_user",
        "object_permission": {"object_permission_id": "test-permission", "mcp_servers": ["interaction-server"]},
    }
    auth_state: Final = auth_module.with_suffix(".json")
    auth_state.write_text(json.dumps(identity))
    with wire_server(interaction_peer) as peer, wire_server(interaction_peer) as other_peer, httpx.Client() as client:
        config: Final = tmp_path / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [],
                    "mcp_servers": {
                        "mrtr": {
                            "server_id": "interaction-server",
                            "url": peer.url,
                            "transport": "http",
                            "protocol_version": "2026-07-28",
                            "allow_elicitation": True,
                        },
                        "other": {
                            "server_id": "other-server",
                            "url": other_peer.url,
                            "transport": "http",
                            "protocol_version": "2026-07-28",
                            "allow_elicitation": True,
                        },
                    },
                    "general_settings": {
                        "master_key": _KEY,
                        "custom_auth": "interaction_auth.authenticate",
                        "store_model_in_db": False,
                        "mcp_advertised_versions": ["2025-11-25", "2026-07-28"],
                    },
                }
            )
        )
        with owned_proxy(
            Gateway(client, _KEY, peer.url),
            tmp_path / "proxy",
            {
                "PYTHONPATH": _PROXY_PYTHONPATH,
                "STORE_MODEL_IN_DB": "False",
                "DISABLE_SCHEMA_UPDATE": "true",
                "LITELLM_SALT_KEY": "caller-test-salt",
            },
            config=config,
            database_setup=(),
            remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "REDIS_URL", "REDIS_HOST"),
        ) as gateway:
            params: Final = {"name": "mrtr-confirm", "arguments": {}}
            initial: Final = rpc(gateway, "tools/call", params)
            assert initial["result"]["resultType"] == "input_required", initial
            retry: Final = {
                **params,
                "requestState": initial["result"]["requestState"],
                "inputResponses": {"consent": {"action": "accept"}},
            }
            for changed in ({"user_id": "bob"}, {"team_id": "team-b"}):
                auth_state.write_text(json.dumps({**identity, **changed}))
                peer.drain()
                rejected: Final = rpc(gateway, "tools/call", retry, status=400)
                assert rejected["error"]["code"] == -32602, rejected
                assert peer.drain() == (), "Caller-bound state must reject before contacting upstream"
            auth_state.write_text(json.dumps(identity))
            resumed: Final = rpc(gateway, "tools/call", retry)
            assert resumed["result"]["content"] == [{"type": "text", "text": "confirmed"}], resumed
            auth_state.write_text(
                json.dumps(
                    {
                        **identity,
                        "object_permission": {
                            "object_permission_id": "test-permission",
                            "mcp_servers": ["denied-server"],
                        },
                    }
                )
            )
            peer.drain()
            revoked: Final = rpc(gateway, "tools/call", retry, status=403)
            assert revoked["detail"] == "MCP continuation target is no longer authorized", revoked
            assert peer.drain() == (), "Revoked access must reject a valid continuation before upstream dispatch"

            auth_state.write_text(json.dumps(identity))
            resource: Final = rpc(gateway, "resources/read", {"uri": "test://confirm"})
            assert resource["result"]["resultType"] == "input_required", resource
            auth_state.write_text(
                json.dumps(
                    {
                        **identity,
                        "object_permission": {
                            "object_permission_id": "test-permission",
                            "mcp_servers": ["other-server"],
                        },
                    }
                )
            )
            peer.drain()
            other_peer.drain()
            response: Final = gateway.client.post(
                "/mcp/",
                headers={
                    "Authorization": "Bearer " + gateway.key,
                    "MCP-Protocol-Version": "2026-07-28",
                    "Mcp-Method": "resources/read",
                    "Mcp-Name": "test://confirm",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "resources/read",
                    "params": {
                        "uri": "test://confirm",
                        "_meta": _META,
                        "requestState": resource["result"]["requestState"],
                        "inputResponses": {"consent": {"action": "accept"}},
                    },
                },
            )
            other_requests: Final = tuple(
                (json.loads(request.body)["method"], json.loads(request.body).get("params", {}).get("requestState"))
                for request in other_peer.drain()
            )
            assert other_requests == (), "Continuation must never send upstream state to another authorized server"
            assert peer.drain() == (), "Revoked original target must not receive a retry"
            assert response.status_code >= 400 or "error" in response.json(), response.text

            auth_state.write_text(
                json.dumps(
                    {
                        **identity,
                        "object_permission": {
                            "object_permission_id": "test-permission",
                            "mcp_servers": ["interaction-server", "other-server"],
                        },
                    }
                )
            )
            completed: Final = rpc(
                gateway,
                "resources/read",
                {
                    "uri": "test://confirm",
                    "requestState": resource["result"]["requestState"],
                    "inputResponses": {"consent": {"action": "accept"}},
                },
            )
            assert completed["result"]["contents"] == [{"uri": "test://confirm", "text": "confirmed"}], completed
            assert other_peer.drain() == (), "Expanded access must keep the continuation on its original target"


def test_missing_continuation_key_reports_configuration_for_each_carrier(tmp_path: Path) -> None:
    with wire_server(interaction_peer) as peer, httpx.Client() as client:
        config: Final = tmp_path / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [],
                    "mcp_servers": {
                        "mrtr": {
                            "url": peer.url,
                            "transport": "http",
                            "protocol_version": "2026-07-28",
                            "allow_elicitation": True,
                        }
                    },
                    "general_settings": {
                        "master_key": _KEY,
                        "store_model_in_db": False,
                        "mcp_advertised_versions": ["2025-11-25", "2026-07-28"],
                    },
                }
            )
        )
        with owned_proxy(
            Gateway(client, _KEY, peer.url),
            tmp_path / "proxy",
            {
                "PYTHONPATH": _PROXY_PYTHONPATH,
                "STORE_MODEL_IN_DB": "False",
                "DISABLE_SCHEMA_UPDATE": "true",
                "LITELLM_SALT_KEY": "",
            },
            config=config,
            database_setup=(),
            remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "REDIS_URL", "REDIS_HOST"),
        ) as gateway:
            for method, params in (
                ("tools/call", {"name": "mrtr-confirm", "arguments": {}}),
                ("prompts/get", {"name": "mrtr-confirm", "arguments": {}}),
                ("resources/read", {"uri": "test://confirm"}),
            ):
                response: Final = rpc(gateway, method, params, status=400)
                assert response["error"]["code"] == -32602, response
                assert "LITELLM_SALT_KEY" in response["error"]["message"], response
