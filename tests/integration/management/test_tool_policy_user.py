import json
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from typing import Final, NamedTuple

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows, scratch_database, write_rows
from integration._support.mcp import echo_tool, register_mcp, scripted_peer, tool_calls
from integration._support.process import owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from jwt.algorithms import RSAAlgorithm
from pydantic import BaseModel, JsonValue

from litellm.repositories.chunked_in import IN_LIST_CHUNK_SIZE

AUDIENCE: Final = "litellm-integration"
KEY_ID: Final = "integration-signing-key"
CLIENT_CLAIM: Final = "client_id"


def _tool_call_request(model: str, tool_name: str) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "tool policy user control"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "integration tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


def _forget_tool(tool_name: str) -> None:
    write_rows('DELETE FROM "LiteLLM_ToolTable" WHERE tool_name = %s', (tool_name,))


def _discovered_tool(gateway: Gateway, tool_name: str) -> dict[str, JsonValue]:
    def rows() -> list[dict[str, JsonValue]]:
        tools: Final = gateway.get("/v1/tool/list")["tools"]
        assert isinstance(tools, list)
        return [object_value(tool) for tool in tools if object_value(tool)["tool_name"] == tool_name]

    return eventually(rows, lambda found: len(found) == 1, seconds=70)[0]


def test_tool_list_reports_the_user_that_owns_the_discovering_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        alias: Final = "integration-alias-" + uuid.uuid4().hex
        user: Final = scenario.user(user_alias=alias, user_email=f"{alias}@integration.example")
        key: Final = scenario.key(user_id=user, models=[model])
        tool_name: Final = "integration_tool_" + uuid.uuid4().hex
        scenario.cleanups.callback(_forget_tool, tool_name)
        response: Final = gateway.request("POST", "/v1/chat/completions", _tool_call_request(model, tool_name), key=key)
        assert response.status_code == 200, response.text
        tool: Final = _discovered_tool(gateway, tool_name)
        assert tool["key_hash"] == sha256(key.encode()).hexdigest(), tool
        assert tool["user"] == {"user_id": user, "user_email": f"{alias}@integration.example", "user_alias": alias}, (
            tool
        )


def test_tool_list_reports_no_user_for_a_key_without_an_owner(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        tool_name: Final = "integration_tool_" + uuid.uuid4().hex
        scenario.cleanups.callback(_forget_tool, tool_name)
        response: Final = gateway.request("POST", "/v1/chat/completions", _tool_call_request(model, tool_name), key=key)
        assert response.status_code == 200, response.text
        tool: Final = _discovered_tool(gateway, tool_name)
        assert tool["key_hash"] == sha256(key.encode()).hexdigest(), tool
        assert tool["user"] is None, tool


JWT_SETTINGS: Final[Mapping[str, JsonValue]] = {
    "enable_jwt_auth": True,
    "litellm_jwtauth": {
        "user_id_jwt_field": "sub",
        "user_email_jwt_field": "email",
        "user_id_upsert": True,
        "virtual_key_claim_field": CLIENT_CLAIM,
        "unregistered_jwt_client_behavior": "auto_register",
    },
}


def _proxy_config(
    directory: Path, model: str, upstream_url: str, general_settings: Mapping[str, JsonValue] = JWT_SETTINGS
) -> Path:
    config: Final = directory / "tool_policy_user_config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": "openai/" + model,
                            "api_base": upstream_url + "/v1",
                            "api_key": "sk-upstream",
                        },
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                    **general_settings,
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


def _signed_token(private_key: rsa.RSAPrivateKey, user_id: str, email: str, client_id: str) -> str:
    now: Final = int(time.time())
    return jwt.encode(
        {"sub": user_id, "email": email, CLIENT_CLAIM: client_id, "aud": AUDIENCE, "iat": now, "exp": now + 300},
        private_key,
        algorithm="RS256",
        headers={"kid": KEY_ID},
    )


def _forget_auto_registered_client(client_id: str, user_id: str) -> None:
    write_rows(
        'DELETE FROM "LiteLLM_VerificationToken" WHERE token IN '
        '(SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_value = %s)',
        (client_id,),
    )
    write_rows('DELETE FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_value = %s', (client_id,))
    write_rows('DELETE FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))


def test_tool_list_reports_the_jwt_user_behind_an_auto_registered_key(gateway: Gateway, tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwks: Final = json.dumps({"keys": [{**public_jwk, "kid": KEY_ID, "use": "sig", "alg": "RS256"}]}).encode()

    def respond(request: Request) -> Reply:
        assert request.target == "/jwks", request
        return Reply(body=jwks)

    model: Final = "integration-jwt-" + uuid.uuid4().hex
    with wire_server(respond) as issuer:
        config: Final = _proxy_config(tmp_path, model, gateway.upstream_url)
        overrides: Final = {"JWT_PUBLIC_KEY_URL": issuer.url + "/jwks", "JWT_AUDIENCE": AUDIENCE}
        with owned_proxy(gateway, tmp_path, overrides, config=config) as candidate, candidate.scenario() as scenario:
            user: Final = "integration-jwt-user-" + uuid.uuid4().hex
            email: Final = f"{user}@integration.example"
            client_id: Final = "integration-client-" + uuid.uuid4().hex
            tool_name: Final = "integration_tool_" + uuid.uuid4().hex
            scenario.cleanups.callback(_forget_tool, tool_name)
            scenario.cleanups.callback(_forget_auto_registered_client, client_id, user)
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                _tool_call_request(model, tool_name),
                key=_signed_token(private_key, user, email, client_id),
            )
            assert response.status_code == 200, response.text
            mapped: Final = read_rows(
                'SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_name = %s AND jwt_claim_value = %s',
                (CLIENT_CLAIM, client_id),
            )
            assert len(mapped) == 1, mapped
            assert read_rows(
                'SELECT user_id FROM "LiteLLM_VerificationToken" WHERE token = %s', (mapped[0]["token"],)
            ) == [{"user_id": user}]
            tool: Final = _discovered_tool(candidate, tool_name)
            assert tool["key_hash"] == mapped[0]["token"], tool
            assert tool["user"] == {"user_id": user, "user_email": email, "user_alias": None}, tool


def _owner(user_id: str, email: str | None, alias: str | None) -> dict[str, JsonValue]:
    return {"user_id": user_id, "user_email": email, "user_alias": alias}


def _discover(gateway: Gateway, cleanups: ExitStack, model: str, key: str) -> str:
    tool_name: Final = "integration_tool_" + uuid.uuid4().hex
    cleanups.callback(_forget_tool, tool_name)
    response: Final = gateway.request("POST", "/v1/chat/completions", _tool_call_request(model, tool_name), key=key)
    assert response.status_code == 200, response.text
    return tool_name


class Owned(NamedTuple):
    tool_name: str
    model: str
    key: str
    owner: dict[str, JsonValue]


def _owned_tool(gateway: Gateway, scenario: Scenario, alias: str | None = None) -> Owned:
    """A discovered tool, the model and key that discovered it, and the owner the tool routes must report."""
    model: Final = scenario.model()
    email: Final = f"{uuid.uuid4().hex}@integration.example"
    fields: Final[Mapping[str, JsonValue]] = {"user_alias": alias} if alias else {}
    user: Final = scenario.user(user_email=email, **fields)
    key: Final = scenario.key(user_id=user, models=[model])
    return Owned(_discover(gateway, scenario.cleanups, model, key), model, key, _owner(user, email, alias))


def _single(gateway: Gateway, tool_name: str) -> dict[str, JsonValue]:
    return gateway.get(f"/v1/tool/{tool_name}")


def _detail_tool(gateway: Gateway, tool_name: str) -> dict[str, JsonValue]:
    return object_value(gateway.get(f"/v1/tool/{tool_name}/detail")["tool"])


def _listed_tools(gateway: Gateway, prefix: str, params: Mapping[str, str] | None = None) -> list[dict[str, JsonValue]]:
    tools: Final = gateway.get("/v1/tool/list", params)["tools"]
    assert isinstance(tools, list)
    return [object_value(tool) for tool in tools if str(object_value(tool)["tool_name"]).startswith(prefix)]


def test_tool_get_reports_the_owner_and_null_for_an_unowned_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        owned, model, _, owner = _owned_tool(gateway, scenario, alias="alias-" + uuid.uuid4().hex)
        unowned: Final = _discover(gateway, scenario.cleanups, model, scenario.key(models=[model]))
        assert _discovered_tool(gateway, owned)["user"] == owner
        _discovered_tool(gateway, unowned)
        assert _single(gateway, owned)["user"] == owner
        assert _single(gateway, unowned)["user"] is None


def test_tool_detail_carries_the_owner_inside_the_tool(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        tool_name, _, _, owner = _owned_tool(gateway, scenario, alias="alias-" + uuid.uuid4().hex)
        assert _discovered_tool(gateway, tool_name)["user"] == owner
        assert _detail_tool(gateway, tool_name)["user"] == owner


def test_filtered_tool_list_keeps_the_owner(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        tool_name, _, _, owner = _owned_tool(gateway, scenario)
        listed: Final = _discovered_tool(gateway, tool_name)
        assert listed["input_policy"] == "untrusted", listed
        filtered: Final = _listed_tools(gateway, tool_name, {"input_policy": "untrusted"})
        assert [tool["user"] for tool in filtered] == [owner], filtered
        assert _listed_tools(gateway, tool_name, {"input_policy": "blocked"}) == []


def test_two_tools_discovered_by_the_same_key_share_the_owner(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        first, model, key, owner = _owned_tool(gateway, scenario)
        second: Final = _discover(gateway, scenario.cleanups, model, key)
        assert [_discovered_tool(gateway, name)["user"] for name in (first, second)] == [owner, owner]


def test_owner_without_alias_or_email_reports_only_the_user_id(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user()
        key: Final = scenario.key(user_id=user, models=[model])
        tool_name: Final = _discover(gateway, scenario.cleanups, model, key)
        assert _discovered_tool(gateway, tool_name)["user"] == _owner(user, None, None)


def test_missing_tool_is_404_on_get_and_detail(gateway: Gateway) -> None:
    missing: Final = "integration_missing_" + uuid.uuid4().hex
    for path in (f"/v1/tool/{missing}", f"/v1/tool/{missing}/detail"):
        response: Final = gateway.request("GET", path)
        assert response.status_code == 404, response.text
        assert response.json() == {"detail": f"Tool '{missing}' not found"}


def test_non_admin_keys_are_rejected_on_every_tool_read_route(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        tool_name, _, _, owner = _owned_tool(gateway, scenario)
        assert _discovered_tool(gateway, tool_name)["user"] == owner
        internal: Final = scenario.key(user_id=scenario.user(user_role="internal_user"))
        plain: Final = scenario.key()
        for key in (internal, plain):
            for path in ("/v1/tool/list", f"/v1/tool/{tool_name}", f"/v1/tool/{tool_name}/detail"):
                response: Final = gateway.request("GET", path, key=key)
                assert response.status_code == 401, (path, response.text)
                assert string_value(owner["user_email"]) not in response.text, response.text


def test_unauthenticated_tool_reads_are_rejected(gateway: Gateway) -> None:
    for path in ("/v1/tool/list", "/v1/tool/some_tool", "/v1/tool/some_tool/detail"):
        response: Final = gateway.client.get(path)
        assert response.status_code == 401, (path, response.text)
        assert "No api key passed in" in response.text, response.text


def test_deleting_the_owner_keeps_the_tool_row_without_a_user(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = uuid.uuid4().hex
        gateway.post("/user/new", {"user_id": user, "auto_create_key": False})
        key: Final = string_value(gateway.post("/key/generate", {"user_id": user, "models": [model]})["key"])
        tool_name: Final = _discover(gateway, scenario.cleanups, model, key)
        assert _discovered_tool(gateway, tool_name)["user"] == _owner(user, None, None)
        deleted: Final = gateway.request("POST", "/user/delete", {"user_ids": [user]})
        assert deleted.status_code == 200, deleted.text
        assert read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE user_id = %s', (user,)) == []
        tool: Final = _discovered_tool(gateway, tool_name)
        assert tool["user"] is None, tool
        assert tool["key_hash"] == sha256(key.encode()).hexdigest()
        assert _single(gateway, tool_name)["user"] is None


def test_deleting_the_key_keeps_the_tool_row_without_a_user(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user()
        key: Final = string_value(gateway.post("/key/generate", {"user_id": user, "models": [model]})["key"])
        tool_name: Final = _discover(gateway, scenario.cleanups, model, key)
        assert _discovered_tool(gateway, tool_name)["user"] == _owner(user, None, None)
        scenario.delete_key(key)
        tool: Final = _discovered_tool(gateway, tool_name)
        assert tool["user"] is None, tool
        assert tool["key_hash"] == sha256(key.encode()).hexdigest()


def test_tool_row_without_a_key_hash_is_listed_without_a_user(gateway: Gateway) -> None:
    tool_name: Final = "integration_tool_" + uuid.uuid4().hex
    with ExitStack() as cleanups:
        cleanups.callback(_forget_tool, tool_name)
        write_rows(
            'INSERT INTO "LiteLLM_ToolTable" (tool_id, tool_name) VALUES (gen_random_uuid()::text, %s)', (tool_name,)
        )
        tool: Final = _discovered_tool(gateway, tool_name)
        assert tool["key_hash"] is None, tool
        assert tool["user"] is None, tool
        assert _single(gateway, tool_name)["user"] is None


def test_tool_row_with_an_unknown_key_hash_is_listed_without_a_user(gateway: Gateway) -> None:
    tool_name: Final = "integration_tool_" + uuid.uuid4().hex
    key_hash: Final = "integration-unknown-" + uuid.uuid4().hex
    with ExitStack() as cleanups:
        cleanups.callback(_forget_tool, tool_name)
        write_rows(
            'INSERT INTO "LiteLLM_ToolTable" (tool_id, tool_name, key_hash) VALUES (gen_random_uuid()::text, %s, %s)',
            (tool_name, key_hash),
        )
        tool: Final = _discovered_tool(gateway, tool_name)
        assert tool["key_hash"] == key_hash, tool
        assert tool["user"] is None, tool


def _forget_prefixed(prefix: str) -> None:
    write_rows('DELETE FROM "LiteLLM_ToolTable" WHERE tool_name LIKE %s', (prefix + "%",))
    write_rows('DELETE FROM "LiteLLM_VerificationToken" WHERE token LIKE %s', (prefix + "%",))


def test_owner_lookup_spans_more_keys_than_one_chunk(gateway: Gateway) -> None:
    prefix: Final = "integration_chunk_" + uuid.uuid4().hex + "_"
    count: Final = IN_LIST_CHUNK_SIZE + 1
    with gateway.scenario() as scenario:
        user: Final = scenario.user(user_alias="chunk-owner-" + uuid.uuid4().hex)
        scenario.cleanups.callback(_forget_prefixed, prefix)
        write_rows(
            'INSERT INTO "LiteLLM_VerificationToken" (token, user_id) '
            "SELECT %s || g, %s FROM generate_series(1, %s::int) AS g",
            (prefix, user, str(count)),
        )
        write_rows(
            'INSERT INTO "LiteLLM_ToolTable" (tool_id, tool_name, key_hash) '
            "SELECT gen_random_uuid()::text, %s || g, %s || g FROM generate_series(1, %s::int) AS g",
            (prefix, prefix, str(count)),
        )
        listed: Final = _listed_tools(gateway, prefix)
        assert len(listed) == count, len(listed)
        owners: Final = {json.dumps(tool["user"], sort_keys=True) for tool in listed}
        assert len(owners) == 1, owners
        assert object_value(listed[0]["user"])["user_id"] == user, listed[0]


def test_repeated_tool_list_reads_are_identical_and_leave_rows_unchanged(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        tool_name, _, _, _ = _owned_tool(gateway, scenario)
        first: Final = _discovered_tool(gateway, tool_name)
        before: Final = read_rows(
            'SELECT tool_name, key_hash, call_count, updated_at::text FROM "LiteLLM_ToolTable" WHERE tool_name = %s',
            (tool_name,),
        )
        second: Final = _discovered_tool(gateway, tool_name)
        after: Final = read_rows(
            'SELECT tool_name, key_hash, call_count, updated_at::text FROM "LiteLLM_ToolTable" WHERE tool_name = %s',
            (tool_name,),
        )
        assert first == second, (first, second)
        assert before == after and len(before) == 1, (before, after)


def test_tool_list_total_matches_the_rows_in_postgres(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        tool_name, _, _, _ = _owned_tool(gateway, scenario)
        _discovered_tool(gateway, tool_name)
        body: Final = gateway.get("/v1/tool/list")
        tools: Final = body["tools"]
        assert isinstance(tools, list)
        names: Final = sorted(str(object_value(tool)["tool_name"]) for tool in tools)
        stored: Final = sorted(
            str(row["tool_name"]) for row in read_rows('SELECT tool_name FROM "LiteLLM_ToolTable"', ())
        )
        assert body["total"] == len(tools) == len(stored), body["total"]
        assert names == stored


def test_concurrent_tool_reads_on_two_workers_stay_consistent_during_discovery(
    gateway: Gateway, tmp_path: Path
) -> None:
    model: Final = "integration-workers-" + uuid.uuid4().hex
    config: Final = _proxy_config(tmp_path, model, gateway.upstream_url, {})
    with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
        candidate: Final = owned.gateway
        with candidate.scenario() as scenario:
            alias: Final = "burst-owner-" + uuid.uuid4().hex
            user: Final = scenario.user(user_alias=alias)
            key: Final = scenario.key(user_id=user, models=[model])
            steady: Final = _discover(candidate, scenario.cleanups, model, key)
            assert _discovered_tool(candidate, steady)["user"] == _owner(user, None, alias)
            paths: Final = tuple(
                ("/v1/tool/list", f"/v1/tool/{steady}", f"/v1/tool/{steady}/detail")[index % 3] for index in range(40)
            )

            def read(index: int) -> tuple[int, dict[str, JsonValue], str | None]:
                burst: Final = _discover(candidate, scenario.cleanups, model, key) if index == 20 else None
                response: Final = candidate.request("GET", paths[index])
                assert response.status_code == 200, (paths[index], response.text)
                return index, JSON_OBJECT.validate_json(response.content), burst

            with ThreadPoolExecutor(max_workers=16) as pool:
                results: Final = tuple(pool.map(read, range(40)))
            for index, body, _ in results:
                tool: Final = (
                    next(object_value(t) for t in body["tools"] if object_value(t)["tool_name"] == steady)
                    if paths[index].endswith("/list")
                    else object_value(body["tool"])
                    if paths[index].endswith("/detail")
                    else body
                )
                assert tool["user"] == _owner(user, None, alias), (paths[index], tool)
            burst: Final = next(name for _, _, name in results if name)
            assert _discovered_tool(candidate, burst)["user"] == _owner(user, None, alias)


def test_owner_lookup_failure_keeps_tools_listed_without_a_user(gateway: Gateway, tmp_path: Path) -> None:
    model: Final = "integration-fault-" + uuid.uuid4().hex
    config: Final = _proxy_config(tmp_path, model, gateway.upstream_url, {})
    with (
        scratch_database() as database_url,
        owned_proxy(
            gateway,
            tmp_path,
            {"DATABASE_URL": database_url},
            config=config,
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as candidate,
    ):
        alias: Final = "fault-owner-" + uuid.uuid4().hex
        user: Final = string_value(
            candidate.post("/user/new", {"user_alias": alias, "auto_create_key": False})["user_id"]
        )
        key: Final = string_value(candidate.post("/key/generate", {"user_id": user, "models": [model]})["key"])
        with ExitStack() as cleanups:
            tool_name: Final = _discover(candidate, cleanups, model, key)
            cleanups.pop_all()
        assert _discovered_tool(candidate, tool_name)["user"] == _owner(user, None, alias)
        write_rows('ALTER TABLE "LiteLLM_UserTable" RENAME TO "LiteLLM_UserTable_away"', (), database_url=database_url)
        try:
            degraded: Final = _discovered_tool(candidate, tool_name)
            assert degraded["user"] is None, degraded
            assert degraded["key_hash"] == sha256(key.encode()).hexdigest(), degraded
            assert _single(candidate, tool_name)["user"] is None
        finally:
            write_rows(
                'ALTER TABLE "LiteLLM_UserTable_away" RENAME TO "LiteLLM_UserTable"', (), database_url=database_url
            )
        assert _discovered_tool(candidate, tool_name)["user"] == _owner(user, None, alias)


_CHAT_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl-tool-policy",
        "object": "chat.completion",
        "created": 1,
        "model": "integration-tool-policy",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
).encode()


def _chat_upstream(request: Request) -> Reply:
    if (request.method, request.target) == ("GET", "/v1/models"):
        return Reply(body=b'{"object": "list", "data": []}')
    assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request
    return Reply(body=_CHAT_REPLY)


def _tool_policy_config(directory: Path, model: str, upstream_url: str) -> Path:
    path: Final = _proxy_config(directory, model, upstream_url, general_settings={})
    config: Final = JSON_OBJECT.validate_json(path.read_text())
    config["guardrails"] = [
        {
            "guardrail_name": "integration-tool-policy",
            "litellm_params": {"guardrail": "tool_policy", "mode": "pre_call", "default_on": True},
        }
    ]
    path.write_text(json.dumps(config))
    return path


def _blocked_detail(tool_name: str) -> dict[str, JsonValue]:
    return {
        "error": {
            "message": "Violated tool policy",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
            "provider_specific_fields": {
                "error": "Violated tool policy",
                "blocked_tools": [tool_name],
                "message": f"Tool(s) {[tool_name]} are blocked by policy.",
                "guardrail_name": "integration-tool-policy",
                "guardrail_mode": "pre_call",
            },
        }
    }


class UpstreamFunction(BaseModel):
    name: str


class UpstreamTool(BaseModel):
    function: UpstreamFunction


class UpstreamChat(BaseModel):
    tools: list[UpstreamTool]


def _upstream_tool_names(upstream: Wire) -> list[list[str]]:
    return [
        [tool.function.name for tool in UpstreamChat.model_validate_json(request.body).tools]
        for request in upstream.drain()
        if request.method == "POST"
    ]


def _assert_chat_refused(gateway: Gateway, upstream: Wire, model: str, key: str, tool_name: str) -> None:
    upstream.drain()
    refused: Final = gateway.request("POST", "/v1/chat/completions", _tool_call_request(model, tool_name), key=key)
    assert refused.status_code == 400, refused.text
    assert refused.json() == _blocked_detail(tool_name), refused.text
    assert _upstream_tool_names(upstream) == [], "a blocked tool reached the model upstream"


def _assert_chat_served(gateway: Gateway, upstream: Wire, model: str, key: str, tool_name: str) -> None:
    upstream.drain()
    served: Final = gateway.request("POST", "/v1/chat/completions", _tool_call_request(model, tool_name), key=key)
    assert served.status_code == 200, served.text
    assert served.json()["choices"][0]["message"]["content"] == "ok", served.text
    assert _upstream_tool_names(upstream) == [[tool_name]]


def test_global_tool_policy_update_reads_back_persists_and_blocks_the_next_request(
    gateway: Gateway, tmp_path: Path
) -> None:
    model: Final = "integration-tool-policy-" + uuid.uuid4().hex
    suffix: Final = uuid.uuid4().hex[:12]
    blocked: Final = f"integration/policy/{suffix}/blocked"
    untouched: Final = f"integration_policy_{suffix}_free"
    with (
        wire_server(_chat_upstream) as upstream,
        owned_proxy(gateway, tmp_path, {}, config=_tool_policy_config(tmp_path, model, upstream.url)) as candidate,
        candidate.scenario() as scenario,
    ):
        scenario.cleanups.callback(_forget_tool, blocked)
        scenario.cleanups.callback(_forget_tool, untouched)
        key: Final = scenario.key(models=[model])
        _assert_chat_served(candidate, upstream, model, key, blocked)
        _assert_chat_served(candidate, upstream, model, key, untouched)
        _discovered_tool(candidate, blocked)
        _discovered_tool(candidate, untouched)

        set_input: Final = candidate.request(
            "POST", "/v1/tool/policy", {"tool_name": blocked, "input_policy": "blocked"}
        )
        assert set_input.status_code == 200, set_input.text
        assert set_input.json() == {
            "tool_name": blocked,
            "input_policy": "blocked",
            "output_policy": "untrusted",
            "updated": True,
            "team_id": None,
            "key_hash": None,
        }, set_input.text
        set_output: Final = candidate.request(
            "POST", "/v1/tool/policy", {"tool_name": blocked, "output_policy": "trusted"}
        )
        assert set_output.status_code == 200, set_output.text
        assert (set_output.json()["input_policy"], set_output.json()["output_policy"]) == ("blocked", "trusted")

        single: Final = _single(candidate, blocked)
        assert (single["tool_name"], single["input_policy"], single["output_policy"]) == (blocked, "blocked", "trusted")
        detail: Final = candidate.get(f"/v1/tool/{blocked}/detail")
        assert detail["overrides"] == [], detail
        detail_tool: Final = object_value(detail["tool"])
        assert (detail_tool["input_policy"], detail_tool["output_policy"]) == ("blocked", "trusted"), detail
        assert read_rows(
            'SELECT input_policy, output_policy FROM "LiteLLM_ToolTable" WHERE tool_name = %s', (blocked,)
        ) == [{"input_policy": "blocked", "output_policy": "trusted"}]
        assert (_single(candidate, untouched)["input_policy"], _single(candidate, untouched)["output_policy"]) == (
            "untrusted",
            "untrusted",
        )

        _assert_chat_refused(candidate, upstream, model, key, blocked)
        _assert_chat_served(candidate, upstream, model, key, untouched)


def test_global_blocked_tool_policy_refuses_the_mcp_rest_call_naming_it(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip(
        "BUG: /mcp-rest/tools/call runs a tool whose global input_policy is blocked, the tool_policy guardrail sees no"
        " user_api_key_request_route in the synthetic MCP payload so it extracts no tool name"
    )
    model: Final = "integration-tool-policy-" + uuid.uuid4().hex
    suffix: Final = uuid.uuid4().hex[:12]
    with (
        wire_server(_chat_upstream) as upstream,
        owned_proxy(gateway, tmp_path, {}, config=_tool_policy_config(tmp_path, model, upstream.url)) as candidate,
        candidate.scenario() as scenario,
        scripted_peer(echo_tool(f"lookup_{suffix}"), echo_tool(f"free_{suffix}")) as peer,
    ):
        identity: Final = register_mcp(scenario, peer, "policy" + suffix)
        mcp_key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        scenario.cleanups.callback(_forget_tool, f"lookup_{suffix}")
        mcp_blocked: Final = candidate.request(
            "POST", "/v1/tool/policy", {"tool_name": f"lookup_{suffix}", "input_policy": "blocked"}
        )
        assert mcp_blocked.status_code == 200, mcp_blocked.text
        peer.drain()
        refused: Final = _mcp_call(candidate, mcp_key, identity, f"lookup_{suffix}")
        assert refused.status_code == 400, refused.text
        assert refused.json()["detail"]["blocked_tools"] == [f"lookup_{suffix}"], refused.text
        assert tool_calls(peer.drain()) == (), "a blocked MCP tool reached the MCP server"
        served: Final = _mcp_call(candidate, mcp_key, identity, f"free_{suffix}")
        assert served.status_code == 200, served.text
        assert served.json()["content"][0]["text"] == '{"probe": "policy"}', served.text
        assert [call["body"]["params"]["name"] for call in tool_calls(peer.drain())] == [f"free_{suffix}"]


def _mcp_call(gateway: Gateway, key: str, identity: str, name: str) -> httpx.Response:
    return gateway.client.post(
        "/mcp-rest/tools/call",
        headers={"x-litellm-api-key": key},
        json={"server_id": identity, "name": name, "arguments": {"probe": "policy"}},
    )


_TEAM_BLOCKED_TOOLS: Final = (
    'SELECT p.blocked_tools FROM "LiteLLM_TeamTable" t JOIN "LiteLLM_ObjectPermissionTable" p'
    " ON p.object_permission_id = t.object_permission_id WHERE t.team_id = %s"
)
_KEY_BLOCKED_TOOLS: Final = (
    'SELECT p.blocked_tools FROM "LiteLLM_VerificationToken" k JOIN "LiteLLM_ObjectPermissionTable" p'
    " ON p.object_permission_id = k.object_permission_id WHERE k.token = %s"
)


def test_scoped_tool_overrides_block_only_their_team_or_key_survive_restart_and_delete_restores(
    gateway: Gateway, tmp_path: Path
) -> None:
    model: Final = "integration-tool-override-" + uuid.uuid4().hex
    tool: Final = f"integration/override/{uuid.uuid4().hex[:12]}/tool"
    config: Final = tmp_path / "config"
    config.mkdir()
    with wire_server(_chat_upstream) as upstream, gateway.scenario() as scenario:
        scenario.cleanups.callback(_forget_tool, tool)
        proxy_config: Final = _tool_policy_config(config, model, upstream.url)
        unscoped: Final[dict[str, JsonValue]] = {"object_permission": {"mcp_servers": []}}
        blocked_team: Final = scenario.team(**unscoped)
        sibling_team: Final = scenario.team(**unscoped)
        team_key: Final = scenario.key(team_id=blocked_team, models=[model])
        sibling_team_key: Final = scenario.key(team_id=sibling_team, models=[model])
        blocked_key: Final = scenario.key(models=[model], **unscoped)
        sibling_key: Final = scenario.key(models=[model], **unscoped)
        blocked_hash: Final = sha256(blocked_key.encode()).hexdigest()

        def assert_scopes(candidate: Gateway, refused: tuple[str, ...], served: tuple[str, ...]) -> None:
            for key in refused:
                _assert_chat_refused(candidate, upstream, model, key, tool)
            for key in served:
                _assert_chat_served(candidate, upstream, model, key, tool)

        first_dir: Final = tmp_path / "first"
        first_dir.mkdir()
        with owned_proxy(gateway, first_dir, {}, config=proxy_config) as first:
            assert_scopes(first, (), (team_key, sibling_team_key, blocked_key, sibling_key))
            for scope in ({"team_id": blocked_team}, {"key_hash": blocked_hash}):
                response = first.request(
                    "POST", "/v1/tool/policy", {"tool_name": tool, "input_policy": "blocked", **scope}
                )
                assert response.status_code == 200, response.text
                assert response.json() == {
                    "tool_name": tool,
                    "input_policy": "blocked",
                    "output_policy": None,
                    "updated": True,
                    "team_id": None,
                    "key_hash": None,
                    **scope,
                }, response.text
            assert_scopes(first, (team_key, blocked_key), (sibling_team_key, sibling_key))
            _discovered_tool(first, tool)
            overrides: Final = first.get(f"/v1/tool/{tool}/detail")["overrides"]
            assert isinstance(overrides, list), overrides
            assert sorted(
                (
                    str(object_value(row)["team_id"]),
                    str(object_value(row)["key_hash"]),
                    object_value(row)["input_policy"],
                )
                for row in overrides
            ) == sorted([(blocked_team, "None", "blocked"), ("None", blocked_hash, "blocked")]), overrides
            assert read_rows(_TEAM_BLOCKED_TOOLS, (blocked_team,)) == [{"blocked_tools": [tool]}]
            assert read_rows(_KEY_BLOCKED_TOOLS, (blocked_hash,)) == [{"blocked_tools": [tool]}]
            assert read_rows(_TEAM_BLOCKED_TOOLS, (sibling_team,)) == [{"blocked_tools": []}]

        second_dir: Final = tmp_path / "second"
        second_dir.mkdir()
        with owned_proxy(gateway, second_dir, {}, config=proxy_config) as second:
            assert_scopes(second, (team_key, blocked_key), (sibling_team_key, sibling_key))
            removed_team: Final = second.request(
                "DELETE", f"/v1/tool/{tool}/overrides", params={"team_id": blocked_team}
            )
            assert removed_team.status_code == 200, removed_team.text
            assert removed_team.json() == {"deleted": True, "tool_name": tool}, removed_team.text
            assert_scopes(second, (blocked_key,), (team_key, sibling_team_key, sibling_key))
            removed_key: Final = second.request(
                "DELETE", f"/v1/tool/{tool}/overrides", params={"key_hash": blocked_hash}
            )
            assert removed_key.status_code == 200, removed_key.text
            assert removed_key.json() == {"deleted": True, "tool_name": tool}, removed_key.text
            assert_scopes(second, (), (team_key, sibling_team_key, blocked_key, sibling_key))
            assert read_rows(_TEAM_BLOCKED_TOOLS, (blocked_team,)) == [{"blocked_tools": []}]
            assert read_rows(_KEY_BLOCKED_TOOLS, (blocked_hash,)) == [{"blocked_tools": []}]
