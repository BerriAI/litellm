from collections.abc import AsyncIterator
import copy
import json
import re

import httpx2
import jsonschema
from pydantic import JsonValue
import pytest

from tests.unit.proxy.admin_mcp.conftest import SPEC, SessionOptions, StubGateway, connected, data, session_for
from litellm.proxy.admin_mcp.catalog import OPERATIONS
from litellm.proxy.admin_mcp.config import Config
from litellm.proxy.admin_mcp.gateway import AdminError, Gateway, HELPERS
from litellm.proxy.admin_mcp.results import ResultStore


async def test_discovery_keeps_named_tools_annotations_and_complete_validation(stub: StubGateway) -> None:
    async with session_for(stub, schema_mode="discovery") as session:
        listed = {t.name: t for t in (await session.list_tools()).tools}
        assert not listed["create_key"].annotations.read_only_hint
        assert listed["create_key"].annotations.destructive_hint
        assert listed["list_keys"].annotations.read_only_hint
        detail = data(await session.call_tool("describe_admin_tool", {"name": "create_key"}))
        jsonschema.Draft202012Validator(detail["inputSchema"]).validate({"body": {"key_alias": "a", "max_budget": 0}})
        rejected = await session.call_tool("create_key", {"body": {"key_alias": "a", "unsupported": True}})
        assert rejected.is_error and "complete tool schema" in rejected.content[0].text
        assert not stub.writes
        result = data(await session.call_tool("create_key", {"body": {"key_alias": "a", "max_budget": 0}}))
        assert result["max_budget"] == 0


async def test_full_schema_exposes_body_fields_without_losing_nested_definitions(stub: StubGateway) -> None:
    async with session_for(stub) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        key_schema = tools["create_key"].input_schema
        assert key_schema["properties"]["body"]["type"] == "object"
        assert "key_alias" in key_schema["properties"]["body"]["properties"]
        assert "$defs" not in key_schema
        model_schema = tools["add_model"].input_schema
        assert "Params" in model_schema["$defs"]
        assert "Deployment" not in model_schema["$defs"]


async def test_schema_itself_can_be_paged_and_retrieved_in_full(stub: StubGateway) -> None:
    stub.spec = copy.deepcopy(SPEC)
    stub.spec["components"]["schemas"]["Key"]["properties"]["key_alias"]["description"] = (
        "Long but useful documentation. " * 2000
    )
    async with session_for(stub, schema_mode="discovery") as session:
        full = data(await session.call_tool("describe_admin_tool", {"name": "create_key"}))
        indexed = data(
            await session.call_tool("describe_admin_tool", {"name": "create_key", "response": {"view": "compact"}})
        )
        assert data(await session.call_tool("read_admin_result", indexed["full_read"])) == full


@pytest.mark.parametrize(
    "schema,value",
    [
        ({"type": "integer"}, 0),
        ({"type": "boolean"}, False),
        ({"type": "null"}, None),
        ({"type": "array", "items": {"type": "string"}}, ["uncommon", "argument"]),
    ],
)
async def test_discovery_preserves_non_object_body_shapes(
    stub: StubGateway, schema: dict[str, JsonValue], value: JsonValue
) -> None:
    stub.spec = copy.deepcopy(SPEC)
    stub.spec["components"]["schemas"]["Key"] = schema
    original = stub.handle
    seen = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/key/generate":
            seen.append(json.loads(request.content))
            return httpx2.Response(200, json={"accepted": seen[-1]})
        return original(request)

    stub.handle = handle
    async with session_for(stub, schema_mode="discovery") as session:
        detail = data(await session.call_tool("describe_admin_tool", {"name": "create_key"}))
        jsonschema.Draft202012Validator(detail["inputSchema"]).validate({"body": value})
        assert data(await session.call_tool("create_key", {"body": value})) == {"accepted": value}
    assert seen == [value]


async def test_post_reads_are_available_in_read_only_mode_but_health_is_not(stub: StubGateway) -> None:
    stub.spec = copy.deepcopy(SPEC)
    for op in OPERATIONS:
        if op.name in {"get_budgets", "get_organization_legacy", "check_key_health"}:
            stub.spec["paths"][op.path] = {op.method.lower(): {"operationId": op.operation_id}}
    original = stub.handle

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path in {"/budget/info", "/organization/info"}:
            return httpx2.Response(200, json=[])
        return original(request)

    stub.handle = handle
    async with session_for(stub, read_only=True, schema_mode="discovery") as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        assert "check_key_health" not in tools
        for name in ("get_budgets", "get_organization_legacy"):
            assert tools[name].annotations.read_only_hint
            assert not tools[name].annotations.destructive_hint
            assert data(await session.call_tool(name, {})) == []


async def test_full_mode_is_unchanged_and_compact_pages_do_not_refetch(stub: StubGateway) -> None:
    stub.keys = [
        {"id": i, "max_budget": None, "spend": 0, "blocked": False, "extra": {"rare_setting": "z" * 8000}}
        for i in range(15)
    ]
    async with session_for(stub, response_view="full") as session:
        original = data(await session.call_tool("list_keys", {}))
        assert original == {"keys": stub.keys}
        paged = data(await session.call_tool("list_keys", {"response": {"view": "compact"}}))
        assert paged["format"] == "litellm-admin-page-v1" and not paged["complete"]
        count = sum(r.url.path == "/key/list" for r in stub.requests)
        stub.keys.clear()
        restored = data(await session.call_tool("read_admin_result", paged["full_read"]))
        assert restored == original
        assert sum(r.url.path == "/key/list" for r in stub.requests) == count


async def test_compact_default_keeps_small_results_and_full_override(stub: StubGateway) -> None:
    async with session_for(stub, response_view="compact") as session:
        assert data(await session.call_tool("list_keys", {})) == {"keys": []}
        stub.keys = [{"id": i, "metadata": "a" * 1000} for i in range(40)]
        assert "result_id" in data(await session.call_tool("list_keys", {}))
        assert data(await session.call_tool("list_keys", {"response": {"view": "full"}})) == {"keys": stub.keys}


@pytest.mark.parametrize("name", ["list_models", "get_model"])
async def test_model_catalog_identifiers_allow_lossless_paging(stub: StubGateway, name: str) -> None:
    operation = next(op for op in OPERATIONS if op.name == name)
    stub.spec = copy.deepcopy(SPEC)
    stub.spec["paths"][operation.path] = {"get": {"operationId": operation.operation_id}}
    models = [
        {
            "model_name": f"deployment-{i}",
            "model_info": {
                "id": f"model-{i}",
                "key": f"openai/example-model-{i}",
                "input_cost_per_token": 0,
                "supports_vision": False,
                "metadata": {"uncommon": None, "details": "x" * 2000},
            },
        }
        for i in range(40)
    ]
    expected = {"data": models, "total_count": len(models), "current_page": 1, "total_pages": 1}
    original = stub.handle

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == operation.path:
            stub.requests.append(request)
            return httpx2.Response(200, json=expected)
        return original(request)

    stub.handle = handle
    async with session_for(stub) as session:
        first = data(await session.call_tool(name, {}))
        assert first["format"] == "litellm-admin-page-v1"
        assert len(json.dumps(first)) < 20000
        assert data(await session.call_tool("read_admin_result", first["full_read"])) == expected
        records = next(e for e in first["entries"] if e["key"] == "data")
        current = data(await session.call_tool("read_admin_result", records["read"]))
        restored = []
        while True:
            for entry in current["entries"]:
                assert entry["complete"]
                restored.append(entry["value"])
            if not current["next"]:
                break
            current = data(await session.call_tool("read_admin_result", current["next"]))
        assert restored == models
        assert sum(r.url.path == operation.path for r in stub.requests) == 1
        assert data(await session.call_tool(name, {"response": {"view": "full"}})) == expected


async def test_full_write_result_can_be_read_without_repeating_write(stub: StubGateway) -> None:
    original_handler = stub.handle
    payload = {"status": "ok", "previous": {"spend": 20, "custom": "x" * 22000}}

    def handle(request: httpx2.Request) -> httpx2.Response:
        response = original_handler(request)
        return httpx2.Response(200, json=payload) if request.url.path.endswith("/reset_spend") else response

    stub.handle = handle
    async with session_for(stub, response_view="compact") as session:
        result = data(await session.call_tool("reset_key_spend", {"path": {"key": "a" * 64}}))
        full = data(await session.call_tool("read_admin_result", result["full_read"]))
        assert full == payload
    assert len(stub.writes) == 1


async def test_large_new_key_response_is_delivered_inline_and_never_cached(stub: StubGateway) -> None:
    original = stub.handle

    def handle(request: httpx2.Request) -> httpx2.Response:
        response = original(request)
        if request.url.path == "/key/generate":
            return httpx2.Response(200, json={**response.json(), "metadata": "x" * 20000})
        return response

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        gateway = Gateway(Config("https://gateway.example.com", response_view="compact"), client)
        async with connected(gateway) as session:
            result = data(await session.call_tool("create_key", {"body": {"key_alias": "new"}}))
            assert result["key"].startswith("sk-created")
            assert len(result["metadata"]) == 20000
            assert not gateway.results._items
    assert len(stub.writes) == 1


async def test_result_reads_require_same_credential_principal_and_current_role(stub: StubGateway) -> None:
    stub.keys = [{"data": "x" * 20000}]
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(stub.handle)) as client:
        gateway = Gateway(Config("https://gateway.example.com", response_view="compact"), client)
        result = await gateway.call("list_keys", {}, "alice-admin")
        for credential in ("bob-admin", "viewer"):
            with pytest.raises(AdminError):
                await gateway.call("read_admin_result", result["full_read"], credential)
        stub.revoked.add("alice-admin")
        with pytest.raises(AdminError):
            await gateway.call("read_admin_result", result["full_read"], "alice-admin")


async def test_revocation_during_result_read_suppresses_snapshot(stub: StubGateway) -> None:
    stub.keys = [{"data": "private" * 4000}]
    revoke = False

    def clock() -> float:
        if revoke:
            stub.revoked.add("alice-admin")
        return 0.0

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(stub.handle)) as client:
        gateway = Gateway(
            Config("https://gateway.example.com", response_view="compact"), client, results=ResultStore(clock=clock)
        )
        result = await gateway.call("list_keys", {}, "alice-admin")
        revoke = True
        with pytest.raises(AdminError, match="proxy admins"):
            await gateway.call("read_admin_result", result["full_read"], "alice-admin")


async def test_schema_helper_respects_allowlist_and_cannot_execute_hidden_operations(stub: StubGateway) -> None:
    async with session_for(stub, allowed_tools=frozenset({"list_keys"}), schema_mode="discovery") as session:
        for name in ("create_key", "flush_everything", "read_admin_result"):
            assert (await session.call_tool("describe_admin_tool", {"name": name})).is_error
        assert (await session.call_tool("read_admin_result", {"result_id": "guess", "method": "POST"})).is_error
        assert (await session.call_tool("create_key", {"body": {"key_alias": "a"}})).is_error
    assert not stub.writes


async def test_warm_call_retains_pre_post_checks_without_discovery_auth(stub: StubGateway) -> None:
    async with session_for(stub) as session:
        await session.list_tools()
        stub.requests.clear()
        await session.call_tool("list_keys", {})
        assert [r.url.path for r in stub.requests] == ["/v2/user/info", "/key/list", "/v2/user/info"]


async def test_identity_change_during_cold_discovery_prevents_write(stub: StubGateway) -> None:
    original = stub.handle
    changed = False

    def handle(request: httpx2.Request) -> httpx2.Response:
        nonlocal changed
        result = original(request)
        if request.url.path == "/openapi.json":
            changed = True
        elif request.url.path == "/v2/user/info" and changed:
            return httpx2.Response(
                200,
                json={"user_id": "new-identity", "user_role": "proxy_admin"},
            )
        return result

    stub.handle = handle
    async with session_for(stub) as session:
        result = await session.call_tool("create_key", {"body": {"key_alias": "a"}})
        assert result.is_error and "identity changed" in result.content[0].text
    assert not stub.writes


async def test_capacity_falls_back_to_full_write_response_without_retry(stub: StubGateway) -> None:
    original = stub.handle
    payload = {"status": "ok", "previous": "x" * 20000}

    def handle(request: httpx2.Request) -> httpx2.Response:
        result = original(request)
        return httpx2.Response(200, json=payload) if request.url.path.endswith("/reset_spend") else result

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        gateway = Gateway(
            Config("https://gateway.example.com", response_view="compact"), client, results=ResultStore(max_bytes=1)
        )
        assert await gateway.call("reset_key_spend", {"path": {"key": "fixture"}}, "alice-admin") == payload
    assert len(stub.writes) == 1


async def test_views_support_large_results_and_stop_at_stream_limit(stub: StubGateway) -> None:
    original = stub.handle

    class Chunks(httpx2.AsyncByteStream):
        def __init__(self) -> None:
            self.count = 0

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for _ in range(30):
                self.count += 1
                yield b"x" * 1_000_000

    stream = Chunks()

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/key/list":
            if request.url.params.get("page") == "2":
                return httpx2.Response(200, stream=stream)
            return httpx2.Response(200, json={"data": "x" * 2_100_000})
        return original(request)

    stub.handle = handle
    async with session_for(stub) as session:
        assert len(data(await session.call_tool("list_keys", {"response": {"view": "full"}}))["data"]) == 2_100_000
        result = data(await session.call_tool("list_keys", {"response": {"view": "compact"}}))
        assert result["entries"][0]["total"] == 2_100_000
        error = await session.call_tool("list_keys", {"query": {"page": 2}, "response": {"view": "compact"}})
        assert error.is_error and "byte limit" in error.content[0].text
    assert stream.count == 17


async def test_full_schemas_support_compact_large_results(stub: StubGateway) -> None:
    config = Config("https://gateway.example.com")
    assert config.response_view == "compact" and config.schema_mode == "full"
    stub.keys = [{"data": "x" * 20000}]
    async with session_for(stub) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        assert "key_alias" in tools["create_key"].input_schema["properties"]["body"]["properties"]
        assert "result_id" in data(await session.call_tool("list_keys", {}))


@pytest.mark.parametrize("mode", ["full", "discovery"])
async def test_every_reviewed_operation_retains_mapping_advanced_inputs_and_result(mode: str) -> None:
    spec = {
        "paths": {},
        "components": {
            "schemas": {
                "Advanced": {
                    "type": "object",
                    "properties": {
                        "model_limits": {"type": "object", "additionalProperties": {"type": ["number", "null"]}},
                        "enabled": {"type": "boolean"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["enabled"],
                    "additionalProperties": False,
                },
                "Body": {
                    "type": "object",
                    "properties": {"advanced": {"$ref": "#/components/schemas/Advanced"}},
                    "required": ["advanced"],
                    "additionalProperties": False,
                },
            }
        },
    }
    for op in OPERATIONS:
        parameters = [
            {"in": "path", "name": name, "required": True, "schema": {"type": "string"}}
            for name in re.findall(r"\{([^}]+)\}", op.path)
        ]
        parameters += [{"in": "query", "name": "filter", "schema": {"type": "array", "items": {"type": "string"}}}]
        definition = {"operationId": op.operation_id, "parameters": parameters}
        if op.method != "GET":
            definition["requestBody"] = {
                "required": True,
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Body"}}},
            }
        spec["paths"].setdefault(op.path, {})[op.method.lower()] = definition
    calls = []
    result = {"budget": None, "spend": 0, "blocked": False, "metadata": {"unexpected_field": [1, "a", None]}}

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/openapi.json":
            return httpx2.Response(200, json=spec)
        if request.url.path == "/v2/user/info" and "litellm-changed-by" not in request.headers:
            return httpx2.Response(200, json={"user_id": "admin", "user_role": "proxy_admin"})
        calls.append(request)
        return httpx2.Response(200, json=result)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        gateway = Gateway(Config("https://gateway.example.com", schema_mode=mode), client)
        async with connected(gateway, "admin") as session:
            listed = {t.name: t for t in (await session.list_tools()).tools}
            assert set(listed) - HELPERS.keys() == {op.name for op in OPERATIONS}
            for op in OPERATIONS:
                detail = data(await session.call_tool("describe_admin_tool", {"name": op.name}))
                assert detail["annotations"]["readOnlyHint"] == op.read_only
                arguments = {"query": {"filter": ["first", "second"]}}
                path_args = {name: "fixture" for name in re.findall(r"\{([^}]+)\}", op.path)}
                if path_args:
                    arguments["path"] = path_args
                if op.method != "GET":
                    arguments["body"] = {
                        "advanced": {
                            "enabled": False,
                            "model_limits": {"rare/model": None, "free/model": 0},
                            "tags": [],
                        }
                    }
                jsonschema.Draft202012Validator(detail["inputSchema"]).validate(arguments)
                assert data(await session.call_tool(op.name, arguments)) == result
                request = calls[-1]
                assert request.method == op.method
                assert request.url.path == re.sub(r"\{[^}]+\}", "fixture", op.path)
                assert request.url.params.get_list("filter") == ["first", "second"]
                assert request.headers["litellm-changed-by"] == "admin"
                if "body" in arguments:
                    assert json.loads(request.content) == arguments["body"]
    assert len(calls) == 65


@pytest.mark.parametrize("config", [{"schema_mode": "unknown"}, {"response_view": "unknown"}])
def test_unknown_modes_fail_instead_of_silently_changing_behavior(config: SessionOptions) -> None:
    with pytest.raises(ValueError, match="LITELLM_ADMIN_"):
        Config("https://gateway.example.com", **config)
