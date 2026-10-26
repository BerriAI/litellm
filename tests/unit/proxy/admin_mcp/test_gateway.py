import copy
import json

import httpx2
from mcp.shared.exceptions import MCPError
import pytest

from tests.unit.proxy.admin_mcp.conftest import SPEC, data, session_for
from litellm.proxy.admin_mcp.catalog import OPERATIONS
from litellm.proxy.admin_mcp.config import Config
from litellm.proxy.admin_mcp.gateway import AdminError, Gateway, HELPERS, MAX_REDACTION_DEPTH, safe_result
from litellm.proxy.admin_mcp.schema import tool_schema
from litellm.proxy.admin_mcp.server import create_http_app


async def test_real_mcp_create_key_and_read_back(stub):
    async with session_for(stub) as session:
        tools = (await session.list_tools()).tools
        assert {t.name for t in tools} == {
            "create_key",
            "list_keys",
            "add_model",
            "get_model",
            "reset_key_spend",
        } | HELPERS.keys()
        created = data(await session.call_tool("create_key", {"body": {"key_alias": "engineering", "max_budget": 100}}))
        assert created["key"].startswith("sk-created-for-")
        assert created["echo"] == "[credential redacted]"
        found = data(await session.call_tool("list_keys", {"query": {"page": 1, "size": 10}}))
        assert found["keys"][0]["key_alias"] == "engineering"
    assert len(stub.writes) == 1
    assert all(r.headers["Authorization"] == "Bearer alice-admin" for r in stub.requests)


async def test_real_mcp_add_model_verify_and_strip_provider_secrets(stub):
    body = {
        "model_name": "support",
        "litellm_params": {"model": "openai/gpt-6.1-sol", "litellm_credential_name": "production"},
        "model_info": {},
    }
    async with session_for(stub) as session:
        created = data(await session.call_tool("add_model", {"body": body}))
        assert created["model_id"] == "model-1"
        read = data(await session.call_tool("get_model", {"query": {"litellm_model_id": "model-1"}}))
        assert read["data"][0]["model_info"]["id"] == "model-1"
        assert read["data"][0]["litellm_params"] == body["litellm_params"]
        assert "secret" not in json.dumps([created, read])
    assert json.loads(stub.writes[0].content) == body


@pytest.mark.parametrize("options", [{"read_only": True}, {"allowed_tools": frozenset({"list_keys"})}])
async def test_policy_blocks_hidden_write_even_when_called_directly(stub, options):
    async with session_for(stub, **options) as session:
        assert "create_key" not in {t.name for t in (await session.list_tools()).tools}
        assert (await session.call_tool("create_key", {"body": {"key_alias": "forbidden"}})).is_error
    assert not stub.writes


async def test_unverified_user_cannot_discover_or_call(stub):
    async with session_for(stub, api_key="viewer") as session:
        with pytest.raises(MCPError):
            await session.list_tools()
        assert (await session.call_tool("create_key", {"body": {"key_alias": "forbidden"}})).is_error
    assert all(r.url.path == "/v2/user/info" for r in stub.requests)


@pytest.mark.parametrize(
    "arguments",
    [{}, {"body": {}}, {"body": {"key_alias": 7}}, {"body": {"key_alias": "a"}, "headers": {"Authorization": "spoof"}}],
)
async def test_invalid_arguments_do_not_reach_management_api(stub, arguments):
    async with session_for(stub) as session:
        assert (await session.call_tool("create_key", arguments)).is_error
    assert not stub.writes


@pytest.mark.parametrize("identifier", ["..", "a/b", "a\\b", "a\n", "a%2Fb", ""])
async def test_path_parameters_cannot_escape_route(stub, identifier):
    async with session_for(stub) as session:
        assert (await session.call_tool("reset_key_spend", {"path": {"key": identifier}})).is_error
    assert not stub.writes


@pytest.mark.parametrize("status,timeout", [(307, False), (403, False), (500, False), (200, True)])
async def test_failed_write_is_not_retried_or_redirected_and_does_not_echo_body(stub, status, timeout):
    stub.status, stub.timeout = status, timeout
    async with session_for(stub) as session:
        result = await session.call_tool("create_key", {"body": {"key_alias": "one"}})
        assert result.is_error and "before retrying" in result.content[0].text
        assert "secret" not in result.content[0].text
    assert len(stub.writes) == 1


async def test_revocation_during_write_suppresses_returned_key(stub):
    stub.revoke = True
    async with session_for(stub) as session:
        result = await session.call_tool("create_key", {"body": {"key_alias": "one"}})
        assert result.is_error and "sk-created" not in str(result)
    assert len(stub.writes) == 1


async def test_operation_id_drift_fails_closed(stub):
    stub.spec = copy.deepcopy(SPEC)
    stub.spec["paths"]["/model/new"]["post"]["operationId"] = "changed"
    async with session_for(stub) as session:
        with pytest.raises(MCPError, match="changed"):
            await session.list_tools()
    assert not stub.writes


def test_catalog_has_unique_stable_tools_and_preserves_existing_surface():
    assert len(OPERATIONS) == 65
    assert len({o.name for o in OPERATIONS}) == len(OPERATIONS)
    assert len({(o.method, o.path) for o in OPERATIONS}) == len(OPERATIONS)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@example.com",
        "https://example.com/path",
        "https://example.com?key=secret",
        "http://public.example.com",
        "https://example.com:bad",
    ],
)
def test_gateway_must_be_a_fixed_trusted_origin(url):
    with pytest.raises(ValueError, match=r"gateway|HTTPS"):
        Config(url)


async def test_http_server_requires_each_request_credential(stub, monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "alice-admin")
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(stub.handle)) as client:
        app = create_http_app(Gateway(Config("https://gateway.example.com"), client))
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="http://localhost:4000"
            ) as caller:
                payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
                headers = {"Accept": "application/json, text/event-stream"}
                assert (await caller.post("/mcp", json=payload, headers=headers)).status_code == 401
                for credential in ("alice-admin", "bob-admin"):
                    response = await caller.post(
                        "/mcp", json=payload, headers={**headers, "Authorization": "Bearer " + credential}
                    )
                    assert response.status_code == 200
                    assert not response.json().get("error")
                assert (await caller.post("/mcp", json=payload, headers=headers)).status_code == 401
    assert {request.headers["authorization"] for request in stub.requests} == {"Bearer alice-admin", "Bearer bob-admin"}


def test_recursive_refs_and_overridden_parameters_have_valid_schema():
    import jsonschema

    spec = {
        "components": {
            "schemas": {
                "Node": {
                    "type": "object",
                    "properties": {"next": {"$ref": "#/components/schemas/Node", "nullable": True}},
                }
            }
        }
    }
    path = {"parameters": [{"in": "query", "name": "limit", "schema": {"type": "string"}}]}
    op = {
        "parameters": [{"in": "query", "name": "limit", "schema": {"type": "integer"}}],
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Node"}}},
        },
    }
    schema = tool_schema(spec, path, op)
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate({"query": {"limit": 3}, "body": {"next": {"next": None}}})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"query": {"limit": "wrong"}, "body": {}})


def test_schema_names_with_json_pointer_escapes():
    import jsonschema

    spec = {"components": {"schemas": {"A/B~C": {"type": "integer"}}}}
    op = {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/A~1B~0C"}}},
        }
    }
    schema = tool_schema(spec, {}, op)
    jsonschema.Draft202012Validator(schema).validate({"body": 3})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate({"body": "wrong"})


async def test_shared_http_client_never_replays_another_callers_gateway_cookie(stub):
    from litellm.proxy.admin_mcp.gateway import Gateway

    original = stub.handle

    def handle(request):
        assert not request.headers.get("cookie")
        result = original(request)
        result.headers["set-cookie"] = "gateway-session=first-caller; Path=/"
        return result

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        gateway = Gateway(Config("https://gateway.example.com"), client)
        for key in ("alice-admin", "bob-admin"):
            await gateway.call("create_key", {"body": {"key_alias": key}}, key)
    assert len(stub.writes) == 2


@pytest.mark.parametrize("container", ["object", "array", "encoded"])
def test_response_redaction_fails_closed_for_excessive_nesting(container):
    value = {"echo": "personal-admin-credential", "password": "provider-secret"}
    for _ in range(MAX_REDACTION_DEPTH):
        value = [value] if container == "array" else {"metadata": value}
    payload = json.dumps(value) if container == "encoded" else value
    with pytest.raises(AdminError, match="Inspect gateway state before retrying"):
        safe_result(payload, "personal-admin-credential")


def test_response_redaction_handles_encoded_secrets_and_preserves_other_values():
    credential = "personal-admin-credential"
    value = {"metadata": json.dumps({"nested": [{"echo": credential, "password": "provider-secret", "count": 7}]})}
    assert safe_result(value, credential) == {
        "metadata": json.dumps({"nested": [{"echo": "[credential redacted]", "password": "[redacted]", "count": 7}]})
    }


@pytest.mark.parametrize(
    "payload",
    [
        "[" * 250 + '{"password":"provider-secret"}' + "]" * 250,
        '  {"password":"provider-secret"',
        '[{"password":"provider-secret"}',
    ],
)
def test_response_redaction_rejects_unparseable_json_containers(payload):
    with pytest.raises(AdminError, match="Inspect gateway state before retrying"):
        safe_result({"metadata": payload}, "personal-admin-credential")
