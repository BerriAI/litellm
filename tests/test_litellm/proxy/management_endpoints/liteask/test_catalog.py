import copy
import json
from typing import Final

import pytest
from fastapi import FastAPI
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.management_endpoints.liteask.catalog import (
    ArgumentsError,
    CatalogError,
    Tool,
    ToolRequest,
    build_catalog,
    tool_request,
    validate_arguments,
)


def _spec(body: JsonValue = None) -> dict[str, JsonValue]:
    return {
        "paths": {
            "/team/member_add": {
                "post": {
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": body or {"$ref": "#/components/schemas/Member"}}},
                    }
                }
            },
            "/key/list": {
                "parameters": [{"$ref": "#/components/parameters/Page"}],
                "get": {
                    "parameters": [
                        {"name": "page", "in": "query", "schema": {"type": "integer", "minimum": 1, "maximum": 100}},
                        {"name": "team_ids", "in": "query", "schema": {"type": "array", "items": {"type": "string"}}},
                        {"name": "return_full_object", "in": "query", "schema": {"type": "boolean"}},
                        {"name": "Authorization", "in": "header", "schema": {"type": "string"}},
                    ]
                },
            },
            "/spend/logs/ui/{request_id}": {
                "get": {"parameters": [{"name": "request_id", "in": "path", "required": True, "schema": {"type": "string"}}]}
            },
            "/not-enabled": {"get": {}},
        },
        "components": {
            "schemas": {
                "Member": {
                    "type": "object",
                    "required": ["team_id", "member"],
                    "properties": {
                        "team_id": {"type": "string", "minLength": 1},
                        "member": {"$ref": "#/components/schemas/User", "description": "Preserve this sibling"},
                    },
                    "additionalProperties": False,
                },
                "User": {
                    "type": "object",
                    "required": ["user_email", "role"],
                    "properties": {
                        "user_email": {"type": "string", "minLength": 3},
                        "role": {"enum": ["admin", "user"]},
                    },
                    "additionalProperties": False,
                },
            },
            "parameters": {"Page": {"name": "page", "in": "query", "schema": {"type": "integer", "minimum": 10}}},
        },
    }


def _find(spec: JsonValue, name: str) -> Tool:
    result: Final = build_catalog(spec)
    assert isinstance(result, tuple), result
    return next(tool for tool in result if tool.operation.name == name)


def test_native_schema_keeps_nested_required_enum_and_ref_siblings() -> None:
    spec: Final = _spec()
    before: Final = copy.deepcopy(spec)
    tool: Final = _find(spec, "team_member_add")
    valid: Final[dict[str, JsonValue]] = {"body": {"team_id": "team", "member": {"user_email": "a@b.c", "role": "admin"}}}
    assert validate_arguments(tool, valid) == valid
    assert isinstance(validate_arguments(tool, {"body": {"team_id": "team", "member": {"role": "admin"}}}), ArgumentsError)
    assert isinstance(validate_arguments(tool, {"body": {"team_id": "team", "member": {"user_email": "a@b.c", "role": "owner"}}}), ArgumentsError)
    assert isinstance(validate_arguments(tool, {}), ArgumentsError)
    assert "Preserve this sibling" in json.dumps(tool.parameters)
    assert spec == before


def test_only_fixed_routes_with_grouped_parameters_are_callable() -> None:
    tools: Final = build_catalog(_spec())
    assert isinstance(tools, tuple)
    assert {tool.operation.name for tool in tools} == {"keys_list", "team_member_add", "request_log_detail"}
    tool: Final = _find(_spec(), "keys_list")
    assert tool_request(tool, {"query": {"page": 1, "team_ids": ["a", "b"], "return_full_object": False}}) == ToolRequest(
        "/key/list", (("page", "1"), ("team_ids", "a"), ("team_ids", "b"), ("return_full_object", "false")), None
    )
    for args in ({"query": {"page": 0}}, {"url": "https://example.com"}, {"query": {"Authorization": "secret"}}, {"headers": {}}, {"query": {"page": 101}}):
        assert isinstance(tool_request(tool, args), ArgumentsError), args


@pytest.mark.parametrize("identifier", ["..", ".", "", "../secret", "a/b", "a\\b", "bad\x00id", "bad\r\nid"])
def test_path_values_cannot_escape_the_fixed_route(identifier: str) -> None:
    tool: Final = _find(_spec(), "request_log_detail")
    assert isinstance(tool_request(tool, {"path": {"request_id": identifier}}), ArgumentsError)


def test_path_values_are_encoded_without_changing_host_or_query() -> None:
    tool: Final = _find(_spec(), "request_log_detail")
    assert tool_request(tool, {"path": {"request_id": "request?extra=true#fragment"}}) == ToolRequest(
        "/spend/logs/ui/request%3Fextra%3Dtrue%23fragment", (), None
    )


@pytest.mark.parametrize("reference", ["https://example.com/schema", "#/components/schemas/Missing", "file:///etc/passwd"])
def test_external_and_missing_references_fail_closed(reference: str) -> None:
    assert isinstance(build_catalog(_spec({"$ref": reference})), CatalogError)


def test_schema_and_argument_limits_fail_before_execution() -> None:
    assert isinstance(build_catalog(_spec({"type": "object", "description": "x" * 120_001})), CatalogError)
    tool: Final = _find(_spec(), "team_member_add")
    assert isinstance(validate_arguments(tool, {"body": {"team_id": "x" * 32_001}}), ArgumentsError)
    assert isinstance(validate_arguments(tool, {"body": {"team_id": float("nan")}}), ArgumentsError)


def test_real_management_openapi_schemas_build_without_static_schema_copies() -> None:
    from litellm.proxy.management_endpoints import (
        budget_management_endpoints,
        internal_user_endpoints,
        key_management_endpoints,
        team_endpoints,
    )
    from litellm.proxy.spend_tracking import spend_management_endpoints

    app: Final = FastAPI()
    for module in (budget_management_endpoints, key_management_endpoints, team_endpoints, internal_user_endpoints, spend_management_endpoints):
        app.include_router(module.router)
    schema: Final = TypeAdapter(JsonValue).validate_python(app.openapi())
    catalog: Final = build_catalog(schema)
    assert isinstance(catalog, tuple), catalog
    for name in ("key_create", "team_member_add", "user_update", "budget_info", "request_logs", "request_log_detail"):
        assert any(tool.operation.name == name for tool in catalog), name
    key_update: Final = next(tool for tool in catalog if tool.operation.name == "key_update")
    assert validate_arguments(key_update, {"body": {"key": "a" * 64, "max_budget": 10}}) == {
        "body": {"key": "a" * 64, "max_budget": 10}
    }
    assert isinstance(validate_arguments(key_update, {"body": {"key": "sk-testcredential123456789012345678901234567890"}}), ArgumentsError)
    for name, group, field, identifier in (
        ("key_info", "query", "key", "a" * 64),
        ("key_delete", "body", "keys", ["a" * 64]),
        ("key_block", "body", "key", "a" * 64),
        ("key_unblock", "body", "key", "a" * 64),
        ("request_logs", "query", "api_key", "a" * 64),
        ("spend_report", "query", "api_key", "a" * 64),
        ("key_spend_report", "query", "api_key", "a" * 64),
    ):
        tool: Final = next(item for item in catalog if item.operation.name == name)
        args: Final[dict[str, JsonValue]] = {group: {field: identifier}}
        assert validate_arguments(tool, args) == args, name
    assert isinstance(validate_arguments(key_update, {"body": {
        "key": "a" * 64, "metadata": {"api_key": "b" * 64}
    }}), ArgumentsError)
    key_create: Final = next(tool for tool in catalog if tool.operation.name == "key_create")
    assert isinstance(validate_arguments(key_create, {"body": {"key": "a" * 64}}), ArgumentsError)
    assert next(tool for tool in catalog if tool.operation.name == "budget_info").operation.mutation is False
    assert all(tool.operation.mutation for tool in catalog if tool.operation.method != "GET" and tool.operation.name != "budget_info")
