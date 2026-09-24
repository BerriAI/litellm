import json

import pytest

from litellm.proxy._experimental.mcp_server.management.catalog import build_catalog
from litellm.proxy._experimental.mcp_server.management.inventory import (
    INVENTORY_PATH,
    catalog_inventory,
)
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter


def _spec(paths, components=None):
    spec = {"openapi": "3.1.0", "info": {"title": "t", "version": "1"}, "paths": paths}
    if components:
        spec["components"] = components
    return spec


def _operation(**kwargs):
    op = {"operationId": "op", "responses": {"200": {"content": {"application/json": {}}}}}
    op.update(kwargs)
    return op


def test_json_tool_with_path_query_and_ref_chain_body():
    spec = _spec(
        {
            "/admin/things/{thing_id}": {
                "post": _operation(
                    operationId="update_thing",
                    summary="Update a thing",
                    description="Longer description.",
                    parameters=[
                        {
                            "name": "thing_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {"name": "verbose", "in": "query", "schema": {"type": "boolean"}},
                    ],
                    requestBody={
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ThingPatch"}}},
                    },
                )
            }
        },
        components={
            "schemas": {
                "ThingPatch": {
                    "type": "object",
                    "properties": {"inner": {"$ref": "#/components/schemas/Inner"}},
                },
                "Inner": {"type": "object", "properties": {"name": {"type": "string"}}},
            }
        },
    )
    catalog = build_catalog(spec)
    tool = catalog.tools["update_thing"]
    assert tool.method == "POST"
    assert tool.path_template == "/admin/things/{thing_id}"
    assert tool.path_param_names == ("thing_id",)
    assert tool.query_param_names == ("verbose",)
    assert tool.has_body is True

    schema = tool.mcp_tool.input_schema
    assert set(schema["properties"]) == {"path", "query", "body"}
    assert schema["required"] == ["path", "body"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["path"]["required"] == ["thing_id"]
    assert schema["properties"]["body"] == {"$ref": "#/$defs/ThingPatch"}
    assert set(schema["$defs"]) == {"ThingPatch", "Inner"}
    assert schema["$defs"]["ThingPatch"]["properties"]["inner"] == {"$ref": "#/$defs/Inner"}

    assert tool.mcp_tool.title == "Update a thing"
    assert "Update a thing" in tool.mcp_tool.description
    assert "Longer description." in tool.mcp_tool.description
    assert tool.mcp_tool.annotations.read_only_hint is False
    assert tool.mcp_tool.annotations.destructive_hint is True
    assert tool.mcp_tool.annotations.idempotent_hint is False


def test_get_and_delete_annotations():
    spec = _spec(
        {
            "/admin/things/{thing_id}": {
                "get": _operation(
                    operationId="read_thing",
                    parameters=[{"name": "thing_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                ),
                "delete": _operation(
                    operationId="drop_thing",
                    parameters=[{"name": "thing_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                ),
            }
        }
    )
    catalog = build_catalog(spec)
    read = catalog.tools["read_thing"].mcp_tool.annotations
    assert read.read_only_hint is True
    assert read.destructive_hint is False
    assert read.idempotent_hint is True
    drop = catalog.tools["drop_thing"].mcp_tool.annotations
    assert drop.read_only_hint is False
    assert drop.destructive_hint is True
    assert drop.idempotent_hint is True


def test_exclusion_reasons():
    spec = _spec(
        {
            "/admin/noid": {"get": {"responses": {"200": {}}}},
            "/chat/completions": {"post": _operation(operationId="chat")},
            "/admin/tagged": {"get": _operation(operationId="tagged", tags=["health"])},
            "/sso/login": {"get": _operation(operationId="sso_login")},
            "/admin/upload": {
                "post": _operation(
                    operationId="upload",
                    requestBody={"content": {"multipart/form-data": {"schema": {"type": "object"}}}},
                )
            },
            "/admin/nobody": {"post": _operation(operationId="nobody")},
            "/admin/weird": {
                "get": _operation(
                    operationId="weird",
                    responses={"200": {"content": {"text/csv": {}}}},
                )
            },
        }
    )
    catalog = build_catalog(spec)
    assert catalog.tools == {}
    assert catalog.exclusions == {
        "GET /admin/noid": "no operationId",
        "POST /chat/completions": "data plane, public or ui route group",
        "GET /admin/tagged": "tag:health",
        "GET /sso/login": "excluded path prefix",
        "POST /admin/upload": "non-JSON request body",
        "POST /admin/nobody": "undeclared request body",
        "GET /admin/weird": "non-JSON response",
    }


def test_duplicate_operation_id_fails_startup_loudly():
    spec = _spec(
        {
            "/admin/a": {"get": _operation(operationId="dup")},
            "/admin/b": {"get": _operation(operationId="dup")},
        }
    )
    with pytest.raises(ValueError, match="duplicate OpenAPI operationId 'dup'"):
        build_catalog(spec)


def test_toolless_path_and_query_sections_omitted():
    spec = _spec({"/admin/simple": {"get": _operation(operationId="simple")}})
    tool = build_catalog(spec).tools["simple"]
    assert tool.mcp_tool.input_schema["properties"] == {}
    assert tool.has_body is False


def test_inventory_matches_checked_in_file():
    # Other suites warm lazy routers on the global app, changing diagnostic
    # tags and WebSocket stubs. Compare the same cold startup as the generator.
    script = """
from litellm.proxy.proxy_server import app
from litellm.proxy._experimental.mcp_server.management.catalog import build_catalog
from litellm.proxy._experimental.mcp_server.management.inventory import render_inventory
print(render_inventory(build_catalog(app.openapi())))
"""
    result = run_child_interpreter(script, timeout=90)
    assert result.returncode == 0, result.stderr
    expected = json.loads(INVENTORY_PATH.read_text())
    actual = json.loads(result.stdout)
    assert actual == expected, (
        "management MCP inventory drifted from the live OpenAPI spec; "
        "regenerate it with: python -m litellm.proxy._experimental.mcp_server.management.inventory --write"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/{mcp_server_name}/mcp",
        "/toolset/{toolset_name}/mcp",
        "/v1/memory",
        "/v1/memory/{key}",
        "/public/complexity_router/fuse_presets",
    ],
)
@pytest.mark.parametrize("method", ["get", "post", "delete"])
def test_transport_and_non_management_paths_excluded(path, method):
    operation = _operation(requestBody={"content": {"application/json": {"schema": {"type": "object"}}}})
    catalog = build_catalog(_spec({path: {method: operation}}))
    assert not catalog.tools
    assert f"{method.upper()} {path}" in catalog.exclusions


@pytest.mark.parametrize("path", ["/key/delete", "/team/delete", "/model/delete", "/budget/delete"])
def test_post_deletions_are_destructive(path):
    operation = _operation(requestBody={"content": {"application/json": {"schema": {"type": "object"}}}})
    tool = build_catalog(_spec({path: {"post": operation}})).tools["op"]
    assert tool.mcp_tool.annotations.destructive_hint is True


@pytest.mark.parametrize("path", ["/key/info", "/team/info", "/user/info", "/model/info", "/v1/mcp/server"])
def test_management_reads_are_not_excluded_by_overlapping_route_groups(path):
    catalog = build_catalog(_spec({path: {"get": _operation()}}))
    assert catalog.tools["op"].path_template == path


def test_inventory_command_writes_current_catalog(tmp_path, monkeypatch, capsys):
    from litellm.proxy._experimental.mcp_server.management import inventory
    from litellm.proxy.proxy_server import app

    target = tmp_path / "inventory.json"
    monkeypatch.setattr(inventory, "INVENTORY_PATH", target)
    assert inventory.main([]) == 2
    assert not target.exists()
    assert inventory.main(["--write"]) == 0
    assert json.loads(target.read_text()) == catalog_inventory(build_catalog(app.openapi()))
    assert str(target) in capsys.readouterr().out


@pytest.mark.parametrize(
    "path",
    [
        "/apply_guardrail",
        "/guardrails/apply_guardrail",
        "/v1/a2a/discover",
        "/policy/templates/enrich/stream",
        "/v1/evals",
        "/v1/evals/eval-id/runs/run-id",
        "/usage/ai/chat",
    ],
)
def test_data_plane_execution_is_excluded_from_management(path):
    spec = _spec(
        {
            path: {
                "post": _operation(
                    operationId="execute_guardrail",
                    tags=["Guardrails"],
                    requestBody={"content": {"application/json": {"schema": {"type": "object"}}}},
                )
            }
        }
    )
    catalog = build_catalog(spec)
    assert not catalog.tools
    assert f"POST {path}" in catalog.exclusions


@pytest.mark.parametrize("tag", ["OpenAI Evals API", "OpenAI Evals API - Runs"])
@pytest.mark.parametrize("method", ["get", "delete"])
def test_provider_eval_resources_are_excluded_from_management(tag, method):
    catalog = build_catalog(_spec({"/provider-eval": {method: _operation(operationId="eval", tags=[tag])}}))
    assert not catalog.tools
    assert f"{method.upper()} /provider-eval" in catalog.exclusions


@pytest.mark.parametrize("required", [True, False], ids=["required-query", "optional-query"])
def test_query_section_requirement_matches_its_parameters(required):
    spec = _spec(
        {
            "/admin/lookup": {
                "get": _operation(
                    operationId="lookup",
                    parameters=[
                        {"name": "resource_id", "in": "query", "required": required, "schema": {"type": "string"}}
                    ],
                )
            }
        }
    )
    schema = build_catalog(spec).tools["lookup"].mcp_tool.input_schema
    assert ("query" in schema["required"]) is required
    assert schema["properties"]["query"]["required"] == (["resource_id"] if required else [])
