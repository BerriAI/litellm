import asyncio
import json

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from litellm.proxy._experimental.mcp_server.management.catalog import build_catalog
from litellm.proxy._experimental.mcp_server.management.dispatcher import (
    Dispatch,
    ManagementRequestContext,
    build_management_asgi_app,
    build_management_client,
    call_tool,
    set_dispatch,
)


def _fixture_app() -> FastAPI:
    app = FastAPI()
    app.state.get_item_hits = 0

    @app.get("/admin/items/{item_id}", operation_id="get_item")
    async def get_item(item_id: str, q: int = 0):
        app.state.get_item_hits += 1
        return {"item_id": item_id, "q": q}

    @app.post("/admin/items", operation_id="make_item")
    async def make_item(payload: dict, request: Request):
        return {"made": payload, "query": dict(request.query_params)}

    @app.delete("/admin/items/{item_id}", operation_id="del_item", status_code=204)
    async def del_item(item_id: str):
        return Response(status_code=204)

    @app.get("/admin/fail", operation_id="fail_route")
    async def fail_route():
        raise HTTPException(status_code=418, detail={"reason": "teapot"})

    @app.get("/admin/empty-error", operation_id="empty_error")
    async def empty_error():
        return Response(status_code=403)

    @app.get("/admin/crash", operation_id="crash")
    async def crash():
        raise RuntimeError("credential sk-test-exception-secret and provider-secret-value")

    @app.get("/admin/not-json", operation_id="not_json")
    async def not_json():
        return Response(content=b"not json", media_type="text/plain")

    @app.get("/admin/slow", operation_id="slow_route")
    async def slow_route():
        await asyncio.sleep(60)

    @app.post("/admin/slow-post", operation_id="slow_post")
    async def slow_post(payload: dict):
        await asyncio.sleep(60)

    @app.get("/admin/huge", operation_id="huge_route")
    async def huge_route():
        return {"blob": "x" * (10 * 1024 * 1024 + 1)}

    @app.post("/admin/huge-post", operation_id="huge_post")
    async def huge_post(payload: dict):
        return {"blob": "x" * (10 * 1024 * 1024 + 1)}

    @app.get("/admin/typed", operation_id="typed_route")
    async def typed_route(size: int):
        return {"size": size}

    @app.get("/admin/whoami", operation_id="whoami")
    async def whoami(request: Request):
        return {
            "authorization": request.headers.get("authorization"),
            "x-litellm-api-key": request.headers.get("x-litellm-api-key"),
            "litellm-changed-by": request.headers.get("litellm-changed-by"),
            "x-request-id": request.headers.get("x-request-id"),
            "x-smuggled": request.headers.get("x-smuggled"),
        }

    return app


@pytest.fixture(autouse=True)
def _dispatch_fixture():
    app = _fixture_app()
    internal_app = build_management_asgi_app(app)
    set_dispatch(
        Dispatch(
            catalog=build_catalog(app.openapi()),
            internal_app=internal_app,
            http_client=build_management_client(internal_app),
        )
    )
    yield app
    set_dispatch(None)


def _ctx(**overrides) -> ManagementRequestContext:
    base = {
        "credential_header": "authorization",
        "credential_value": "Bearer sk-caller",
        "client": ("10.1.2.3", 4321),
        "root_path": "",
        "litellm_changed_by": None,
        "request_id": None,
    }
    base.update(overrides)
    return ManagementRequestContext(**base)


def _text(result) -> str:
    assert result.content, "tool result has no content"
    return result.content[0].text


@pytest.mark.asyncio
async def test_unknown_tool_is_error_result():
    result = await call_tool("does_not_exist", {}, _ctx())
    assert result.is_error is True
    assert "unknown tool" in _text(result)


@pytest.mark.asyncio
async def test_get_parity_status_and_body():
    result = await call_tool("get_item", {"path": {"item_id": "it-1"}, "query": {"q": 7}}, _ctx())
    assert result.is_error is False, _text(result)
    assert result.structured_content == {"item_id": "it-1", "q": 7}
    assert json.loads(_text(result)) == result.structured_content


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [True, False])
async def test_dispatch_uses_current_routes_after_backend_partition(_dispatch_fixture, replacement):
    app = _dispatch_fixture
    app.router.routes = [route for route in app.routes if getattr(route, "path", None) != "/admin/items/{item_id}"]
    if replacement:

        @app.get("/admin/items/{item_id}")
        async def current_item(item_id: str):
            return {"current_item": item_id}

    result = await call_tool("get_item", {"path": {"item_id": "current"}}, _ctx())
    if replacement:
        assert result.is_error is False
        assert result.structured_content == {"current_item": "current"}
    else:
        assert result.is_error is True
        assert "Not Found" in _text(result)
    assert app.state.get_item_hits == 0


@pytest.mark.asyncio
async def test_body_reaches_handler_never_the_query_string():
    result = await call_tool("make_item", {"body": {"name": "n", "q": "not-query"}}, _ctx())
    assert result.is_error is False, _text(result)
    body = json.loads(_text(result))
    assert body["made"] == {"name": "n", "q": "not-query"}
    assert body["query"] == {}


@pytest.mark.asyncio
async def test_path_traversal_rejected():
    for bad in ("a/b", "..", "a\\b"):
        result = await call_tool("get_item", {"path": {"item_id": bad}}, _ctx())
        assert result.is_error is True, bad
        assert "400" in _text(result)


@pytest.mark.asyncio
async def test_missing_path_param_is_400_and_handler_not_hit(_dispatch_fixture):
    result = await call_tool("get_item", {"path": {}}, _ctx())
    assert result.is_error is True
    assert "missing path parameter 'item_id' for tool 'get_item'" in _text(result)
    assert _dispatch_fixture.state.get_item_hits == 0


@pytest.mark.asyncio
async def test_argument_section_validation():
    result = await call_tool("get_item", {"path": {"item_id": "x"}, "header": {"x": 1}}, _ctx())
    assert result.is_error is True
    assert "unexpected argument section" in _text(result)
    result = await call_tool("get_item", {"path": "not-an-object"}, _ctx())
    assert result.is_error is True
    assert "must be an object" in _text(result)
    result = await call_tool("get_item", {"path": {"item_id": "x"}, "body": {"a": 1}}, _ctx())
    assert result.is_error is True
    assert "does not accept a request body" in _text(result)


@pytest.mark.asyncio
async def test_only_allowlisted_headers_forward():
    result = await call_tool(
        "whoami",
        {},
        _ctx(litellm_changed_by="admin@x", request_id="req-9"),
    )
    assert result.is_error is False, _text(result)
    echoed = json.loads(_text(result))
    assert echoed["authorization"] == "Bearer sk-caller"
    assert echoed["litellm-changed-by"] == "admin@x"
    assert echoed["x-request-id"] == "req-9"
    assert echoed["x-smuggled"] is None


@pytest.mark.asyncio
async def test_tool_supplied_headers_cannot_smuggle():
    result = await call_tool(
        "whoami",
        {"headers": {"x-smuggled": "yes"}, "query": {"x-smuggled": "yes"}},
        _ctx(),
    )
    assert result.is_error is True
    assert "unexpected argument section" in _text(result)


@pytest.mark.asyncio
async def test_x_litellm_api_key_credential_forwarded():
    result = await call_tool(
        "whoami",
        {},
        _ctx(credential_header="x-litellm-api-key", credential_value="sk-raw-key"),
    )
    echoed = json.loads(_text(result))
    assert echoed["x-litellm-api-key"] == "sk-raw-key"
    assert echoed["authorization"] is None


@pytest.mark.asyncio
async def test_422_surfaces_as_tool_error_text():
    result = await call_tool("typed_route", {"query": {"size": "not-an-int"}}, _ctx())
    assert result.is_error is True
    body = _text(result)
    assert "422" in body or "size" in body
    detail = json.loads(body)
    assert detail["detail"]


@pytest.mark.asyncio
async def test_http_exception_maps_to_rest_error_body():
    result = await call_tool("fail_route", {}, _ctx())
    assert result.is_error is True
    assert json.loads(_text(result)) == {"detail": {"reason": "teapot"}}


@pytest.mark.asyncio
async def test_204_maps_to_empty_object():
    result = await call_tool("del_item", {"path": {"item_id": "gone"}}, _ctx())
    assert result.is_error is False, _text(result)
    assert result.structured_content == {}
    assert _text(result) == "{}"


@pytest.mark.asyncio
async def test_timeout_reports_unknown_outcome_for_mutation(monkeypatch):
    import litellm.proxy._experimental.mcp_server.management.dispatcher as disp

    monkeypatch.setattr(disp, "_HANDLER_TIMEOUT_SECONDS", 0.05)
    result = await call_tool("slow_post", {"body": {}}, _ctx())
    assert result.is_error is True
    assert "may or may not have completed" in _text(result)


@pytest.mark.asyncio
async def test_timeout_plain_message_for_read(monkeypatch):
    import litellm.proxy._experimental.mcp_server.management.dispatcher as disp

    monkeypatch.setattr(disp, "_HANDLER_TIMEOUT_SECONDS", 0.05)
    result = await call_tool("slow_route", {}, _ctx())
    assert result.is_error is True
    assert "slow_route timed out" in _text(result)
    assert "may or may not" not in _text(result)


@pytest.mark.asyncio
async def test_oversized_response_capped():
    result = await call_tool("huge_route", {}, _ctx())
    assert result.is_error is True
    assert "too large" in _text(result)


@pytest.mark.asyncio
async def test_oversized_mutation_result_reports_completed_but_unreturnable():
    result = await call_tool("huge_post", {"body": {}}, _ctx())
    assert result.is_error is True
    assert "completed but the result exceeded" in _text(result)


@pytest.mark.asyncio
async def test_cancellation_propagates():
    task = asyncio.ensure_future(call_tool("slow_route", {}, _ctx()))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_empty_http_error_is_not_success():
    result = await call_tool("empty_error", {}, _ctx())
    assert result.is_error is True
    assert json.loads(_text(result))["status"] == 403


@pytest.mark.asyncio
async def test_exception_credentials_are_not_logged(caplog):
    result = await call_tool("crash", {}, _ctx())
    assert result.is_error is True
    assert "internal error" in _text(result)
    assert "management MCP tool crash failed" in caplog.text
    assert "sk-test-exception-secret" not in caplog.text
    assert "provider-secret-value" not in caplog.text


@pytest.mark.asyncio
async def test_non_json_success_is_tool_error():
    result = await call_tool("not_json", {}, _ctx())
    assert result.is_error is True
    assert "returned a non-JSON response" in _text(result)


@pytest.mark.asyncio
async def test_non_object_body_rejected_without_dispatch():
    result = await call_tool("make_item", {"body": ["not-an-object"]}, _ctx())
    assert result.is_error is True
    assert "must be an object" in _text(result)
