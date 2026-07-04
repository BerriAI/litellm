"""
OpenAPI documentation test for ``POST /mcp-rest/tools/call`` (issue #32121).

The endpoint used to read its body as an untyped ``await request.json()``, so
FastAPI generated no request-body schema — the OpenAPI spec listed no params,
which broke clients (and LLMs) generating MCP tool calls from the spec. It now
declares an ``MCPToolCallRequest`` body model, so the params are documented.
"""

import pytest
from fastapi import FastAPI

from litellm.proxy._experimental.mcp_server import rest_endpoints

pytestmark = pytest.mark.skipif(
    not rest_endpoints.MCP_AVAILABLE,
    reason="MCP REST routes only register when the `mcp` package is available",
)


def _tools_call_post_spec():
    app = FastAPI()
    app.include_router(rest_endpoints.router)
    spec = app.openapi()
    return spec, spec["paths"]["/mcp-rest/tools/call"]["post"]


def test_tools_call_documents_request_body():
    spec, post = _tools_call_post_spec()

    assert "requestBody" in post, "POST /mcp-rest/tools/call must document a request body"
    schema = post["requestBody"]["content"]["application/json"]["schema"]
    # Optional body -> FastAPI may emit a direct $ref or an anyOf wrapping it.
    refs = [schema["$ref"]] if "$ref" in schema else [s["$ref"] for s in schema.get("anyOf", []) if "$ref" in s]
    assert any(r.endswith("/MCPToolCallRequest") for r in refs), schema

    props = spec["components"]["schemas"]["MCPToolCallRequest"]["properties"]
    assert {"server_id", "name", "arguments"} <= set(props)


def test_tool_call_request_model_allows_extra_and_optional_fields():
    # Fields are optional + extra keys allowed, so the tool-search / tool-call
    # helper flows that share this route are not rejected with 422.
    model = rest_endpoints.MCPToolCallRequest(server_id="s1", name="t", extra_key="ok")
    dumped = model.model_dump()
    assert dumped["server_id"] == "s1"
    assert dumped["name"] == "t"
    assert dumped["extra_key"] == "ok"
    # all declared fields default to None when omitted
    assert rest_endpoints.MCPToolCallRequest().model_dump() == {
        "server_id": None,
        "name": None,
        "arguments": None,
    }
