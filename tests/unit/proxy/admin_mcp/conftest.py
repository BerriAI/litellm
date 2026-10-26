from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import json
from typing import Final
from typing_extensions import ReadOnly, TypedDict

import anyio
import httpx2
from mcp import ClientSession, types
from mcp.shared.memory import create_client_server_memory_streams
from pydantic import JsonValue
import pytest

from litellm.proxy.admin_mcp._types import JSON_OBJECT, JSON_VALUE, object_value
from litellm.proxy.admin_mcp.config import Config
from litellm.proxy.admin_mcp.gateway import Gateway
from litellm.proxy.admin_mcp.server import create_server


SPEC: Final[dict[str, JsonValue]] = {
    "openapi": "3.1.0",
    "paths": {
        "/key/generate": {
            "post": {
                "operationId": "generate_key_fn_key_generate_post",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Key"}}},
                },
            }
        },
        "/key/list": {
            "get": {
                "operationId": "list_keys_key_list_get",
                "parameters": [
                    {"in": "query", "name": "page", "schema": {"type": "integer", "minimum": 1}},
                    {"in": "query", "name": "size", "schema": {"type": "integer", "minimum": 1, "maximum": 50}},
                ],
            }
        },
        "/model/new": {
            "post": {
                "operationId": "add_new_model_model_new_post",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Deployment"}}},
                },
            }
        },
        "/v1/model/info": {
            "get": {
                "operationId": "model_info_v1_v1_model_info_get",
                "parameters": [
                    {"in": "query", "name": "litellm_model_id", "schema": {"type": "string"}},
                ],
            }
        },
        "/key/{key}/reset_spend": {
            "post": {
                "operationId": "reset_key_spend_fn_key__key__reset_spend_post",
                "parameters": [{"in": "path", "name": "key", "required": True, "schema": {"type": "string"}}],
            }
        },
        "/cache/flushall": {"post": {"operationId": "flush_everything"}},
    },
    "components": {
        "schemas": {
            "Key": {
                "type": "object",
                "properties": {"key_alias": {"type": "string"}, "max_budget": {"type": "number"}},
                "required": ["key_alias"],
                "additionalProperties": False,
            },
            "Deployment": {
                "type": "object",
                "required": ["model_name", "litellm_params", "model_info"],
                "properties": {
                    "model_name": {"type": "string"},
                    "litellm_params": {"$ref": "#/components/schemas/Params"},
                    "model_info": {"type": "object"},
                },
            },
            "Params": {
                "type": "object",
                "required": ["model"],
                "properties": {"model": {"type": "string"}, "litellm_credential_name": {"type": "string"}},
            },
        }
    },
}


class StubGateway:
    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.writes: list[httpx2.Request] = []
        self.keys: list[JsonValue] = []
        self.models: list[JsonValue] = []
        self.status = 200
        self.timeout = False
        self.revoke = False
        self.revoked: set[str] = set()
        self.spec = SPEC

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        token: Final = request.headers.get("authorization", "").removeprefix("Bearer ")
        if request.url.path == "/v2/user/info" and token == "team-key":
            return httpx2.Response(400, json={"error": {"message": "A user-bound key is required"}})
        if request.url.path == "/v2/user/info":
            role: Final = (
                "proxy_admin"
                if token in {"alice-admin", "bob-admin"} and token not in self.revoked
                else "internal_user"
            )
            return httpx2.Response(200, json={"user_id": token, "user_role": role})
        if request.url.path == "/openapi.json":
            return httpx2.Response(200, json=self.spec)
        assert token in {"alice-admin", "bob-admin"}
        assert request.headers["litellm-changed-by"] == token
        if request.method != "GET":
            self.writes.append(request)
            if self.revoke:
                self.revoked.add(token)
            if self.timeout:
                raise httpx2.ReadTimeout("secret upstream message must not escape")
            if self.status != 200:
                return httpx2.Response(
                    self.status, json={"secret": "upstream-secret"}, headers={"Location": "https://untrusted.example"}
                )
        if request.url.path == "/key/generate":
            key_body: Final = JSON_OBJECT.validate_json(request.content)
            value: Final[dict[str, JsonValue]] = {**key_body, "key": "sk-created-for-" + token, "echo": token}
            self.keys.append(value)
            return httpx2.Response(200, json=value)
        if request.url.path == "/key/list":
            return httpx2.Response(200, json={"keys": self.keys})
        if request.url.path == "/model/new":
            model_body: Final = JSON_OBJECT.validate_json(request.content)
            model: Final[dict[str, JsonValue]] = {
                **model_body,
                "model_info": {"id": "model-1"},
                "litellm_params": {
                    **object_value(model_body["litellm_params"]),
                    "api_key": "provider-secret",
                    "aws_session_token": "aws-secret",
                },
            }
            self.models.append(model)
            return httpx2.Response(
                200, json={"model_id": "model-1", "litellm_params": json.dumps(model["litellm_params"])}
            )
        if request.url.path == "/v1/model/info":
            return httpx2.Response(200, json={"data": self.models})
        if request.url.path.endswith("/reset_spend"):
            return httpx2.Response(200, json={"status": "ok"})
        raise AssertionError("Unexpected gateway route")


class SessionOptions(TypedDict, total=False):
    read_only: ReadOnly[bool]
    allowed_tools: ReadOnly[frozenset[str]]
    response_view: ReadOnly[str]
    schema_mode: ReadOnly[str]


@pytest.fixture
def stub() -> StubGateway:
    return StubGateway()


@asynccontextmanager
async def connected(gateway: Gateway, api_key: str = "alice-admin") -> AsyncGenerator[ClientSession, None]:
    server: Final = create_server(gateway, lambda context: api_key)
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as group:
            group.start_soon(server.run, *server_streams, server.create_initialization_options())
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                yield session
            group.cancel_scope.cancel()


@asynccontextmanager
async def session_for(
    stub: StubGateway,
    api_key: str = "alice-admin",
    *,
    read_only: bool = False,
    allowed_tools: frozenset[str] = frozenset(),
    response_view: str = "compact",
    schema_mode: str = "full",
) -> AsyncGenerator[ClientSession, None]:
    config: Final = Config(
        "https://gateway.example.com",
        read_only=read_only,
        allowed_tools=allowed_tools,
        response_view=response_view,
        schema_mode=schema_mode,
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(stub.handle)) as client:
        async with connected(Gateway(config, client), api_key) as session:
            yield session


def data(result: types.CallToolResult) -> JsonValue:
    assert not result.is_error, result.content
    content: Final = result.content[0]
    assert isinstance(content, types.TextContent), content
    return JSON_VALUE.validate_json(content.text)
