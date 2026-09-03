import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import Response
from fastapi.routing import APIRoute
from openai import AsyncOpenAI
from starlette.requests import Request

import litellm
from litellm.files.main import openai_files_instance
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.anthropic_endpoints import skills_endpoints
from litellm.proxy.anthropic_endpoints.skills_endpoints import (
    _native_skill_data,
)
from litellm.proxy.anthropic_endpoints.skills_endpoints import (
    router as skills_router,
)
from litellm.proxy.openai_files_endpoints.common_utils import extract_model_param
from litellm.router import Router
from litellm.skills.main import (
    _azure_skills_api_base,
    _native_skill_request,
    _validate_skill_operation,
)
from litellm.types.router import GenericLiteLLMParams

SKILL = {
    "id": "skill_1",
    "created_at": 1,
    "default_version": "1",
    "description": "description",
    "latest_version": "1",
    "name": "test-skill",
    "object": "skill",
}
VERSION = {
    "id": "version_1",
    "created_at": 1,
    "description": "description",
    "name": "test-skill",
    "object": "skill.version",
    "skill_id": "skill_1",
    "version": "1",
}


def _mock_response(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/content"):
        return httpx.Response(200, content=b"skill archive", headers={"content-type": "application/zip"})
    if request.method == "DELETE" and "/versions/" in path:
        return httpx.Response(
            200, json={"id": "skill_1", "deleted": True, "object": "skill.version.deleted", "version": "1"}
        )
    if request.method == "DELETE":
        return httpx.Response(200, json={"id": "skill_1", "deleted": True, "object": "skill.deleted"})
    if path.endswith("/versions") and request.method == "GET":
        return httpx.Response(200, json={"object": "list", "data": [VERSION], "has_more": False})
    if "/versions/" in path or path.endswith("/versions"):
        return httpx.Response(200, json=VERSION)
    if path.endswith("/skills") and request.method == "GET":
        return httpx.Response(200, json={"object": "list", "data": [SKILL], "has_more": False})
    return httpx.Response(200, json=SKILL)


def _native_request(
    body: dict[str, Any] | None = None,
    *,
    method: str = "POST",
    path: str = "/v1/skills/skill_1",
    query_string: bytes = b"",
    path_params: dict[str, str] | None = None,
) -> Request:
    payload = json.dumps(body).encode() if body is not None else b""

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [(b"content-type", b"application/json")],
            "query_string": query_string,
            "path_params": path_params or {"skill_id": "skill_1"},
        },
        receive,
    )


@pytest.fixture
def native_endpoint_harness(monkeypatch: pytest.MonkeyPatch) -> tuple[type, dict[str, Any]]:
    captured: dict[str, Any] = {}

    class FakeProcessor:
        result: Any = {"ok": True}
        error: Exception | None = None
        handled_error: Exception = RuntimeError("handled")

        def __init__(self, data: dict[str, Any]) -> None:
            captured["data"] = data

        async def base_process_llm_request(self, **kwargs: Any) -> Any:
            captured["kwargs"] = kwargs
            if self.error is not None:
                raise self.error
            return self.result

        async def _handle_llm_api_exception(self, **kwargs: Any) -> Exception:
            captured["error"] = kwargs["e"]
            return self.handled_error

    monkeypatch.setattr(skills_endpoints, "ProxyBaseLLMRequestProcessing", FakeProcessor)
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(
            general_settings={},
            llm_router=None,
            proxy_config=None,
            proxy_logging_obj=None,
            select_data_generator=None,
            user_api_base=None,
            user_max_tokens=None,
            user_model=None,
            user_request_timeout=None,
            user_temperature=None,
            version="test",
        ),
    )
    return FakeProcessor, captured


async def _call(operation: str, client: AsyncOpenAI, provider: str = "openai") -> Any:
    common = {
        "custom_llm_provider": provider,
        "client": client,
        "api_base": "https://resource.openai.azure.com/openai/v1" if provider == "azure" else None,
        "extra_headers": {"x-test-header": "present"} if provider == "azure" else None,
        "_skill_operation": operation,
    }
    if operation == "create":
        return await litellm.acreate_skill(files=[("SKILL.md", b"skill")], **common)
    if operation == "update":
        return await litellm.acreate_skill(skill_id="skill_1", default_version=2, **common)
    if operation == "create_version":
        return await litellm.acreate_skill(skill_id="skill_1", files=[("SKILL.md", b"skill")], default=True, **common)
    if operation in {"list", "list_versions"}:
        return await litellm.alist_skills(skill_id="skill_1", after="cursor", limit=2, order="desc", **common)
    if operation in {"get", "content", "version", "version_content"}:
        return await litellm.aget_skill(skill_id="skill_1", version="1", **common)
    return await litellm.adelete_skill(skill_id="skill_1", version="1", **common)


@pytest.mark.asyncio
async def test_openai_sdk_handles_every_native_skill_operation() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _mock_response(request)

    client = AsyncOpenAI(
        api_key="test",
        base_url="https://api.openai.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    operations = (
        "create",
        "list",
        "get",
        "update",
        "delete",
        "content",
        "create_version",
        "list_versions",
        "version",
        "delete_version",
        "version_content",
    )
    try:
        results = [await _call(operation, client) for operation in operations]
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
        await client.close()

    actual_requests = [(request.method, request.url.path) for request in requests]
    expected_requests = [
        ("POST", "/v1/skills"),
        ("GET", "/v1/skills"),
        ("GET", "/v1/skills/skill_1"),
        ("POST", "/v1/skills/skill_1"),
        ("DELETE", "/v1/skills/skill_1"),
        ("GET", "/v1/skills/skill_1/content"),
        ("POST", "/v1/skills/skill_1/versions"),
        ("GET", "/v1/skills/skill_1/versions"),
        ("GET", "/v1/skills/skill_1/versions/1"),
        ("DELETE", "/v1/skills/skill_1/versions/1"),
        ("GET", "/v1/skills/skill_1/versions/1/content"),
    ]
    assert actual_requests == expected_requests, actual_requests
    assert dict(requests[1].url.params) == {"after": "cursor", "limit": "2", "order": "desc"}
    assert json.loads(requests[3].content) == {"default_version": 2}
    assert b'name="files[]"' in requests[0].content
    assert b'name="default"' in requests[6].content
    assert b"SKILL.md" in requests[0].content
    assert dict(requests[7].url.params) == {"after": "cursor", "limit": "2", "order": "desc"}
    assert results[5].response.content == b"skill archive"
    assert results[10].response.content == b"skill archive"
    assert results[0].id == "skill_1"
    assert results[1].data[0].id == "skill_1"
    assert results[4].deleted is True
    assert results[6].skill_id == "skill_1"
    assert results[9].deleted is True


@pytest.mark.asyncio
async def test_azure_does_not_use_foundry_preview_header_with_existing_sdk_client() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _mock_response(request)

    client = AsyncOpenAI(
        api_key="test",
        base_url="https://resource.openai.azure.com/openai/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        await _call("get", client, provider="azure")
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
        await client.close()

    assert "foundry-features" not in requests[0].headers
    assert requests[0].headers["x-test-header"] == "present"


@pytest.mark.asyncio
async def test_router_model_configuration_overrides_request_provider() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _mock_response(request)

    client = AsyncOpenAI(
        api_key="test",
        base_url="https://api.openai.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    router = Router(
        model_list=[
            {
                "model_name": "skills-openai",
                "litellm_params": {
                    "model": "openai/gpt-5",
                    "api_key": "test",
                },
            }
        ]
    )
    try:
        with patch.object(openai_files_instance, "get_openai_client", return_value=client) as get_client:
            await router.acreate_skill(
                model="skills-openai",
                files=[("SKILL.md", b"skill")],
                custom_llm_provider="anthropic",
            )
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
        await client.close()

    assert requests[0].url.path == "/v1/skills"
    assert get_client.call_args.kwargs["api_key"] == "test"


@pytest.mark.parametrize(
    ("body_model", "query", "header_model", "expected"),
    [
        ("body", "model=query", "header", "body"),
        ("", "model=query", "header", "query"),
        (None, "model=query", "header", "query"),
        (None, "", "header", "header"),
        (None, "", None, None),
    ],
)
@pytest.mark.asyncio
async def test_native_endpoint_model_priority(
    body_model: object | None,
    query: str,
    header_model: str | None,
    expected: str | None,
) -> None:
    body = json.dumps({"model": body_model} if body_model is not None else {}).encode()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/skills/skill_1",
            "headers": [
                (b"content-type", b"application/json"),
                *(([(b"x-litellm-model", header_model.encode())]) if header_model else []),
            ],
            "query_string": query.encode(),
            "path_params": {"skill_id": "skill_1"},
        },
        receive,
    )

    data = await _native_skill_data(request, "update")

    if expected is None:
        assert "model" not in data
    else:
        assert data["model"] == expected
    assert data["skill_id"] == "skill_1"
    assert data["custom_llm_provider"] == "openai"
    assert data["_skill_operation"] == "update"


@pytest.mark.asyncio
async def test_native_endpoint_drops_invalid_body_model(native_endpoint_harness: tuple[type, dict[str, Any]]) -> None:
    _, captured = native_endpoint_harness
    endpoint = skills_endpoints._native_skill_route("update", "acreate_skill")

    result = await endpoint(
        _native_request({"model": {"invalid": "type"}}),
        Response(),
        object(),
    )

    assert result == {"ok": True}
    assert "model" not in captured["data"]
    assert captured["data"]["skill_id"] == "skill_1"


@pytest.mark.asyncio
async def test_native_endpoint_returns_provider_content_response(
    native_endpoint_harness: tuple[type, dict[str, Any]],
) -> None:
    processor, _ = native_endpoint_harness
    processor.result = SimpleNamespace(
        response=httpx.Response(
            206,
            content=b"skill archive",
            headers={"content-type": "application/zip", "x-test": "present"},
        )
    )

    result = await skills_endpoints._native_skill_endpoint(
        _native_request(method="GET", path="/v1/skills/skill_1/content"),
        Response(),
        object(),
        "content",
        "aget_skill",
    )

    assert isinstance(result, Response)
    assert result.status_code == 206
    assert result.body == b"skill archive"
    assert result.headers["content-type"] == "application/zip"
    assert result.headers["x-test"] == "present"


@pytest.mark.asyncio
async def test_native_endpoint_maps_invalid_content_response(
    native_endpoint_harness: tuple[type, dict[str, Any]],
) -> None:
    processor, captured = native_endpoint_harness
    processor.result = SimpleNamespace(response=object())
    processor.handled_error = RuntimeError("invalid content response")

    with pytest.raises(RuntimeError, match="invalid content response"):
        await skills_endpoints._native_skill_endpoint(
            _native_request(method="GET", path="/v1/skills/skill_1/content"),
            Response(),
            object(),
            "content",
            "aget_skill",
        )

    assert isinstance(captured["error"], TypeError)


@pytest.mark.asyncio
async def test_native_endpoint_maps_provider_error(
    native_endpoint_harness: tuple[type, dict[str, Any]],
) -> None:
    processor, captured = native_endpoint_harness
    processor.error = ValueError("provider failure")
    processor.handled_error = RuntimeError("mapped provider failure")

    with pytest.raises(RuntimeError, match="mapped provider failure"):
        await skills_endpoints._native_skill_endpoint(
            _native_request({"skill_id": "skill_1"}),
            Response(),
            object(),
            "update",
            "acreate_skill",
        )

    assert isinstance(captured["error"], ValueError)


def test_native_skill_request_rejects_azure_without_api_base() -> None:
    with patch(
        "litellm.llms.azure.common_utils.get_azure_credentials",
        return_value=SimpleNamespace(api_base=None, api_key="test"),
    ):
        with pytest.raises(ValueError, match="api_base is required"):
            _native_skill_request(
                "get",
                {"skill_id": "skill_1"},
                "azure",
                GenericLiteLLMParams(api_key="test"),
                None,  # type: ignore[arg-type]
                None,
                False,
            )


def test_native_skill_request_rejects_uninitialized_openai_client() -> None:
    with (
        patch(
            "litellm.llms.openai.common_utils.get_openai_credentials",
            return_value=SimpleNamespace(api_base=None, api_key="test", organization=None),
        ),
        patch.object(openai_files_instance, "get_openai_client", return_value=None),
    ):
        with pytest.raises(ValueError, match="client is not initialized"):
            _native_skill_request(
                "get",
                {"skill_id": "skill_1"},
                "openai",
                GenericLiteLLMParams(api_key="test"),
                None,  # type: ignore[arg-type]
                None,
                False,
            )


@pytest.mark.parametrize(
    "operation",
    ["update", "content", "create_version", "list_versions", "version", "delete_version", "version_content"],
)
def test_native_only_skill_operations_reject_non_native_providers(operation: str) -> None:
    with pytest.raises(ValueError, match="only supported for OpenAI and Azure OpenAI"):
        _validate_skill_operation(operation, "anthropic")


def test_extract_model_param_ignores_non_string_body_model() -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/skills/skill_1",
            "headers": [],
            "query_string": b"",
        }
    )

    assert extract_model_param(request, {"model": {"unexpected": "type"}}) is None


def test_extract_model_param_falls_back_from_empty_body_model() -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/skills/skill_1",
            "headers": [(b"x-litellm-model", b"header-model")],
            "query_string": b"model=query-model",
        }
    )

    assert extract_model_param(request, {"model": ""}) == "query-model"


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (None, None),
        ("https://resource.openai.azure.com", "https://resource.openai.azure.com"),
        ("https://resource.openai.azure.com/openai", "https://resource.openai.azure.com"),
        ("https://resource.openai.azure.com/openai/v1", "https://resource.openai.azure.com"),
        ("https://resource.openai.azure.com/openai/responses?api-version=preview", "https://resource.openai.azure.com"),
        ("https://resource.openai.azure.com/openai/v1/responses", "https://resource.openai.azure.com"),
    ],
)
def test_azure_skills_api_base(api_base: str | None, expected: str | None) -> None:
    assert _azure_skills_api_base(api_base) == expected


def test_native_routes_are_registered_without_anthropic_response_coercion() -> None:
    expected = {
        ("POST", "/v1/skills/{skill_id}"),
        ("GET", "/v1/skills/{skill_id}/content"),
        ("POST", "/v1/skills/{skill_id}/versions"),
        ("GET", "/v1/skills/{skill_id}/versions"),
        ("GET", "/v1/skills/{skill_id}/versions/{version}"),
        ("DELETE", "/v1/skills/{skill_id}/versions/{version}"),
        ("GET", "/v1/skills/{skill_id}/versions/{version}/content"),
    }
    routes = [route for route in skills_router.routes if isinstance(route, APIRoute)]

    assert expected <= {(method, route.path) for route in routes for method in route.methods}
    assert all(route.response_model is None for route in routes)
