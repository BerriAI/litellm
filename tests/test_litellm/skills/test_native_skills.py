import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi.routing import APIRoute
from openai import AsyncOpenAI
from starlette.requests import Request

import litellm
from litellm.files.main import openai_files_instance
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.anthropic_endpoints.skills_endpoints import (
    _native_skill_data,
)
from litellm.proxy.anthropic_endpoints.skills_endpoints import (
    router as skills_router,
)
from litellm.router import Router
from litellm.skills.main import _azure_skills_api_base

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
async def test_azure_uses_preview_header_with_existing_sdk_client() -> None:
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

    assert requests[0].headers["foundry-features"] == "Skills=V1Preview"
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
        (None, "model=query", "header", "query"),
        (None, "", "header", "header"),
    ],
)
@pytest.mark.asyncio
async def test_native_endpoint_model_priority(
    body_model: str | None,
    query: str,
    header_model: str,
    expected: str,
) -> None:
    body = json.dumps({"model": body_model} if body_model else {}).encode()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/skills/skill_1",
            "headers": [(b"content-type", b"application/json"), (b"x-litellm-model", header_model.encode())],
            "query_string": query.encode(),
            "path_params": {"skill_id": "skill_1"},
        },
        receive,
    )

    data = await _native_skill_data(request, "update")

    assert data["model"] == expected
    assert data["skill_id"] == "skill_1"
    assert data["custom_llm_provider"] == "openai"
    assert data["_skill_operation"] == "update"


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("https://resource.openai.azure.com", "https://resource.openai.azure.com"),
        ("https://resource.openai.azure.com/openai", "https://resource.openai.azure.com"),
        ("https://resource.openai.azure.com/openai/v1", "https://resource.openai.azure.com"),
        ("https://resource.openai.azure.com/openai/responses?api-version=preview", "https://resource.openai.azure.com"),
        ("https://resource.openai.azure.com/openai/v1/responses", "https://resource.openai.azure.com"),
    ],
)
def test_azure_skills_api_base(api_base: str, expected: str) -> None:
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
