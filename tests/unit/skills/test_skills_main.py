import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio
import respx
from openai import OpenAI

import litellm
import litellm.skills.main as skills_main
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import LlmProviders


def test_create_skill_forwards_description_and_instructions_from_top_level_kwargs(
    monkeypatch,
) -> None:
    """The REST /v1/skills form endpoint passes description/instructions as top-level
    kwargs (not extra_body). Regression for a bug where the litellm_proxy dispatch
    branch of create_skill() dropped both, so every LiteLLM-hosted skill was created
    with description=None and instructions=None regardless of what the caller sent."""
    handler = MagicMock()
    monkeypatch.setattr(skills_main, "_get_litellm_skills_handler", lambda: handler)

    skills_main.create_skill(
        display_title="Document Translator",
        description="Converts files from one language into another",
        instructions="Take an uploaded document and produce it in the target language",
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )

    assert handler.create_skill_handler.call_args.kwargs["description"] == (
        "Converts files from one language into another"
    )
    assert handler.create_skill_handler.call_args.kwargs["instructions"] == (
        "Take an uploaded document and produce it in the target language"
    )


def test_create_skill_forwards_description_and_instructions_from_extra_body(monkeypatch) -> None:
    """The SDK convention (see tests/unit/skills/test_skills_db.py) nests them under
    extra_body instead of passing them as top-level kwargs; both paths must reach the DB."""
    handler = MagicMock()
    monkeypatch.setattr(skills_main, "_get_litellm_skills_handler", lambda: handler)

    skills_main.create_skill(
        display_title="Warehouse SQL Analyst",
        extra_body={"description": "Runs SQL against the inventory database", "instructions": "Summarize results"},
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )

    assert handler.create_skill_handler.call_args.kwargs["description"] == ("Runs SQL against the inventory database")
    assert handler.create_skill_handler.call_args.kwargs["instructions"] == "Summarize results"


def test_create_skill_without_description_or_instructions_passes_none(monkeypatch) -> None:
    handler = MagicMock()
    monkeypatch.setattr(skills_main, "_get_litellm_skills_handler", lambda: handler)

    skills_main.create_skill(display_title="Bare Skill", custom_llm_provider=LlmProviders.LITELLM_PROXY.value)

    assert handler.create_skill_handler.call_args.kwargs["description"] is None
    assert handler.create_skill_handler.call_args.kwargs["instructions"] is None


NATIVE_SKILL: Final = {
    "id": "skill_1",
    "object": "skill",
    "created_at": 1,
    "name": "test-skill",
    "description": "Test skill",
    "default_version": "1",
    "latest_version": "1",
}


@pytest_asyncio.fixture
async def native_skill_provider(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[list[httpx.Request]]:
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "skill_1", "object": "skill.deleted", "deleted": True})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"archive", headers={"content-type": "application/zip"})
        if request.method == "GET" and request.url.path.endswith("/skills"):
            return httpx.Response(200, json={"object": "list", "data": [NATIVE_SKILL], "has_more": False})
        return httpx.Response(200, json=NATIVE_SKILL)

    respx_mock.route(host="skills-provider.test").mock(side_effect=respond)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respx_mock.handler)) as provider_http:
        monkeypatch.setattr(litellm, "aclient_session", provider_http)
        yield requests
        await GLOBAL_LOGGING_WORKER.flush()


@pytest.mark.parametrize("provider", ("openai", "azure"))
@pytest.mark.parametrize("operation", ("create", "list", "get", "delete"))
def test_native_skill_sync_entrypoints_use_existing_sdk_factories(
    native_skill_provider: list[httpx.Request], provider: str, operation: str
) -> None:
    parameters: Final = {
        "custom_llm_provider": provider,
        "api_base": "https://skills-provider.test" if provider == "azure" else "https://skills-provider.test/v1",
        "api_key": "test-native-key",
        "max_retries": 0,
        "extra_headers": {"x-extra-header": "test"},
    }
    if operation == "create":
        response = litellm.create_skill(files=[("test-skill/SKILL.md", b"manifest")], **parameters)
    elif operation == "list":
        response = litellm.list_skills(limit=1, after="cursor", order="asc", **parameters)
    elif operation == "get":
        response = litellm.get_skill("skill_1", **parameters)
    else:
        response = litellm.delete_skill("skill_1", **parameters)
    if operation == "list":
        assert response.data[0].id == "skill_1"
        assert dict(native_skill_provider[-1].url.params) == {"limit": "1", "after": "cursor", "order": "asc"}
    else:
        assert response.id == "skill_1"
    assert native_skill_provider[-1].headers["authorization"] == "Bearer test-native-key"
    assert native_skill_provider[-1].headers["x-extra-header"] == "test"
    assert "Foundry-Features" not in native_skill_provider[-1].headers
    assert native_skill_provider[-1].url.path.startswith("/openai/v1/skills" if provider == "azure" else "/v1/skills")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix",
    ("", "/openai", "/openai/v1", "/openai/v1/", "/openai/responses", "/openai/v1/responses?api-version=old"),
)
async def test_native_skill_azure_resource_url_normalizes_existing_endpoint_variants(
    native_skill_provider: list[httpx.Request], suffix: str
) -> None:
    skill: Final = await litellm.aget_skill(
        "skill_1",
        custom_llm_provider="azure",
        api_base=f"https://skills-provider.test/gateway{suffix}",
        api_key="test-azure-key",
        max_retries=0,
    )
    assert skill.id == "skill_1"
    assert native_skill_provider[-1].url.path == "/gateway/openai/v1/skills/skill_1"
    assert "api-version" not in native_skill_provider[-1].url.params
    await GLOBAL_LOGGING_WORKER.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", (False, True))
async def test_native_skill_azure_existing_token_provider_refreshes_for_each_request(
    native_skill_provider: list[httpx.Request], monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    for name in ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "azure_key", None)
    tokens: Final = iter(("first-token", "second-token"))

    def token_provider() -> str:
        return next(tokens)

    parameters: Final = {
        "custom_llm_provider": "azure",
        "api_base": "https://skills-provider.test",
        "max_retries": 0,
        "azure_ad_token_provider": token_provider,
    }
    if is_async:
        first = await litellm.aget_skill("skill_1", **parameters)
        second = await litellm.aget_skill("skill_1", **parameters)
    else:
        first = await asyncio.to_thread(litellm.get_skill, "skill_1", **parameters)
        second = await asyncio.to_thread(litellm.get_skill, "skill_1", **parameters)
    assert first.id == second.id == "skill_1"
    assert [request.headers["authorization"] for request in native_skill_provider] == [
        "Bearer first-token",
        "Bearer second-token",
    ]
    await GLOBAL_LOGGING_WORKER.flush()


@pytest.mark.asyncio
async def test_native_skill_azure_existing_static_entra_token_authentication(
    native_skill_provider: list[httpx.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "azure_key", None)
    skill: Final = await litellm.aget_skill(
        "skill_1",
        custom_llm_provider="azure",
        api_base="https://skills-provider.test",
        max_retries=0,
        azure_ad_token="test-entra-token",
    )
    assert skill.id == "skill_1"
    assert native_skill_provider[-1].headers["authorization"] == "Bearer test-entra-token"
    await GLOBAL_LOGGING_WORKER.flush()


@pytest.mark.asyncio
async def test_native_skill_async_list_accepts_a_supplied_sync_sdk_client(
    native_skill_provider: list[httpx.Request], respx_mock: respx.MockRouter
) -> None:
    with OpenAI(
        api_key="test-supplied-key",
        base_url="https://skills-provider.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respx_mock.handler)),
    ) as supplied_client:
        result: Final = await litellm.alist_skills(custom_llm_provider="openai", client=supplied_client)
        assert result.data[0].id == NATIVE_SKILL["id"]
        assert not supplied_client.is_closed()
        assert native_skill_provider[-1].headers["authorization"] == "Bearer test-supplied-key"


@pytest.mark.parametrize("api_base", (None, ""))
def test_native_skill_azure_missing_resource_url_fails_before_provider_http(
    native_skill_provider: list[httpx.Request], monkeypatch: pytest.MonkeyPatch, api_base: str | None
) -> None:
    monkeypatch.setattr(litellm, "api_base", None)
    if api_base is None:
        monkeypatch.delenv("AZURE_API_BASE", raising=False)
    else:
        monkeypatch.setenv("AZURE_API_BASE", api_base)
    with pytest.raises(litellm.APIConnectionError, match="api_base is required"):
        litellm.get_skill("skill_1", custom_llm_provider="azure", api_key="test-native-key", api_base=api_base)
    assert not native_skill_provider


@pytest.mark.parametrize("provider", ("anthropic", "litellm_proxy"))
@pytest.mark.parametrize(
    ("entrypoint", "operation", "kwargs"),
    (
        (litellm.create_skill, "update", {"skill_id": "skill_1", "default_version": "1"}),
        (litellm.create_skill, "create_version", {"skill_id": "skill_1", "files": []}),
        (litellm.list_skills, "list_versions", {"skill_id": "skill_1"}),
        (litellm.get_skill, "version", {"skill_id": "skill_1", "version": "1"}),
        (litellm.get_skill, "content", {"skill_id": "skill_1"}),
        (litellm.delete_skill, "delete_version", {"skill_id": "skill_1", "version": "1"}),
        (litellm.get_skill, "version_content", {"skill_id": "skill_1", "version": "1"}),
    ),
)
def test_native_only_skill_operations_reject_non_native_providers_before_http(
    native_skill_provider: list[httpx.Request], provider: str, entrypoint: Callable, operation: str, kwargs: dict
) -> None:
    with pytest.raises(litellm.BadRequestError, match="only supported for OpenAI and Azure OpenAI"):
        entrypoint(custom_llm_provider=provider, _skill_operation=operation, **kwargs)
    assert not native_skill_provider


@pytest.mark.asyncio
async def test_existing_anthropic_crud_remains_the_default_provider(respx_mock: respx.MockRouter) -> None:
    requests: Final[list[httpx.Request]] = []
    payload: Final = {
        "id": "skill_1",
        "type": "skill",
        "display_title": "Existing skill",
        "source": "custom",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "latest_version": "1",
    }

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "skill_1", "type": "skill_deleted"})
        if request.method == "GET" and request.url.path.endswith("/skills"):
            return httpx.Response(200, json={"data": [payload], "has_more": False, "next_page": None})
        return httpx.Response(200, json=payload)

    respx_mock.route(host="anthropic-provider.test").mock(side_effect=respond)
    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respx_mock.handler))
    parameters: Final = {
        "api_key": "test-anthropic-key",
        "api_base": "https://anthropic-provider.test",
        "client": handler,
    }
    try:
        created: Final = await litellm.acreate_skill(
            display_title="Existing skill", files=[("test-skill/SKILL.md", b"manifest")], **parameters
        )
        listed: Final = await litellm.alist_skills(limit=2, page="cursor", source="custom", **parameters)
        retrieved: Final = await litellm.aget_skill(created.id, **parameters)
        deleted: Final = await litellm.adelete_skill(created.id, **parameters)
        assert created.display_title == retrieved.display_title == "Existing skill"
        assert listed.data[0].id == deleted.id == created.id
        assert [request.method for request in requests] == ["POST", "GET", "GET", "DELETE"]
        assert all(request.headers["x-api-key"] == "test-anthropic-key" for request in requests)
        assert all("skills" in request.headers["anthropic-beta"] for request in requests)
        assert dict(requests[1].url.params) == {"limit": "2", "page": "cursor", "source": "custom"}
        assert b"manifest" in requests[0].content
    finally:
        await handler.close()
        await GLOBAL_LOGGING_WORKER.flush()
