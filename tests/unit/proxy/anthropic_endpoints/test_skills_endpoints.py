import gzip
import json
from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
import respx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from openai import APIConnectionError, AsyncOpenAI

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.litellm_proxy.skills.skill_search import skill_search_text
from litellm.proxy._types import LiteLLM_SkillsTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.skills_endpoints import router
from litellm.proxy.auth.auth_checks import route_skips_budget_checks
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient
from litellm.router import Router

TRANSLATOR: Final = LiteLLM_SkillsTable(
    skill_id="translate-file",
    display_title="Document Translator",
    description="Converts files from one language into another",
    instructions="Take an uploaded document and produce it in the target language",
)
SQL_ANALYST: Final = LiteLLM_SkillsTable(
    skill_id="warehouse-sql-analyst",
    display_title="Warehouse SQL Analyst",
    description="Runs SQL against the inventory database",
)
TRIP_PLANNER: Final = LiteLLM_SkillsTable(
    skill_id="trip-planner",
    display_title="Trip Planner",
    description="Books flights and hotels",
)
SKILLS: Final = (TRANSLATOR, SQL_ANALYST, TRIP_PLANNER)

VECTORS: Final = MappingProxyType(
    {
        "language translation": (1.0, 0.0, 0.0),
        skill_search_text(TRANSLATOR): (0.9, 0.1, 0.0),
        skill_search_text(SQL_ANALYST): (0.0, 1.0, 0.0),
        skill_search_text(TRIP_PLANNER): (0.3, 0.0, 1.0),
    }
)


def _client(role: LitellmUserRoles) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="u", user_role=role)
    return TestClient(app)


@pytest.fixture
def accessible_skills(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    list_for_search = AsyncMock(return_value=list(SKILLS))
    monkeypatch.setattr(
        "litellm.llms.litellm_proxy.skills.handler.LiteLLMSkillsHandler.list_skills_for_search", list_for_search
    )
    return list_for_search


@pytest.fixture
def key_limits(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    limits = MagicMock()
    limits.pre_call_hook = AsyncMock(side_effect=lambda user_api_key_dict, data, call_type: data)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", limits)
    return limits


@pytest.fixture
def embedding_router(monkeypatch: pytest.MonkeyPatch, key_limits: MagicMock) -> MagicMock:
    embedding_router = MagicMock()
    embedding_router.aembedding = AsyncMock(
        side_effect=lambda model, input, metadata: litellm.EmbeddingResponse(
            model=model,
            data=[{"object": "embedding", "index": i, "embedding": list(VECTORS[t])} for i, t in enumerate(input)],
        )
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", embedding_router)
    monkeypatch.setattr(litellm, "skill_search_embedding_model", "text-embedding-3-small")
    return embedding_router


class TestGetSkillsQuery:
    def test_query_ranks_and_scores_and_truncates(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "language translation", "top_k": 2},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 200
        body = response.json()["data"]
        assert [skill["id"] for skill in body] == ["translate-file", "trip-planner"]
        assert body[0]["search_score"] > body[1]["search_score"]
        assert embedding_router.aembedding.await_args.kwargs["metadata"]["user_api_key_user_id"] == "u"

    def test_restricted_key_only_ranks_the_skills_it_can_access(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        accessible_skills.return_value = [SQL_ANALYST]
        response = _client(LitellmUserRoles.INTERNAL_USER).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "language translation"},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 200
        assert [skill["id"] for skill in response.json()["data"]] == ["warehouse-sql-analyst"]

    def test_no_accessible_skills_is_a_no_match_empty_result(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        accessible_skills.return_value = []
        response = _client(LitellmUserRoles.INTERNAL_USER).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "anything"},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 200
        assert response.json()["data"] == []
        embedding_router.aembedding.assert_not_awaited()

    def test_query_is_unsupported_for_the_anthropic_passthrough_provider(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills", params={"query": "anything"}, headers={"Authorization": "Bearer k"}
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "skill_search_unsupported_provider"
        accessible_skills.assert_not_awaited()

    def test_missing_embedding_model_is_a_400(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "skill_search_embedding_model", None)
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "anything"},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "skill_search_not_configured"

    def test_embedding_provider_failure_is_a_503(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        embedding_router.aembedding = AsyncMock(side_effect=APIConnectionError(request=MagicMock()))
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "anything"},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 503
        assert response.json()["detail"]["error"] == "skill_search_unavailable"

    def test_a_key_over_its_rate_limit_gets_a_429_without_embedding(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock, key_limits: MagicMock
    ) -> None:
        key_limits.pre_call_hook = AsyncMock(side_effect=ProxyRateLimitError(detail="rpm exceeded"))
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "language translation"},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 429
        embedding_router.aembedding.assert_not_awaited()

    def test_top_k_is_validated(self, accessible_skills: AsyncMock, embedding_router: MagicMock) -> None:
        response = _client(LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/skills",
            params={"custom_llm_provider": "litellm_proxy", "query": "anything", "top_k": 0},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 422


@pytest_asyncio.fixture
async def native_skill_clients(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> AsyncIterator[tuple[AsyncOpenAI, list[httpx.Request]]]:
    from litellm.proxy import proxy_server

    requests: Final[list[httpx.Request]] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/content"):
            return httpx.Response(
                200,
                content=gzip.compress(b"PK skill archive"),
                headers={
                    "content-type": "application/zip",
                    "content-encoding": "gzip",
                    "content-disposition": 'attachment; filename="skill.zip"',
                },
            )
        skill: Final = {
            "id": "skill_1",
            "object": "skill",
            "created_at": 1,
            "name": "test-skill",
            "description": "Test skill",
            "default_version": "1",
            "latest_version": "1",
            "native_extension": {"kept": True},
        }
        version: Final = {
            "id": "version_2",
            "object": "skill.version",
            "created_at": 1,
            "name": "test-skill",
            "description": "Test skill",
            "skill_id": "skill_1",
            "version": "2",
            "native_extension": {"kept": True},
        }
        if request.method == "DELETE":
            return httpx.Response(
                200,
                json={
                    "id": "skill_1",
                    "object": "skill.version.deleted" if "/versions/" in request.url.path else "skill.deleted",
                    "deleted": True,
                    "version": "1",
                },
            )
        if request.method == "GET" and request.url.path.endswith(("/skills", "/versions")):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [version if request.url.path.endswith("/versions") else skill],
                    "has_more": False,
                },
            )
        return httpx.Response(200, json=version if "/versions" in request.url.path else skill)

    respx_mock.route(host="provider.test").mock(side_effect=provider)
    respx_mock.route(host="other-account.test").mock(side_effect=provider)

    async def proxy_request(request: httpx.Request) -> None:
        await request.aread()
        requests.append(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as provider_http:
        monkeypatch.setattr(litellm, "aclient_session", provider_http)
        model_router: Final = Router(
            model_list=[
                {
                    "model_name": alias,
                    "litellm_params": {
                        "model": "openai/gpt-5.5",
                        "api_key": alias,
                        "api_base": "https://provider.test/v1",
                        "organization": f"{alias}-org",
                        "extra_headers": {"x-deployment-route": alias},
                        "max_retries": 0,
                    },
                }
                for alias in ("body-route", "query-route", "header-route")
            ]
            + [
                {
                    "model_name": alias,
                    "litellm_params": {
                        "model": "azure/deployment",
                        "api_key": alias if alias == "azure-route" else None,
                        "azure_ad_token": alias if alias == "azure-ad-route" else None,
                        "api_base": "https://provider.test/openai/v1",
                        "api_version": "2025-04-01-preview",
                        "extra_headers": {"x-deployment-route": alias},
                        "max_retries": 0,
                    },
                }
                for alias in ("azure-route", "azure-ad-route")
            ],
            num_retries=0,
        )
        monkeypatch.setattr(proxy_server, "llm_router", model_router)
        app: Final = FastAPI(exception_handlers=proxy_server.app.exception_handlers)
        app.include_router(router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="skills-test")
        async with AsyncOpenAI(
            base_url="http://proxy.test/v1",
            api_key="sk-native-skills-test",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), event_hooks={"request": [proxy_request]}
            ),
        ) as sdk:
            yield sdk, requests
            await GLOBAL_LOGGING_WORKER.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("GET", "/v1/skills/skill_1/versions"),
        ("DELETE", "/v1/skills/skill_1/versions/1"),
    ),
)
async def test_native_skill_get_and_delete_use_body_model_before_query_and_header(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], method: str, path: str
) -> None:
    sdk, requests = native_skill_clients
    response: Final = await sdk._client.request(
        method,
        f"http://proxy.test{path}?model=query-route",
        headers={"x-litellm-model": "header-route"},
        json={"model": "body-route"},
    )
    assert response.status_code == 200
    assert requests[-1].headers["authorization"] == "Bearer body-route"
    assert "model" not in requests[-1].url.params


@pytest.mark.asyncio
async def test_native_skill_content_is_decoded_once_and_keeps_proxy_headers(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]],
) -> None:
    sdk, requests = native_skill_clients
    content: Final = await sdk.skills.content.retrieve("skill_1", extra_headers={"x-litellm-model": "header-route"})
    assert content.content == b"PK skill archive"
    assert content.response.headers["content-length"] == str(len(content.content))
    assert content.response.headers["content-disposition"] == 'attachment; filename="skill.zip"'
    assert content.response.headers["x-litellm-call-id"]
    assert requests[-1].headers["authorization"] == "Bearer header-route"


@pytest.mark.asyncio
async def test_native_skill_content_rejects_incompatible_post_call_replacement(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], monkeypatch: pytest.MonkeyPatch
) -> None:
    class ReplaceContent(CustomLogger):
        async def async_post_call_success_hook(
            self, data: Mapping[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
        ) -> litellm.ModelResponse:
            return litellm.ModelResponse(choices=[])

    monkeypatch.setattr(litellm, "callbacks", [ReplaceContent()])
    sdk, requests = native_skill_clients
    response: Final = await sdk._client.get(
        "http://proxy.test/v1/skills/skill_1/content", headers={"x-litellm-model": "header-route"}
    )
    assert response.status_code == 500
    assert "Skills content response did not contain an HTTP response" in response.json()["error"]["message"]
    assert requests[-1].headers["authorization"] == "Bearer header-route"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("header-route", "azure-route"))
@pytest.mark.parametrize("path", ("/v1/skills", "/v1/skills/skill_1/versions"))
async def test_native_skill_zip_upload_preserves_archive_bytes(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], model: str, path: str
) -> None:
    sdk, requests = native_skill_clients
    response: Final = await sdk._client.post(
        f"http://proxy.test{path}",
        files={"files": ("skill.zip", b"PK archive", "application/zip")},
        data={"default": "true"} if path.endswith("/versions") else None,
        headers={"x-litellm-model": model},
    )
    assert response.status_code == 200
    assert b'name="files";' in requests[0].content
    assert b'name="files";' in requests[-1].content
    assert b"PK archive" in requests[-1].content
    assert requests[-1].headers["authorization"] == f"Bearer {model}"
    if path.endswith("/versions"):
        assert b'name="default"\r\n\r\ntrue' in requests[-1].content


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("header-route", "azure-route"))
async def test_native_skill_sdk_crud_versions_and_content_use_deployment_credentials(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], model: str
) -> None:
    sdk, requests = native_skill_clients
    headers: Final = {"x-litellm-model": model}
    files: Final = [
        ("test-skill/SKILL.md", b"manifest", "text/markdown"),
        ("test-skill/tool.py", b"script", "text/plain"),
    ]
    created: Final = await sdk.skills.create(files=files, extra_headers=headers)
    assert created.name == "test-skill"
    assert created.model_extra == {"native_extension": {"kept": True}}
    listed: Final = await sdk.skills.list(limit=2, after="cursor", order="desc", extra_headers=headers)
    assert listed.data[0].id == created.id
    retrieved: Final = await sdk.skills.retrieve(created.id, extra_headers=headers)
    assert retrieved.default_version == "1"
    assert retrieved.model_extra == created.model_extra
    version: Final = await sdk.skills.versions.create(created.id, files=files, default=True, extra_headers=headers)
    assert version.skill_id == created.id
    assert version.model_extra == created.model_extra
    versions: Final = await sdk.skills.versions.list(created.id, limit=2, extra_headers=headers)
    assert versions.data[0].version == version.version
    retrieved_version: Final = await sdk.skills.versions.retrieve(
        version.version, skill_id=created.id, extra_headers=headers
    )
    assert retrieved_version.id == version.id
    updated: Final = await sdk.skills.update(created.id, default_version=version.version, extra_headers=headers)
    assert updated.id == created.id
    content: Final = await sdk.skills.content.retrieve(created.id, extra_headers=headers)
    assert content.content == b"PK skill archive"
    version_content: Final = await sdk.skills.versions.content.retrieve(
        version.version, skill_id=created.id, extra_headers=headers
    )
    assert version_content.content == content.content
    deleted_version: Final = await sdk.skills.versions.delete("1", skill_id=created.id, extra_headers=headers)
    assert deleted_version.deleted
    deleted: Final = await sdk.skills.delete(created.id, extra_headers=headers)
    assert deleted.deleted
    forwarded: Final = tuple(request for request in requests if request.url.host == "provider.test")
    assert len(forwarded) == 11
    assert all(request.headers["authorization"] == f"Bearer {model}" for request in forwarded)
    assert all("model" not in request.url.params for request in forwarded)
    assert b"manifest" in forwarded[0].content and b"script" in forwarded[0].content
    assert b"manifest" in forwarded[3].content and b"script" in forwarded[3].content
    assert dict(forwarded[1].url.params) == {"limit": "2", "after": "cursor", "order": "desc"}
    assert all("/deployments/" not in request.url.path for request in forwarded)
    expected_prefix: Final = "/openai/v1/skills" if model == "azure-route" else "/v1/skills"
    assert all(request.url.path.startswith(expected_prefix) for request in forwarded)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", (42, 0, False, {"alias": "body-route"}, ["body-route"]))
@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("GET", "/v1/skills"),
        ("GET", "/v1/skills/skill_1"),
        ("DELETE", "/v1/skills/skill_1"),
        ("POST", "/v1/skills/skill_1"),
    ),
)
async def test_native_skill_invalid_body_model_does_not_fall_back_to_another_account(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], model: object, method: str, path: str
) -> None:
    sdk, requests = native_skill_clients
    response: Final = await sdk._client.request(
        method,
        f"http://proxy.test{path}?model=query-route",
        json={"model": model, "default_version": "1"},
        headers={"x-litellm-model": "header-route"},
    )
    assert response.status_code == 400
    assert not any(request.url.host == "provider.test" for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("POST", "/v1/skills/skill_1/versions"),
        ("GET", "/v1/skills/skill_1/versions"),
        ("GET", "/v1/skills"),
        ("GET", "/v1/skills/skill_1"),
        ("DELETE", "/v1/skills/skill_1"),
    ),
)
@pytest.mark.parametrize("body", ([], None, "not an object"))
async def test_native_skill_non_object_body_is_rejected_without_forwarding(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], method: str, path: str, body: object
) -> None:
    sdk, requests = native_skill_clients
    response: Final = await sdk._client.request(
        method,
        f"http://proxy.test{path}?model=header-route",
        content=json.dumps(body),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert not any(request.url.host == "provider.test" for request in requests)


@pytest.mark.asyncio
async def test_native_skill_unknown_stream_flag_cannot_change_response_protocol(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]],
) -> None:
    sdk, requests = native_skill_clients
    response: Final = await sdk._client.post(
        "http://proxy.test/v1/skills/skill_1?model=header-route",
        json={"default_version": "1", "stream": True},
    )
    assert response.status_code == 200
    assert response.json()["id"] == "skill_1"
    assert response.headers["content-type"] == "application/json"
    assert "stream" not in requests[-1].content.decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", (400, 401, 403, 404, 409, 422, 429, 500, 503))
@pytest.mark.parametrize("path", ("/v1/skills/skill_1", "/v1/skills/skill_1/versions/1/content"))
async def test_native_skill_provider_errors_preserve_http_status(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]],
    respx_mock: respx.MockRouter,
    status_code: int,
    path: str,
) -> None:
    sdk, requests = native_skill_clients
    provider_error: Final = respx_mock.route(host="provider.test").respond(
        status_code, json={"error": {"message": "provider rejected request", "type": "invalid_request_error"}}
    )
    response: Final = await sdk._client.get(f"http://proxy.test{path}", headers={"x-litellm-model": "header-route"})
    assert response.status_code == status_code
    assert "provider rejected request" in response.json()["error"]["message"]
    assert provider_error.call_count == 1


@pytest.mark.asyncio
async def test_native_skill_path_and_operation_cannot_be_overridden_by_payload(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]],
) -> None:
    sdk, requests = native_skill_clients
    retrieved: Final = await sdk.skills.versions.retrieve(
        "1",
        skill_id="skill_1",
        extra_headers={"x-litellm-model": "header-route"},
        extra_query={"operation": "delete", "route_type": "adelete_skill", "version": "9", "skill_id": "other"},
        extra_body={"_skill_operation": "delete", "skill_id": "other", "version": "9"},
    )
    assert retrieved.skill_id == "skill_1"
    assert requests[-1].method == "GET"
    assert requests[-1].url.path == "/v1/skills/skill_1/versions/1"


@pytest.mark.asyncio
async def test_native_skill_configured_model_provider_wins_over_request_provider(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]],
) -> None:
    sdk, requests = native_skill_clients
    skill: Final = await sdk.skills.retrieve(
        "skill_1",
        extra_headers={"x-litellm-model": "azure-route"},
        extra_query={"custom_llm_provider": "anthropic"},
    )
    assert skill.id == "skill_1"
    assert requests[-1].url.path == "/openai/v1/skills/skill_1"
    assert requests[-1].headers["authorization"] == "Bearer azure-route"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("header-route", "azure-route", "azure-ad-route"))
@pytest.mark.parametrize(
    "payload",
    (
        {"api_key": "payload-key", "api_base": "https://other-account.test", "custom_llm_provider": "azure"},
        {"api_base": "https://other-account.test/v1"},
        {
            "extra_body": {
                "api_key": "payload-key",
                "api_base": "https://other-account.test",
                "custom_llm_provider": "azure",
            }
        },
        {"extra_headers": {"Authorization": "Bearer payload-key"}},
        {"organization": "payload-org"},
        {"azure_ad_token": "payload-token"},
    ),
)
async def test_native_skill_nested_payload_cannot_replace_deployment_credentials(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], payload: Mapping[str, object], model: str
) -> None:
    sdk, requests = native_skill_clients
    updated: Final = await sdk.skills.update(
        "skill_1",
        default_version="1",
        extra_headers={"x-litellm-model": model},
        extra_body=payload,
    )
    assert updated.id == "skill_1"
    assert requests[-1].url.host == "provider.test"
    assert requests[-1].headers["authorization"] == f"Bearer {model}"
    if model == "header-route":
        assert requests[-1].headers["openai-organization"] == "header-route-org"
    assert requests[-1].headers["x-deployment-route"] == model


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("header-route", "azure-ad-route"))
@pytest.mark.parametrize(
    "payload",
    (
        {"api_key": "payload-key"},
        {"api_base": "https://other-account.test/v1"},
        {"azure_ad_token": "payload-token"},
    ),
)
async def test_native_skill_query_cannot_replace_deployment_credentials(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], payload: Mapping[str, str], model: str
) -> None:
    sdk, requests = native_skill_clients
    retrieved: Final = await sdk.skills.retrieve("skill_1", extra_query={"model": model, **payload})
    assert retrieved.id == "skill_1"
    assert requests[-1].url.host == "provider.test"
    assert requests[-1].headers["authorization"] == f"Bearer {model}"
    assert requests[-1].headers["x-deployment-route"] == model


@pytest.mark.asyncio
@pytest.mark.parametrize("default_source", ("cli", "settings"))
async def test_native_skill_default_model_keeps_deployment_credentials(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]],
    monkeypatch: pytest.MonkeyPatch,
    default_source: str,
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "user_model", "header-route" if default_source == "cli" else None)
    monkeypatch.setattr(
        proxy_server, "general_settings", {"completion_model": "header-route"} if default_source == "settings" else {}
    )
    sdk, requests = native_skill_clients
    updated: Final = await sdk.skills.update(
        "skill_1", default_version="1", extra_body={"api_base": "https://other-account.test/v1"}
    )
    assert updated.id == "skill_1"
    assert requests[-1].url.host == "provider.test"
    assert requests[-1].headers["authorization"] == "Bearer header-route"
    assert requests[-1].headers["x-deployment-route"] == "header-route"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("/v1/skills/skill_1", "/v1/skills/skill_1/content"))
async def test_native_skill_unknown_model_is_rejected_without_provider_request(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], path: str
) -> None:
    sdk, requests = native_skill_clients
    response: Final = await sdk._client.get(f"http://proxy.test{path}?model=unknown-account")
    assert response.status_code == 400
    assert not any(request.url.host == "provider.test" for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("/v1/skills/skill_1/content", "/v1/skills/skill_1/versions"))
async def test_native_skill_unsupported_provider_is_a_client_error(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], path: str
) -> None:
    sdk, requests = native_skill_clients
    response: Final = await sdk._client.get(f"http://proxy.test{path}?custom_llm_provider=anthropic")
    assert response.status_code == 400
    assert "only supported for OpenAI and Azure OpenAI" in response.json()["error"]["message"]
    assert not any(request.url.host == "provider.test" for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", (False, 0, [], {}))
@pytest.mark.parametrize("model", (None, "header-route"))
async def test_native_skill_invalid_provider_does_not_fall_back_to_openai(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]],
    monkeypatch: pytest.MonkeyPatch,
    provider: object,
    model: str | None,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "default-openai")
    monkeypatch.setenv("OPENAI_API_BASE", "https://provider.test/v1")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "api_base", None)
    sdk, requests = native_skill_clients
    response: Final = await sdk._client.post(
        "http://proxy.test/v1/skills/skill_1",
        json={"default_version": "1", "custom_llm_provider": provider, "model": model},
    )
    if model is None:
        assert response.status_code == 400
        assert not any(request.url.host == "provider.test" for request in requests)
        return
    assert response.status_code == 200
    assert requests[-1].headers["authorization"] == "Bearer header-route"


@pytest.mark.asyncio
async def test_native_skill_unsupported_content_type_is_a_client_error(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]],
) -> None:
    sdk, requests = native_skill_clients
    response: Final = await sdk._client.post(
        "http://proxy.test/v1/skills/skill_1?model=header-route",
        content="default_version=1",
        headers={"content-type": "text/plain"},
    )
    assert response.status_code == 400
    assert not any(request.url.host == "provider.test" for request in requests)


@pytest.mark.asyncio
async def test_native_skill_sdk_pagination_keeps_the_routing_header(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], respx_mock: respx.MockRouter
) -> None:
    sdk, requests = native_skill_clients
    forwarded: Final[list[httpx.Request]] = []

    def page(request: httpx.Request) -> httpx.Response:
        forwarded.append(request)
        second_page: Final = request.url.params.get("after") == "skill_1"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "has_more": not second_page,
                "data": [
                    {
                        "id": "skill_2" if second_page else "skill_1",
                        "object": "skill",
                        "created_at": 1,
                        "name": "test",
                        "description": "test",
                        "default_version": "1",
                        "latest_version": "1",
                    }
                ],
            },
        )

    respx_mock.route(host="provider.test").mock(side_effect=page)
    ids: Final = tuple(
        [skill.id async for skill in sdk.skills.list(limit=1, extra_headers={"x-litellm-model": "header-route"})]
    )
    assert ids == ("skill_1", "skill_2")
    assert all(request.headers["authorization"] == "Bearer header-route" for request in forwarded)
    assert forwarded[-1].url.params["after"] == "skill_1"


@pytest_asyncio.fixture
async def authenticated_native_skills(
    native_skill_clients: tuple[AsyncOpenAI, list[httpx.Request]], monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[AsyncOpenAI, list[httpx.Request], UserAPIKeyAuth]]:
    from litellm.proxy import proxy_server

    sdk, requests = native_skill_clients
    cache: Final = UserApiKeyCache()
    token: Final = UserAPIKeyAuth(
        token=proxy_server.hash_token("sk-native-skills-test"),
        user_role=LitellmUserRoles.INTERNAL_USER,
        models=["header-route"],
        allowed_routes=["openai_routes"],
    )
    cache.set_cache(key=token.token, value=token)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "master_key", "sk-unrelated-master-key")
    monkeypatch.setattr(
        proxy_server,
        "prisma_client",
        PrismaClient(
            database_url="postgresql://test:test@localhost:5432/test", proxy_logging_obj=proxy_server.proxy_logging_obj
        ),
    )
    sdk._client._transport.app.dependency_overrides.pop(user_api_key_auth)
    yield sdk, requests, token


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("POST", "/v1/skills/skill_1"),
        ("GET", "/v1/skills/skill_1/content"),
        ("POST", "/v1/skills/skill_1/versions"),
        ("GET", "/v1/skills/skill_1/versions"),
        ("GET", "/v1/skills/skill_1/versions/1"),
        ("DELETE", "/v1/skills/skill_1/versions/1"),
        ("GET", "/v1/skills/skill_1/versions/1/content"),
    ),
)
async def test_native_only_skill_routes_default_to_openai_with_existing_auth(
    authenticated_native_skills: tuple[AsyncOpenAI, list[httpx.Request], UserAPIKeyAuth],
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "default-openai")
    monkeypatch.setenv("OPENAI_API_BASE", "https://provider.test/v1")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "api_base", None)
    sdk, requests, token = authenticated_native_skills
    response: Final = await sdk._client.request(
        method,
        f"http://proxy.test{path}",
        headers={"authorization": "Bearer sk-native-skills-test"},
        json={"default_version": "1"},
        files={"files": ("skill.zip", b"PK archive", "application/zip")}
        if method == "POST" and path.endswith("/versions")
        else None,
    )
    assert response.status_code == 200
    assert requests[-1].url.path.startswith("/v1/skills")
    assert requests[-1].headers["authorization"] == "Bearer default-openai"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    (
        "/v1/skills",
        "/v1/skills/skill_1",
        "/v1/skills/skill_1/content",
        "/v1/skills/skill_1/versions",
        "/v1/skills/skill_1/versions/1",
        "/v1/skills/skill_1/versions/1/content",
    ),
)
async def test_native_skill_real_auth_accepts_allowed_key_and_rejects_wrong_model(
    authenticated_native_skills: tuple[AsyncOpenAI, list[httpx.Request], UserAPIKeyAuth], path: str
) -> None:
    sdk, requests, token = authenticated_native_skills
    accepted: Final = await sdk._client.get(
        f"http://proxy.test{path}?model=header-route", headers={"authorization": "Bearer sk-native-skills-test"}
    )
    assert accepted.status_code == 200
    forwarded_count: Final = len(tuple(request for request in requests if request.url.host == "provider.test"))
    rejected: Final = await sdk._client.get(
        f"http://proxy.test{path}?model=query-route", headers={"authorization": "Bearer sk-native-skills-test"}
    )
    assert rejected.status_code == 403
    assert len(tuple(request for request in requests if request.url.host == "provider.test")) == forwarded_count


@pytest.mark.asyncio
async def test_native_skill_real_auth_checks_model_and_preserves_all_upload_parts(
    authenticated_native_skills: tuple[AsyncOpenAI, list[httpx.Request], UserAPIKeyAuth],
) -> None:
    sdk, requests, token = authenticated_native_skills
    skill: Final = await sdk.skills.create(
        files=[("test-skill/SKILL.md", b"manifest"), ("test-skill/tool.py", b"script")],
        extra_headers={"x-litellm-model": "header-route"},
    )
    assert skill.id == "skill_1"
    assert requests[-1].headers["authorization"] == "Bearer header-route"
    assert b"manifest" in requests[-1].content and b"script" in requests[-1].content


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("/v1/skills/skill_1/content", "/v1/skills/skill_1/versions"))
async def test_native_skill_real_auth_rejects_disallowed_routes_without_forwarding(
    authenticated_native_skills: tuple[AsyncOpenAI, list[httpx.Request], UserAPIKeyAuth], path: str
) -> None:
    from litellm.proxy import proxy_server

    sdk, requests, token = authenticated_native_skills
    token.allowed_routes = ["/v1/chat/completions"]
    proxy_server.user_api_key_cache.set_cache(key=token.token, value=token)
    response: Final = await sdk._client.get(
        f"http://proxy.test{path}?model=header-route", headers={"authorization": "Bearer sk-native-skills-test"}
    )
    assert response.status_code == 403
    assert not any(request.url.host == "provider.test" for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("/v1/skills/skill_1/content", "/v1/skills/skill_1/versions"))
async def test_native_skill_real_auth_enforces_key_budget_before_provider_http(
    authenticated_native_skills: tuple[AsyncOpenAI, list[httpx.Request], UserAPIKeyAuth], path: str
) -> None:
    from litellm.proxy import proxy_server

    sdk, requests, token = authenticated_native_skills
    token.max_budget = 1
    token.spend = 2
    proxy_server.user_api_key_cache.set_cache(key=token.token, value=token)
    response: Final = await sdk._client.get(
        f"http://proxy.test{path}?model=header-route", headers={"authorization": "Bearer sk-native-skills-test"}
    )
    assert 400 <= response.status_code < 500
    assert "budget" in response.json()["error"]["message"].lower()
    assert not any(request.url.host == "provider.test" for request in requests)


@pytest.mark.parametrize(
    "path",
    (
        "/v1/skills",
        "/v1/skills/skill_1",
        "/v1/skills/skill_1/content",
        "/v1/skills/skill_1/versions",
        "/v1/skills/skill_1/versions/1",
        "/v1/skills/skill_1/versions/1/content",
    ),
)
def test_native_skill_routes_use_existing_virtual_key_permissions_and_budget_checks(path: str) -> None:
    allowed: Final = UserAPIKeyAuth(allowed_routes=["openai_routes"])
    denied: Final = UserAPIKeyAuth(allowed_routes=["/v1/chat/completions"])
    assert RouteChecks.is_virtual_key_allowed_to_call_route(route=path, valid_token=allowed)
    assert not route_skips_budget_checks(path)
    with pytest.raises(HTTPException):
        RouteChecks.is_virtual_key_allowed_to_call_route(route=path, valid_token=denied)
