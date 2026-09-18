"""Behavior pins for ``proxy_server.py`` model-info routes.

Pins (PR2):
    - GET /v2/model/info
    - GET /v1/model/info
    - GET /model/info
    - GET /model_group/info
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy import proxy_server
from litellm.utils import _invalidate_model_cost_lowercase_map

from .conftest import normalize  # type: ignore[import-not-found]


@pytest.mark.parametrize(
    ("backend_model", "base_model"),
    (
        ("azure/hosted-model", "fallback-model"),
        ("openai/org/fallback-model", None),
        ("openai/hosted-model", "fallback-model"),
        ("openai/fallback-model", "unknown-base-model"),
    ),
)
@pytest.mark.parametrize("advertised_limit", (None, 2048))
async def test_discovery_preserves_model_info_fallbacks(
    backend_model: str, base_model: str | None, advertised_limit: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "local",
                "litellm_params": {
                    "model": backend_model,
                    "api_base": "https://fallback.test/v1",
                    "api_key": "local-key",
                },
                "model_info": {"id": "fallback-deployment", "base_model": base_model, "max_output_tokens": 333},
            }
        ]
    )
    builtin: Final = {
        "litellm_provider": "openai",
        "mode": "chat",
        "max_input_tokens": 7000,
        "max_output_tokens": 2000,
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
    }
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "fallback-model": builtin,
            "openai/fallback-model": builtin,
            "fallback-deployment": {"litellm_provider": "openai", "mode": "chat"},
        },
    )
    _invalidate_model_cost_lowercase_map()
    monkeypatch.setattr(proxy_server, "llm_router", router)
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": backend_model.split("/", 1)[1],
                            "max_model_len": advertised_limit,
                        }
                    ]
                },
            )
        )
    ) as client:
        handler.client = client
        await router.arefresh_model_info(client=handler)
    deployment: Final = {
        **router.model_list[0],
        "model_info": {**router.model_list[0]["model_info"], "mode": None},
    }
    enriched_models: Final = (
        proxy_server._get_proxy_model_info(copy.deepcopy(deployment)),
        proxy_server._enrich_model_info_with_litellm_data(copy.deepcopy(deployment), llm_router=router),
    )
    expected_input: Final = (
        advertised_limit
        if advertised_limit is not None and backend_model.startswith("openai/")
        else builtin["max_input_tokens"]
    )
    for enriched in enriched_models:
        info: Final = enriched["model_info"]
        assert info.get("max_input_tokens") == expected_input
        assert info["max_output_tokens"] == 333
        assert info["input_cost_per_token"] == builtin["input_cost_per_token"]
        assert info["output_cost_per_token"] == builtin["output_cost_per_token"]
        assert info["mode"] is None
    _invalidate_model_cost_lowercase_map()


async def test_upstream_limits_reach_model_info_routes(
    client: TestClient,
    auth_as: Callable[[], AbstractContextManager[object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "local",
                "litellm_params": {
                    "model": "hosted_vllm/org/local-model",
                    "api_base": "https://backend.test/v1",
                    "api_key": "local-key",
                },
                "model_info": {"id": "local-deployment", "max_output_tokens": 512, "max_input_tokens": None},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", router.get_model_list())
    monkeypatch.setattr(proxy_server, "user_model", None)

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "org/local-model", "max_model_len": 4096}]})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as upstream:
        handler.client = upstream
        litellm.in_memory_llm_clients_cache.set_cache("async_httpx_clientopenai", handler)
        await proxy_server.ProxyStartupEvent.refresh_model_info()
        with auth_as():
            for path in ("/v1/model/info", "/model/info"):
                response: Final = client.get(path)
                assert response.status_code == 200, response.text
                info: Final = response.json()["data"][0]["model_info"]
                assert (info["max_input_tokens"], info["max_output_tokens"]) == (4096, 512)
            group_response: Final = client.get("/model_group/info")
            assert group_response.status_code == 200, group_response.text
            assert group_response.json()["data"][0]["max_input_tokens"] == 4096
    _invalidate_model_cost_lowercase_map()


# ---------------------------------------------------------------------------
# GET /v2/model/info
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_router(monkeypatch):
    router = MagicMock()
    router.model_list = []
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", [])
    yield router


@pytest.fixture
def null_router(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "llm_model_list", None)
    yield


def test_v2_model_info_empty_router_happy_path(client, auth_as, empty_router):
    """Pins ``GET /v2/model/info`` (empty router branch returns deterministic shape)."""
    with auth_as():
        response = client.get("/v2/model/info")
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "data": [],
        "total_count": 0,
        "current_page": 1,
        "total_pages": 0,
        "size": 50,
    }


def test_v2_model_info_invalid_page_returns_422(client, auth_as, empty_router):
    """Pins ``GET /v2/model/info`` (error: invalid page parameter)."""
    with auth_as():
        response = client.get("/v2/model/info", params={"page": 0})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_v2_model_info_in_openapi_schema():
    """``GET /v2/model/info`` is published in the proxy OpenAPI/Swagger spec."""
    from litellm.proxy.proxy_server import get_openapi_schema

    schema = get_openapi_schema()
    assert "/v2/model/info" in schema["paths"]
    assert "get" in schema["paths"]["/v2/model/info"]


# ---------------------------------------------------------------------------
# GET /v1/model/info, GET /model/info
# ---------------------------------------------------------------------------


@pytest.fixture
def configured_router(monkeypatch):
    deployment = MagicMock()
    deployment.model_dump = MagicMock(
        return_value={
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "abc", "db_model": False},
        }
    )
    router = MagicMock()
    router.get_deployment = MagicMock(return_value=deployment)
    router.get_model_names = MagicMock(return_value=["gpt-4"])
    router.get_model_access_groups = MagicMock(return_value={})
    router.get_model_list = MagicMock(return_value=[])
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", [{"model_name": "gpt-4"}])
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server, "_get_proxy_model_info", lambda model: model)
    yield router


@pytest.mark.parametrize("path", ["/v1/model/info", "/model/info"])
def test_v1_model_info_specific_id_happy(client, auth_as, configured_router, path):
    """Pins ``GET /v1/model/info`` and ``GET /model/info`` (happy: specific id).

    Includes ``litellm_model_id`` so the early-return branch produces a
    deterministic ``{"data": [<one deployment>]}`` body without touching
    the full model-info enrichment pipeline.
    """
    with auth_as():
        response = client.get(path, params={"litellm_model_id": "abc"})
    assert response.status_code == 200
    body = normalize(response.json())
    assert body == {
        "data": [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"id": "<VOLATILE>", "db_model": False},
            }
        ]
    }


@pytest.mark.parametrize("path", ["/v1/model/info", "/model/info"])
def test_v1_model_info_no_model_list_error(client, auth_as, null_router, path):
    """Pins ``GET /v1/model/info`` and ``GET /model/info`` (error: no model list)."""
    with auth_as():
        response = client.get(path)
    assert response.status_code == 500
    assert "LLM Model List not loaded" in response.text


def test_get_proxy_model_info_surfaces_supports_parallel_function_calling(local_model_cost_map):
    """``GET /v1/model/info`` enriches each deployment through ``_get_proxy_model_info``; a registry
    entry declaring parallel function calling must land in ``model_info`` instead of null."""
    enriched = proxy_server._get_proxy_model_info(
        model={
            "model_name": "glm-5.3-flash",
            "litellm_params": {"model": "together_ai/zai-org/GLM-5.3-Flash"},
            "model_info": {"id": "glm-deployment", "db_model": False},
        }
    )
    assert enriched["model_info"]["supports_parallel_function_calling"] is True


def _enriched_model_info(monkeypatch, litellm_params: dict, model_info: dict) -> dict:
    monkeypatch.setattr(proxy_server, "llm_router", None)
    enriched: Final = proxy_server._get_proxy_model_info(
        model={"model_name": "gpt-5.6", "litellm_params": litellm_params, "model_info": model_info}
    )
    return enriched["model_info"]


def test_get_proxy_model_info_reports_no_pricing_overrides_for_a_cost_map_priced_deployment(
    monkeypatch, local_model_cost_map
):
    """LIT-8064. A deployment with no price of its own follows the cost map, and ``/model/info``
    says so with an empty ``pricing_overrides``."""
    info = _enriched_model_info(monkeypatch, {"model": "openai/gpt-5.6"}, {"id": "dep-synced", "db_model": True})
    assert info["pricing_overrides"] == ()
    assert info["input_cost_per_token"] == litellm.model_cost["gpt-5.6"]["input_cost_per_token"]


def test_get_proxy_model_info_shows_litellm_params_pricing_and_names_it_as_an_override(
    monkeypatch, local_model_cost_map
):
    """A price on ``litellm_params`` is what the deployment bills at, so the model page shows that
    value rather than the cost map's and lists the field under ``pricing_overrides``."""
    info = _enriched_model_info(
        monkeypatch,
        {"model": "openai/gpt-5.6", "input_cost_per_token_batches": 1e-09},
        {"id": "dep-batches", "db_model": True},
    )
    assert info["pricing_overrides"] == ("input_cost_per_token_batches",)
    assert info["input_cost_per_token_batches"] == 1e-09
    assert info["input_cost_per_token"] == litellm.model_cost["gpt-5.6"]["input_cost_per_token"]


def test_get_proxy_model_info_names_config_model_info_pricing_as_an_override(monkeypatch, local_model_cost_map):
    """Pricing declared under ``model_info`` in config.yaml overrides the cost map too."""
    info = _enriched_model_info(
        monkeypatch, {"model": "openai/gpt-5.6"}, {"id": "dep-config", "db_model": False, "output_cost_per_token": 7e-06}
    )
    assert info["pricing_overrides"] == ("output_cost_per_token",)
    assert info["output_cost_per_token"] == 7e-06


def test_v2_model_info_reports_pricing_overrides_to_the_admin_ui(client, auth_as, monkeypatch, local_model_cost_map):
    """LIT-8064. The Admin UI model page reads ``GET /v2/model/info``, so the override report
    has to ride that route too, not only ``/model/info``."""
    model_list: Final = [
        {
            "model_name": "gpt-5.6",
            "litellm_params": {"model": "openai/gpt-5.6", "input_cost_per_token": 3e-06},
            "model_info": {"id": "dep-typed", "db_model": True},
        },
        {
            "model_name": "gpt-5.6",
            "litellm_params": {"model": "openai/gpt-5.6"},
            "model_info": {"id": "dep-synced", "db_model": True},
        },
    ]
    router: Final = MagicMock()
    router.model_list = model_list
    router.get_discovered_model_info = MagicMock(return_value={})
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", model_list)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server.proxy_config, "get_config", AsyncMock(return_value={}))
    monkeypatch.setattr(
        proxy_server,
        "_apply_search_filter_to_models",
        AsyncMock(side_effect=lambda all_models, **kw: (all_models, len(all_models))),
    )
    import litellm.proxy.agent_endpoints.model_list_helpers as mlh

    monkeypatch.setattr(mlh, "append_agents_to_model_info", AsyncMock(side_effect=lambda models, **kw: models))

    with auth_as():
        response = client.get("/v2/model/info")

    assert response.status_code == 200, response.text
    by_id: Final = {m["model_info"]["id"]: m["model_info"] for m in response.json()["data"]}
    assert by_id["dep-typed"]["pricing_overrides"] == ["input_cost_per_token"]
    assert by_id["dep-typed"]["input_cost_per_token"] == 3e-06
    assert by_id["dep-synced"]["pricing_overrides"] == []
    assert by_id["dep-synced"]["input_cost_per_token"] == litellm.model_cost["gpt-5.6"]["input_cost_per_token"]


def test_v1_model_info_star_wildcard_filter_keeps_provider_expansion(monkeypatch):
    from litellm.proxy._types import SpecialModelNames, UserAPIKeyAuth
    from litellm.proxy.auth import model_checks

    def fake_get_provider_models(provider, litellm_params=None):
        if provider == "openai":
            return ["gpt-4o"]
        return []

    deployment = {
        "model_name": "*",
        "litellm_params": {"model": "openai/*"},
    }
    router = MagicMock()
    router.get_model_access_groups = MagicMock(return_value={})
    router.get_model_names = MagicMock(return_value=["*"])
    router.get_model_list = MagicMock(return_value=[deployment])
    monkeypatch.setattr(model_checks, "get_provider_models", fake_get_provider_models)

    expanded_deployments = proxy_server.expand_wildcard_deployments_for_model_info([deployment])
    allowed_model_names = proxy_server._get_v1_model_info_allowed_model_names(
        user_api_key_dict=UserAPIKeyAuth(
            api_key="sk-test",
            models=[SpecialModelNames.all_proxy_models.value],
        ),
        llm_router=router,
    )

    result = proxy_server._filter_v1_model_info_deployments(
        all_models=expanded_deployments,
        allowed_model_names=allowed_model_names,
    )

    assert [model["model_name"] for model in result] == ["openai/gpt-4o"]


# ---------------------------------------------------------------------------
# GET /model/info — team BYOK scoping (issue #30983)
# ---------------------------------------------------------------------------

_BYOK_TEAM_ID = "team-abc"
_BYOK_PUBLIC_NAME = "my-byok-gpt-4"
_BYOK_INTERNAL_NAME = f"model_name_{_BYOK_TEAM_ID}_0123456789abcdef"


@pytest.fixture
def byok_team_router(monkeypatch):
    """Router holding one team-scoped BYOK deployment for team `team-abc`.

    Mirrors how a team's own-key BYOK model lives in the router: the routing
    key is an internal mangled name while the public name lives in
    `model_info.team_public_model_name`.
    """
    byok_deployment = {
        "model_name": _BYOK_INTERNAL_NAME,
        "litellm_params": {"model": "openai/gpt-4"},
        "model_info": {
            "id": "byok-deployment-id",
            "db_model": True,
            "team_id": _BYOK_TEAM_ID,
            "team_public_model_name": _BYOK_PUBLIC_NAME,
        },
    }

    router = MagicMock()
    router.model_list = [byok_deployment]
    router.get_model_list_from_model_alias = MagicMock(return_value=[])
    router.get_model_names = MagicMock(return_value=[])
    router.get_model_access_groups = MagicMock(return_value={})
    router.get_model_ids = MagicMock(return_value=[])

    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", [byok_deployment])
    monkeypatch.setattr(proxy_server, "user_model", None)
    yield router


@pytest.mark.parametrize("path", ["/v1/model/info", "/model/info"])
def test_model_info_team_key_sees_own_byok_model(client, auth_as, byok_team_router, mock_prisma, monkeypatch, path):
    """Regression for #30983: a team key (user_id=None) must see its own
    team's BYOK model under the public name.

    Before the fix `_get_caller_byok_team_scope` keyed only off the bound
    user's team memberships, returned an empty set for a team key, and the
    BYOK row was dropped -> `{"data": []}`.
    """
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(proxy_server, "prisma_client", mock_prisma)
    mock_prisma.db.litellm_usertable.find_unique.return_value = None

    with auth_as(
        role=LitellmUserRoles.INTERNAL_USER,
        user_id=None,
        team_id=_BYOK_TEAM_ID,
        team_models=[_BYOK_PUBLIC_NAME],
    ):
        response = client.get(path)

    assert response.status_code == 200
    data = response.json()["data"]
    surfaced_names = [m.get("model_name") for m in data]
    assert _BYOK_PUBLIC_NAME in surfaced_names
    assert _BYOK_INTERNAL_NAME not in surfaced_names


@pytest.mark.parametrize("path", ["/v1/model/info", "/model/info"])
def test_model_info_team_key_cannot_see_other_teams_byok_model(
    client, auth_as, byok_team_router, mock_prisma, monkeypatch, path
):
    """A team key for a different team must NOT see team-abc's BYOK row.

    Guards the fix from over-broadening into a cross-team metadata leak.
    """
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr(proxy_server, "prisma_client", mock_prisma)
    mock_prisma.db.litellm_usertable.find_unique.return_value = None

    with auth_as(
        role=LitellmUserRoles.INTERNAL_USER,
        user_id=None,
        team_id="other-team",
        team_models=[_BYOK_PUBLIC_NAME],
    ):
        response = client.get(path)

    assert response.status_code == 200
    data = response.json()["data"]
    surfaced_names = [m.get("model_name") for m in data]
    assert _BYOK_PUBLIC_NAME not in surfaced_names
    assert _BYOK_INTERNAL_NAME not in surfaced_names


# ---------------------------------------------------------------------------
# GET /model_group/info
# ---------------------------------------------------------------------------


def test_model_group_info_no_models_happy(client, auth_as, null_router):
    """Pins ``GET /model_group/info`` (happy: empty list when no models)."""
    with auth_as():
        response = client.get("/model_group/info")
    assert response.status_code == 200
    summary = {
        "status_code": response.status_code,
        "body": normalize(response.json()),
        "object_kind": "model_group_info",
    }
    assert summary == {
        "status_code": 200,
        "body": {"data": []},
        "object_kind": "model_group_info",
    }


def test_model_group_info_invalid_method(client, auth_as, null_router):
    """Pins ``GET /model_group/info`` (error: method not allowed)."""
    with auth_as():
        response = client.post("/model_group/info", json={})
    assert response.status_code == 405
    assert len(response.content) > 0


@pytest.fixture
def model_group_info_router(monkeypatch):
    from litellm.types.proxy.management_endpoints.model_management_endpoints import ModelGroupInfoProxy

    model_names = ["gpt-4", "claude-3"]
    router = MagicMock()
    router.get_model_names.return_value = model_names
    router.get_model_access_groups.return_value = {}
    router.get_model_list.return_value = []

    def model_group_info(*, llm_router, all_models_str, model_group):
        return [ModelGroupInfoProxy(model_group=name, providers=[]) for name in all_models_str]

    async def append_agents_to_model_group(*, model_groups, user_api_key_dict):
        return model_groups

    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", [{"model_name": name} for name in model_names])
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", None)
    monkeypatch.setattr(proxy_server, "_get_model_group_info", model_group_info)

    from litellm.proxy.agent_endpoints import model_list_helpers

    monkeypatch.setattr(
        model_list_helpers,
        "append_agents_to_model_group",
        AsyncMock(side_effect=append_agents_to_model_group),
    )
    return router


@pytest.mark.parametrize("admin_role", ["proxy_admin", "proxy_admin_viewer"])
def test_model_group_info_proxy_admin_ignores_key_model_restriction(
    client, auth_as, model_group_info_router, admin_role
):
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles(admin_role), models=["no-default-models"]):
        response = client.get("/model_group/info")

    assert response.status_code == 200
    assert [model["model_group"] for model in response.json()["data"]] == ["gpt-4", "claude-3"]


@pytest.mark.parametrize("admin_role", ["proxy_admin", "proxy_admin_viewer"])
def test_model_group_info_proxy_admin_expands_wildcard_deployments(client, auth_as, model_group_info_router, admin_role):
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    model_group_info_router.get_model_names.return_value = ["gpt-4", "anthropic/*"]
    known_anthropic_models = get_known_models_from_wildcard(wildcard_model="anthropic/*")
    assert known_anthropic_models

    with auth_as(LitellmUserRoles(admin_role), models=["no-default-models"]):
        response = client.get("/model_group/info")

    assert response.status_code == 200
    assert [model["model_group"] for model in response.json()["data"]] == ["gpt-4", *known_anthropic_models]


def test_model_group_info_internal_user_key_model_restriction_applies(client, auth_as, model_group_info_router):
    from litellm.proxy._types import LitellmUserRoles

    with auth_as(LitellmUserRoles.INTERNAL_USER, models=["gpt-4"]):
        response = client.get("/model_group/info")

    assert response.status_code == 200
    assert [model["model_group"] for model in response.json()["data"]] == ["gpt-4"]


# ---------------------------------------------------------------------------
# GET /v2/model/info?exclude_auto_routers
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_auto_router_router(monkeypatch):
    """Router carrying one ordinary deployment per auto-router strategy plus two plain ones."""
    model_list = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
            "model_info": {"id": "plain-1", "db_model": False},
        },
        {
            "model_name": "tri-tier-router",
            "litellm_params": {"model": "auto_router/complexity_router"},
            "model_info": {"id": "auto-complexity", "db_model": True},
        },
        {
            "model_name": "support-router",
            "litellm_params": {"model": "auto_router/support-router"},
            "model_info": {"id": "auto-semantic", "db_model": True},
        },
        {
            "model_name": "adaptive-router",
            "litellm_params": {"model": "auto_router/adaptive_router"},
            "model_info": {"id": "auto-adaptive", "db_model": True},
        },
        {
            "model_name": "claude-opus",
            "litellm_params": {"model": "anthropic/claude-opus-4-6"},
            "model_info": {"id": "plain-2", "db_model": False},
        },
    ]
    from unittest.mock import AsyncMock

    router = MagicMock()
    router.model_list = model_list
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", model_list)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server.proxy_config, "get_config", AsyncMock(return_value={}))
    monkeypatch.setattr(
        proxy_server,
        "_apply_search_filter_to_models",
        AsyncMock(side_effect=lambda all_models, **kw: (all_models, len(all_models))),
    )
    monkeypatch.setattr(proxy_server, "_enrich_model_info_with_litellm_data", lambda model, **kw: model)

    import litellm.proxy.agent_endpoints.model_list_helpers as mlh

    monkeypatch.setattr(mlh, "append_agents_to_model_info", AsyncMock(side_effect=lambda models, **kw: models))
    yield router


def _model_names(payload) -> list:
    return [m["model_name"] for m in payload["data"]]


def test_v2_model_info_includes_auto_routers_by_default(client, auth_as, mixed_auto_router_router):
    """The new param is opt-in; omitting it must not change what any existing caller sees."""
    with auth_as():
        response = client.get("/v2/model/info")
    assert response.status_code == 200
    payload = response.json()
    assert "tri-tier-router" in _model_names(payload)
    assert payload["total_count"] == 5


def test_v2_model_info_excludes_every_auto_router_strategy(client, auth_as, mixed_auto_router_router):
    """All four `auto_router/*` strategies go, not just the semantic one that
    Router._is_auto_router_deployment recognises."""
    with auth_as():
        response = client.get("/v2/model/info", params={"exclude_auto_routers": "true"})
    assert response.status_code == 200
    payload = response.json()
    assert _model_names(payload) == ["gpt-4o-mini", "claude-opus"]


def test_v2_model_info_exclude_auto_routers_shrinks_total_count(client, auth_as, mixed_auto_router_router):
    """The filter must run before the count, or the table pages off a total that
    includes rows it never renders (49 shown, 50 claimed)."""
    with auth_as():
        response = client.get("/v2/model/info", params={"exclude_auto_routers": "true"})
    payload = response.json()
    assert payload["total_count"] == 2
    assert len(payload["data"]) == payload["total_count"]


def test_v2_model_info_exclude_auto_routers_paginates_over_the_filtered_set(client, auth_as, mixed_auto_router_router):
    """Page size applies to the filtered list, so no page silently comes back short."""
    with auth_as():
        response = client.get("/v2/model/info", params={"exclude_auto_routers": "true", "page": 1, "size": 1})
    payload = response.json()
    assert payload["total_count"] == 2
    assert payload["total_pages"] == 2
    assert len(payload["data"]) == 1


@pytest.mark.asyncio
async def test_model_info_v2_query_sentinel_does_not_filter(monkeypatch, mixed_auto_router_router):
    """Called directly (not through FastAPI) the default arrives as a truthy Query object.
    Guarding on `is True` is what stops every direct-call test from silently filtering."""
    from unittest.mock import AsyncMock

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server.proxy_config, "get_config", AsyncMock(return_value={}))
    monkeypatch.setattr(
        proxy_server,
        "_apply_search_filter_to_models",
        AsyncMock(side_effect=lambda all_models, **kw: (all_models, len(all_models))),
    )
    monkeypatch.setattr(proxy_server, "_enrich_model_info_with_litellm_data", lambda model, **kw: model)

    import litellm.proxy.agent_endpoints.model_list_helpers as mlh

    monkeypatch.setattr(mlh, "append_agents_to_model_info", AsyncMock(side_effect=lambda models, **kw: models))

    admin = UserAPIKeyAuth(user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN)
    # Deliberately omit exclude_auto_routers, exactly as the pre-existing direct-call tests do.
    resp = await proxy_server.model_info_v2(
        user_api_key_dict=admin,
        model=None,
        user_models_only=False,
        include_team_models=False,
        debug=False,
        page=1,
        size=50,
        search=None,
        modelId=None,
        teamId=None,
        sortBy=None,
        sortOrder="asc",
    )

    assert "tri-tier-router" in [m["model_name"] for m in resp["data"]]


# ---------------------------------------------------------------------------
# GET /v2/model/info?access_group / ?wildcard_only
# ---------------------------------------------------------------------------


@pytest.fixture
def access_group_router(monkeypatch):
    """Router with one sales-team deployment, one wildcard sales-team deployment and one ungrouped one."""
    model_list = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
            "model_info": {"id": "sales-1", "db_model": False, "access_groups": ["sales-team"]},
        },
        {
            "model_name": "openai/*",
            "litellm_params": {"model": "openai/*"},
            "model_info": {"id": "sales-wildcard", "db_model": False, "access_groups": ["sales-team", "eng"]},
        },
        {
            "model_name": "claude-opus",
            "litellm_params": {"model": "anthropic/claude-opus-4-6"},
            "model_info": {"id": "plain-1", "db_model": False},
        },
    ]
    from unittest.mock import AsyncMock

    router = MagicMock()
    router.model_list = model_list
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", model_list)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server.proxy_config, "get_config", AsyncMock(return_value={}))
    monkeypatch.setattr(
        proxy_server,
        "_apply_search_filter_to_models",
        AsyncMock(side_effect=lambda all_models, **kw: (all_models, len(all_models))),
    )
    monkeypatch.setattr(proxy_server, "_enrich_model_info_with_litellm_data", lambda model, **kw: model)

    import litellm.proxy.agent_endpoints.model_list_helpers as mlh

    monkeypatch.setattr(mlh, "append_agents_to_model_info", AsyncMock(side_effect=lambda models, **kw: models))
    yield router


def test_v2_model_info_without_new_filters_returns_everything(client, auth_as, access_group_router):
    with auth_as():
        response = client.get("/v2/model/info")
    payload = response.json()
    assert payload["total_count"] == 3
    assert len(payload["data"]) == 3


def test_v2_model_info_access_group_filters_rows_and_total(client, auth_as, access_group_router):
    """The table pages off total_count, so the filter must shrink the total, not only the page."""
    with auth_as():
        response = client.get("/v2/model/info", params={"access_group": "sales-team"})
    payload = response.json()
    assert _model_names(payload) == ["gpt-4o-mini", "openai/*"]
    assert payload["total_count"] == 2


def test_v2_model_info_unknown_access_group_is_empty(client, auth_as, access_group_router):
    with auth_as():
        response = client.get("/v2/model/info", params={"access_group": "nobody"})
    payload = response.json()
    assert payload["data"] == []
    assert payload["total_count"] == 0


def test_v2_model_info_wildcard_only_filters_rows_and_total(client, auth_as, access_group_router):
    with auth_as():
        response = client.get("/v2/model/info", params={"wildcard_only": "true"})
    payload = response.json()
    assert _model_names(payload) == ["openai/*"]
    assert payload["total_count"] == 1


def test_v2_model_info_access_group_paginates_over_the_filtered_set(client, auth_as, access_group_router):
    with auth_as():
        response = client.get("/v2/model/info", params={"access_group": "sales-team", "page": 2, "size": 1})
    payload = response.json()
    assert _model_names(payload) == ["openai/*"]
    assert payload["total_count"] == 2
    assert payload["total_pages"] == 2
