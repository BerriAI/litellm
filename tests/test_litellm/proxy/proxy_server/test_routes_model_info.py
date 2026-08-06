"""Behavior pins for ``proxy_server.py`` model-info routes.

Pins (PR2):
    - GET /v2/model/info
    - GET /v1/model/info
    - GET /model/info
    - GET /model_group/info
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from litellm.proxy import proxy_server

from .conftest import normalize  # type: ignore[import-not-found]

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

    expanded_deployments = proxy_server.expand_wildcard_deployments_for_model_info(
        [deployment]
    )
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


def test_v2_model_info_exclude_auto_routers_paginates_over_the_filtered_set(
    client, auth_as, mixed_auto_router_router
):
    """Page size applies to the filtered list, so no page silently comes back short."""
    with auth_as():
        response = client.get(
            "/v2/model/info", params={"exclude_auto_routers": "true", "page": 1, "size": 1}
        )
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
