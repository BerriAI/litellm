"""
Route-level tests for /v2/model/info — verify that
LiteLLM_UserTable.models ("Personal Models") narrows the returned
deployment list on every flag combination, including the default
(flagless) branch which previously leaked the full router.model_list.

Same setup pattern as test_v1_model_info_user_filter.py: real FastAPI
route → real user_api_key_auth override → real model_info_v2 handler →
real apply_user_models_filter_to_deployments → mocked get_user_object.

These tests do NOT depend on team membership; the include_team_models
branch is exercised in case 17 (e2e) but here we focus on the
defense-in-depth final-step filter that runs for every request shape.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm.proxy.proxy_server as ps
from litellm.caching.caching import DualCache
from litellm.proxy._types import (
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.proxy_server import app

_PROXY_MODELS = [
    {"model_name": "gpt-4", "litellm_params": {"model": "openai/gpt-4"}},
    {
        "model_name": "claude-3-opus",
        "litellm_params": {"model": "anthropic/claude-3-opus"},
    },
    {
        "model_name": "claude-3-haiku",
        "litellm_params": {"model": "anthropic/claude-3-haiku"},
    },
    {
        "model_name": "anthropic/claude-3-5-sonnet",
        "litellm_params": {"model": "anthropic/claude-3-5-sonnet"},
    },
    {
        "model_name": "anthropic/claude-3-7-sonnet",
        "litellm_params": {"model": "anthropic/claude-3-7-sonnet"},
    },
]


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def configure_router(monkeypatch):
    import litellm

    router = litellm.Router(
        model_list=_PROXY_MODELS,
        model_group_alias={},
    )
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "llm_model_list", _PROXY_MODELS)
    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(ps, "prisma_client", MagicMock())
    monkeypatch.setattr(ps, "user_api_key_cache", DualCache())
    return router


def _override_auth(user_id):
    def _auth():
        return UserAPIKeyAuth(
            api_key="test-key",
            user_id=user_id,
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=[],
            team_id=None,
            team_models=[],
        )

    app.dependency_overrides[ps.user_api_key_auth] = _auth


@pytest.fixture(autouse=True)
def _clear_auth_override():
    yield
    app.dependency_overrides.pop(ps.user_api_key_auth, None)


def _patch_user(monkeypatch, models, teams=None):
    async def _fake(*args, **kwargs):
        return LiteLLM_UserTable(
            user_id="u-test",
            max_budget=None,
            user_email=None,
            models=models,
            teams=teams or [],
        )

    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.get_user_object",
        _fake,
    )


def _model_names(resp_json):
    return sorted({item["model_name"] for item in resp_json["data"]})


# ---------------------------------------------------------------------------
# Default branch (no flags) — was leaking full router.model_list before fix
# ---------------------------------------------------------------------------


def test_v2_model_info_default_branch_filters_by_user_models(
    client, configure_router, monkeypatch
):
    """user.models=['claude-3-opus'] must narrow even when caller passes no flags.

    Pre-fix this branch returned the entire router.model_list verbatim —
    leaking every deployment's litellm_params (including api_base) to
    any virtual-key holder.
    """
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["claude-3-opus"])

    resp = client.get("/v2/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == ["claude-3-opus"]


def test_v2_model_info_default_branch_no_filter_for_master_key(
    client, configure_router, monkeypatch
):
    """Master key (user_id=None) -> no narrowing, full deployment list."""

    async def _should_not_be_called(*args, **kwargs):
        raise AssertionError("get_user_object must not be called when user_id is None")

    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.get_user_object",
        _should_not_be_called,
    )

    _override_auth(user_id=None)

    resp = client.get("/v2/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == sorted({m["model_name"] for m in _PROXY_MODELS})


def test_v2_model_info_default_branch_no_filter_when_user_models_empty(
    client, configure_router, monkeypatch
):
    """user.models == [] -> unrestricted, full deployment list returned."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=[])

    resp = client.get("/v2/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == sorted({m["model_name"] for m in _PROXY_MODELS})


def test_v2_model_info_default_branch_no_default_models_returns_empty(
    client, configure_router, monkeypatch
):
    """`no-default-models` sentinel -> /v2/model/info returns []."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["no-default-models"])

    resp = client.get("/v2/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_v2_model_info_default_branch_all_proxy_models_no_filter(
    client, configure_router, monkeypatch
):
    """`all-proxy-models` sentinel -> user gets the full deployment list."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["all-proxy-models"])

    resp = client.get("/v2/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == sorted({m["model_name"] for m in _PROXY_MODELS})


def test_v2_model_info_default_branch_user_wildcard(
    client, configure_router, monkeypatch
):
    """user.models contains 'anthropic/*' -> only matching deployments returned."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["anthropic/*"])

    resp = client.get("/v2/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == [
        "anthropic/claude-3-5-sonnet",
        "anthropic/claude-3-7-sonnet",
    ]


def test_v2_model_info_user_models_takes_precedence_over_permissive_key(
    client, configure_router, monkeypatch
):
    """Even with key.models=['all-proxy-models'], user.models still narrows."""

    def _auth_with_all_proxy():
        return UserAPIKeyAuth(
            api_key="test-key",
            user_id="u-test",
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=["all-proxy-models"],
            team_id=None,
            team_models=[],
        )

    app.dependency_overrides[ps.user_api_key_auth] = _auth_with_all_proxy
    _patch_user(monkeypatch, models=["claude-3-haiku"])

    resp = client.get("/v2/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == ["claude-3-haiku"]


def test_v2_model_info_default_branch_preserves_litellm_params(
    client, configure_router, monkeypatch
):
    """Sanity: filter only drops disallowed deployments — surviving entries
    retain their litellm_params payload (model, api_base, etc.)."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["claude-3-opus"])

    resp = client.get("/v2/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["model_name"] == "claude-3-opus"
    # litellm_params is enriched but still carries the original "model" key
    assert data[0]["litellm_params"]["model"] == "anthropic/claude-3-opus"


# ---------------------------------------------------------------------------
# include_team_models=true branch — user.models must STILL narrow after the
# endpoint expands the list via team access. This is the path the UI hits
# (modelInfoCall in networking.tsx always passes include_team_models=true).
# ---------------------------------------------------------------------------


def test_v2_model_info_include_team_models_branch_still_filters_user_models(
    client, configure_router, monkeypatch
):
    """include_team_models=true expands the deployment list via team access
    (every model marked `access_via_team_ids=['t-a']`), simulating a user
    in a team with full proxy access. The user-level filter must still
    narrow the result to user.models = ['claude-3-opus'].

    Without the defense-in-depth final-step filter, the endpoint would
    leak all 5 deployments.
    """
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["claude-3-opus"])

    async def _fake_team_access(
        *, user_api_key_dict, prisma_client, llm_router, all_models
    ):
        for m in all_models:
            m.setdefault("model_info", {})
            m["model_info"]["access_via_team_ids"] = ["t-a"]
        return all_models

    monkeypatch.setattr(ps, "get_all_team_and_direct_access_models", _fake_team_access)

    resp = client.get(
        "/v2/model/info?include_team_models=true",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert resp.status_code == 200
    assert _model_names(resp.json()) == ["claude-3-opus"]


def test_v2_model_info_include_team_models_unrestricted_user_sees_team_set(
    client, configure_router, monkeypatch
):
    """Open user (models=[]) + include_team_models=true → no user-level
    narrowing, returns whatever the team-access path returns.

    Verifies the defense-in-depth filter doesn't accidentally over-narrow
    when user.models is empty: it must be a pure pass-through in that case.
    """
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=[])

    async def _fake_team_access(
        *, user_api_key_dict, prisma_client, llm_router, all_models
    ):
        # Team grants access to only 2 of the 5 — keep them, drop the rest.
        allowed_via_team = {"claude-3-opus", "claude-3-haiku"}
        filtered = []
        for m in all_models:
            if m["model_name"] in allowed_via_team:
                m.setdefault("model_info", {})
                m["model_info"]["access_via_team_ids"] = ["t-a"]
                filtered.append(m)
        return filtered

    monkeypatch.setattr(ps, "get_all_team_and_direct_access_models", _fake_team_access)

    resp = client.get(
        "/v2/model/info?include_team_models=true",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert resp.status_code == 200
    assert _model_names(resp.json()) == ["claude-3-haiku", "claude-3-opus"]


def test_v2_model_info_include_team_models_no_default_models_returns_empty(
    client, configure_router, monkeypatch
):
    """`no-default-models` sentinel must wipe the list even when a team
    granted access. Inference-time `can_user_call_model` raises on this
    sentinel; the discovery endpoint must agree."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["no-default-models"])

    async def _fake_team_access(
        *, user_api_key_dict, prisma_client, llm_router, all_models
    ):
        for m in all_models:
            m.setdefault("model_info", {})
            m["model_info"]["access_via_team_ids"] = ["t-a"]
        return all_models

    monkeypatch.setattr(ps, "get_all_team_and_direct_access_models", _fake_team_access)

    resp = client.get(
        "/v2/model/info?include_team_models=true",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == []
