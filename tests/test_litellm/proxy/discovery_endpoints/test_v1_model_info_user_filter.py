"""
Route-level tests for /v1/model/info (and the /model/info alias) —
verify that LiteLLM_UserTable.models ("Personal Models") narrows the
returned deployment list on Path B (no `litellm_model_id` query
param), closing the discovery-vs-inference gap described in
BerriAI/litellm#26420 and the follow-up audit that found it open on
this sibling endpoint.

These tests exercise the real FastAPI route → real user_api_key_auth
dependency override → real model_info_v1 handler → real
get_available_models_for_user → real _apply_user_models_filter →
mocked get_user_object. No real DB, no live HTTP server.
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


def _patch_user(monkeypatch, models):
    async def _fake(*args, **kwargs):
        return LiteLLM_UserTable(
            user_id="u-test",
            max_budget=None,
            user_email=None,
            models=models,
        )

    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.get_user_object",
        _fake,
    )


def _model_names(resp_json):
    return sorted({item["model_name"] for item in resp_json["data"]})


# ---------------------------------------------------------------------------
# /v1/model/info Path B (no litellm_model_id) — the leak path
# ---------------------------------------------------------------------------


def test_v1_model_info_filters_by_user_personal_models(
    client, configure_router, monkeypatch
):
    """Headline regression: user.models=['claude-3-opus'] must narrow the list."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["claude-3-opus"])

    resp = client.get("/v1/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == ["claude-3-opus"]


def test_v1_model_info_no_filter_when_user_models_empty(
    client, configure_router, monkeypatch
):
    """user.models == [] -> unrestricted, full deployment list returned."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=[])

    resp = client.get("/v1/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == sorted({m["model_name"] for m in _PROXY_MODELS})


def test_v1_model_info_no_filter_when_user_id_is_none(
    client, configure_router, monkeypatch
):
    """Master key / service account (user_id=None) -> no filter applied."""

    async def _should_not_be_called(*args, **kwargs):
        raise AssertionError("get_user_object must not be called when user_id is None")

    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.get_user_object",
        _should_not_be_called,
    )

    _override_auth(user_id=None)

    resp = client.get("/v1/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == sorted({m["model_name"] for m in _PROXY_MODELS})


def test_v1_model_info_no_default_models_returns_empty(
    client, configure_router, monkeypatch
):
    """`no-default-models` sentinel -> /v1/model/info returns []."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["no-default-models"])

    resp = client.get("/v1/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_v1_model_info_all_proxy_models_no_filter(
    client, configure_router, monkeypatch
):
    """`all-proxy-models` sentinel -> user gets the full deployment list."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["all-proxy-models"])

    resp = client.get("/v1/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == sorted({m["model_name"] for m in _PROXY_MODELS})


def test_v1_model_info_user_wildcard(client, configure_router, monkeypatch):
    """user.models contains 'anthropic/*' -> only matching deployments returned."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["anthropic/*"])

    resp = client.get("/v1/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == [
        "anthropic/claude-3-5-sonnet",
        "anthropic/claude-3-7-sonnet",
    ]


def test_v1_model_info_user_models_takes_precedence_over_permissive_key(
    client, configure_router, monkeypatch
):
    """Even with key.models=['all-proxy-models'], user.models still narrows.

    Parity claim: /v1/model/info matches /v1/models which matches
    can_user_call_model at inference time.
    """

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

    resp = client.get("/v1/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == ["claude-3-haiku"]


def test_v1_model_info_alias_route_filters_identically(
    client, configure_router, monkeypatch
):
    """Sanity: /model/info (without /v1/ prefix) shares the same handler
    and therefore the same filter behavior."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["claude-3-opus"])

    resp = client.get("/model/info", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_names(resp.json()) == ["claude-3-opus"]
