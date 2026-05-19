"""
Route-level tests for /v1/models — verify that LiteLLM_UserTable.models
("Personal Models") is honored, closing the inconsistency between the
discovery endpoint and inference (BerriAI/litellm#26420).

These tests exercise the real FastAPI route → real user_api_key_auth
dependency override → real model_list handler → real
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

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


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
    """Wire a real Router with a known model list onto proxy_server globals."""
    import litellm

    router = litellm.Router(
        model_list=_PROXY_MODELS,
        model_group_alias={},
    )
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "llm_model_list", _PROXY_MODELS)
    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "general_settings", {})
    # _apply_user_models_filter only needs prisma_client to be truthy;
    # the actual user fetch is mocked below.
    monkeypatch.setattr(ps, "prisma_client", MagicMock())
    monkeypatch.setattr(ps, "user_api_key_cache", DualCache())
    return router


def _override_auth(user_id):
    """Install an auth override that returns a UserAPIKeyAuth with the
    given user_id (or None for master-key-style requests). The key itself
    grants 'all-proxy-models' so key-level filtering is a no-op and we
    can isolate the user-level filter under test.
    """

    def _auth():
        return UserAPIKeyAuth(
            api_key="test-key",
            user_id=user_id,
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=[],  # empty key.models → falls through to proxy list
            team_id=None,
            team_models=[],
        )

    app.dependency_overrides[ps.user_api_key_auth] = _auth


@pytest.fixture(autouse=True)
def _clear_auth_override():
    yield
    app.dependency_overrides.pop(ps.user_api_key_auth, None)


def _patch_user(monkeypatch, models):
    """Make get_user_object (as imported inside _apply_user_models_filter)
    return a user with the given Personal Models list."""

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


def _model_ids(resp_json):
    return [item["id"] for item in resp_json["data"]]


# ---------------------------------------------------------------------------
# Cases — these all hit GET /v1/models for real.
# ---------------------------------------------------------------------------


def test_v1_models_filters_by_user_personal_models(
    client, configure_router, monkeypatch
):
    """The headline bug: user.models = ['claude-3-opus'] must narrow the list."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["claude-3-opus"])

    resp = client.get("/v1/models", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_ids(resp.json()) == ["claude-3-opus"]


def test_v1_models_no_filter_when_user_models_empty(
    client, configure_router, monkeypatch
):
    """user.models == [] → unrestricted, full proxy list returned."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=[])

    resp = client.get("/v1/models", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    returned = set(_model_ids(resp.json()))
    assert returned == {m["model_name"] for m in _PROXY_MODELS}


def test_v1_models_no_filter_when_user_id_is_none(
    client, configure_router, monkeypatch
):
    """Master key / service account (user_id=None) → no filter applied."""

    async def _should_not_be_called(*args, **kwargs):
        raise AssertionError("get_user_object must not be called when user_id is None")

    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.get_user_object",
        _should_not_be_called,
    )

    _override_auth(user_id=None)

    resp = client.get("/v1/models", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert len(_model_ids(resp.json())) == len(_PROXY_MODELS)


def test_v1_models_no_default_models_returns_empty(
    client, configure_router, monkeypatch
):
    """`no-default-models` sentinel → /v1/models returns []."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["no-default-models"])

    resp = client.get("/v1/models", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_v1_models_all_proxy_models_no_filter(client, configure_router, monkeypatch):
    """`all-proxy-models` sentinel → user gets the full proxy list."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["all-proxy-models"])

    resp = client.get("/v1/models", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    returned = set(_model_ids(resp.json()))
    assert returned == {m["model_name"] for m in _PROXY_MODELS}


def test_v1_models_user_wildcard(client, configure_router, monkeypatch):
    """user.models contains 'anthropic/*' → only anthropic/* models returned."""
    _override_auth(user_id="u-test")
    _patch_user(monkeypatch, models=["anthropic/*"])

    resp = client.get("/v1/models", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    returned = set(_model_ids(resp.json()))
    assert returned == {
        "anthropic/claude-3-5-sonnet",
        "anthropic/claude-3-7-sonnet",
    }


def test_v1_models_user_models_takes_precedence_over_permissive_key(
    client, configure_router, monkeypatch
):
    """
    Even if the key has 'all-proxy-models', user.models still narrows the
    list. This is the parity claim: /v1/models matches can_user_call_model
    at inference time.
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

    resp = client.get("/v1/models", headers={"Authorization": "Bearer sk-test"})
    assert resp.status_code == 200
    assert _model_ids(resp.json()) == ["claude-3-haiku"]
