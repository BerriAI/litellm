"""Behavior pins for ``proxy_server.py`` model routes.

Pins (PR2):
    - GET /v1/models
    - GET /models
    - GET /v1/models/{model_id}
    - GET /models/{model_id}
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import litellm
from litellm.proxy import proxy_server

from .conftest import normalize  # type: ignore[import-not-found]


def _stub_model_info_response(
    model_id: str = "gpt-4", provider: str = "openai"
) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": provider,
    }


@pytest.fixture
def patched_models(monkeypatch):
    """Stub router + utility helpers used by the /models routes."""
    from litellm.proxy import utils as proxy_utils

    router = MagicMock()
    router.get_fully_blocked_model_names = MagicMock(return_value=set())
    router.get_model_names = MagicMock(return_value=["gpt-4", "claude-sonnet"])
    router.get_model_access_groups = MagicMock(return_value={})

    deployment = MagicMock()
    deployment.litellm_params.model = "gpt-4"
    router.get_deployment_by_model_group_name = MagicMock(return_value=deployment)

    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())

    async def _fake_get_available_models_for_user(**kwargs):
        return ["gpt-4", "claude-sonnet"]

    monkeypatch.setattr(
        proxy_utils,
        "get_available_models_for_user",
        _fake_get_available_models_for_user,
    )

    def _fake_create_model_info_response(model_id, provider="openai", **kwargs):
        return _stub_model_info_response(model_id=model_id, provider=provider)

    monkeypatch.setattr(
        proxy_utils, "create_model_info_response", _fake_create_model_info_response
    )

    monkeypatch.setattr(proxy_utils, "validate_model_access", lambda **kwargs: None)

    monkeypatch.setattr(
        litellm,
        "get_llm_provider",
        lambda model: (model, "openai", None, None),
    )

    return router


@pytest.mark.parametrize("path", ["/v1/models", "/models"])
def test_get_models_happy_path(client, auth_as, patched_models, path):
    """Pins: ``GET /v1/models``, ``GET /models``."""
    with auth_as():
        response = client.get(path)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "data": [
            {
                "id": "<VOLATILE>",
                "object": "model",
                "created": "<VOLATILE>",
                "owned_by": "openai",
            },
            {
                "id": "<VOLATILE>",
                "object": "model",
                "created": "<VOLATILE>",
                "owned_by": "openai",
            },
        ],
        "object": "list",
    }


@pytest.mark.parametrize("path", ["/v1/models", "/models"])
def test_get_models_invalid_scope_returns_400(client, auth_as, patched_models, path):
    """Pins: ``GET /v1/models``, ``GET /models`` (error path: invalid scope)."""
    with auth_as():
        response = client.get(path, params={"scope": "not-a-real-scope"})
    assert response.status_code == 400
    assert "Invalid scope parameter" in str(response.json())


@pytest.mark.parametrize("path", ["/v1/models/gpt-4", "/models/gpt-4"])
def test_get_model_by_id_happy_path(client, auth_as, patched_models, path):
    """Pins: ``GET /v1/models/{model_id}``, ``GET /models/{model_id}``."""
    with auth_as():
        response = client.get(path)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "object": "model",
        "created": "<VOLATILE>",
        "owned_by": "openai",
    }


@pytest.mark.parametrize("path", ["/v1/models/missing", "/models/missing"])
def test_get_model_by_id_not_found(client, auth_as, patched_models, path):
    """Pins: ``GET /v1/models/{model_id}``, ``GET /models/{model_id}`` (error: 404)."""
    patched_models.get_deployment_by_model_group_name = MagicMock(return_value=None)
    with auth_as():
        response = client.get(path)
    assert response.status_code == 404
    assert "not found" in response.text.lower()


@pytest.mark.parametrize("path", ["/v1/models", "/models"])
def test_get_models_reports_real_provider_per_model(client, auth_as, monkeypatch, path):
    """Regression: `owned_by` must reflect each model's actual provider.

    GET /v1/models previously hardcoded `provider="openai"` at its callsite for
    every model, unlike GET /v1/models/{model_id} which already resolved the
    real provider from the deployment. This exercises the real router +
    provider-resolution stack (no mocking of create_model_info_response or
    litellm.get_llm_provider), so it fails if the listing callsite regresses
    to a hardcoded provider.
    """
    from litellm import Router
    from litellm.proxy import utils as proxy_utils

    router = Router(
        model_list=[
            {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}},
            {
                "model_name": "claude-sonnet",
                "litellm_params": {"model": "anthropic/claude-3-5-sonnet-latest"},
            },
        ]
    )

    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())

    async def _fake_get_available_models_for_user(**kwargs):
        return ["gpt-4", "claude-sonnet"]

    monkeypatch.setattr(
        proxy_utils,
        "get_available_models_for_user",
        _fake_get_available_models_for_user,
    )

    with auth_as():
        response = client.get(path)

    assert response.status_code == 200
    owned_by = {m["id"]: m["owned_by"] for m in response.json()["data"]}
    assert owned_by == {"gpt-4": "openai", "claude-sonnet": "anthropic"}
