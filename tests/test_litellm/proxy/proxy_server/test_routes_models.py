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
from litellm.proxy import utils as proxy_utils
from litellm.proxy.utils import create_model_info_response

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
def test_get_models_anthropic_format_when_header_present(
    client, auth_as, patched_models, path
):
    """Pins: ``GET /v1/models`` returns the Anthropic-native models shape when
    the caller sends an ``anthropic-version`` header (Claude Code gateway
    discovery), while the default OpenAI shape is unchanged without it."""
    with auth_as():
        response = client.get(path, headers={"anthropic-version": "2023-06-01"})
    assert response.status_code == 200
    body = response.json()
    assert "object" not in body
    assert body["has_more"] is False
    assert body["first_id"] == "gpt-4"
    assert body["last_id"] == "claude-sonnet"
    assert [m["id"] for m in body["data"]] == ["gpt-4", "claude-sonnet"]
    for entry in body["data"]:
        assert entry["type"] == "model"
        assert entry["display_name"] == entry["id"]
        assert entry["created_at"].endswith("Z")


@pytest.mark.parametrize("path", ["/v1/models", "/models"])
def test_anthropic_format_exposes_token_limits(
    client, auth_as, patched_models, monkeypatch, path
):
    """Claude Code sizes requests off the listing, so the Anthropic-native entries
    carry the same token limits the OpenAI listing resolves, with the output budget
    named max_tokens as the Messages API names it."""
    from litellm.proxy import utils as proxy_utils

    def _create_model_info_response(model_id, provider="openai", **kwargs):
        if model_id != "claude-sonnet":
            return _stub_model_info_response(model_id=model_id, provider=provider)
        return {
            **_stub_model_info_response(model_id=model_id, provider=provider),
            "max_input_tokens": 200000,
            "max_output_tokens": 64000,
        }

    monkeypatch.setattr(
        proxy_utils, "create_model_info_response", _create_model_info_response
    )

    with auth_as():
        response = client.get(path, headers={"anthropic-version": "2023-06-01"})

    assert response.status_code == 200
    gpt_4, claude = response.json()["data"]
    assert claude["max_input_tokens"] == 200000
    assert claude["max_tokens"] == 64000
    assert "max_output_tokens" not in claude
    assert gpt_4["max_input_tokens"] is None
    assert gpt_4["max_tokens"] is None


@pytest.mark.parametrize("path", ["/v1/models", "/models"])
def test_anthropic_format_carries_router_configured_token_limits(client, auth_as, patched_models, monkeypatch, path):
    """Pins the whole resolution chain, not just the formatter: a deployment's
    configured limits beat the cost map, and the configured output budget is what
    lands on the Anthropic ``max_tokens``. All eight limits differ, so an entry
    built from another entry's lookup shows up as the wrong numbers."""

    def _configured(model_name):
        return (300000, 32000) if model_name == "gpt-4" else (500000, 4096)

    def _cost_map_lookup(model_id):
        max_input, max_output = (200000, 64000) if model_id == "gpt-4" else (100000, 8000)
        return {"max_input_tokens": max_input, "max_output_tokens": max_output, "mode": "chat"}

    patched_models.get_configured_token_limits = MagicMock(side_effect=_configured)

    def _resolved(**kwargs):
        return create_model_info_response(**kwargs, get_model_info=_cost_map_lookup)

    monkeypatch.setattr(proxy_utils, "create_model_info_response", _resolved)

    with auth_as():
        response = client.get(path, headers={"anthropic-version": "2023-06-01"})

    assert response.status_code == 200
    gpt_4, claude = response.json()["data"]
    assert (gpt_4["max_input_tokens"], gpt_4["max_tokens"]) == (300000, 32000)
    assert (claude["max_input_tokens"], claude["max_tokens"]) == (500000, 4096)


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


@pytest.mark.parametrize("params", [{}, {"scope": "expand"}])
def test_anthropic_format_returns_public_team_model_name(
    client, auth_as, patched_models, monkeypatch, params
):
    """Regression: the Anthropic-native listing must go through the same team
    name translation as the OpenAI listing, so a caller never sees the internal
    ``model_name_{team_id}_{uuid}`` routing key."""
    from litellm.proxy import utils as proxy_utils
    from litellm.proxy.auth import model_checks

    internal_name = "model_name_team-1_c0ffee"

    patched_models.get_model_list = MagicMock(
        return_value=[
            {
                "model_name": internal_name,
                "model_info": {
                    "team_id": "team-1",
                    "team_public_model_name": "gpt-4-team",
                },
            }
        ]
    )
    patched_models.get_model_names = MagicMock(return_value=[internal_name])

    async def _fake_get_available_models_for_user(**kwargs):
        return [internal_name]

    monkeypatch.setattr(
        proxy_utils,
        "get_available_models_for_user",
        _fake_get_available_models_for_user,
    )
    monkeypatch.setattr(
        model_checks, "get_complete_model_list", lambda **kwargs: [internal_name]
    )

    with auth_as():
        response = client.get(
            "/v1/models", params=params, headers={"anthropic-version": "2023-06-01"}
        )

    assert response.status_code == 200
    assert [m["id"] for m in response.json()["data"]] == ["gpt-4-team"]
    assert internal_name not in response.text
