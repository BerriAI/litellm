"""
VERIA-44: ``router_settings_override.fallbacks`` must be validated
against the API key's model allowlist at auth time. Without this, the
override is promoted to per-request kwargs after auth and lets a caller
execute requests against models their API key cannot call.
"""

from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import iter_request_fallback_targets
from litellm.proxy.auth.user_api_key_auth import (
    _enforce_key_and_fallback_model_access,
    _fallback_target_model_name,
)


def _fallback_model_names(fallbacks):
    """Model names the auth check validates for a top-level ``fallbacks`` value."""
    return [
        name
        for target in iter_request_fallback_targets({"fallbacks": fallbacks})
        if (name := _fallback_target_model_name(target)) is not None
    ]


def _key_with_models(models: List[str]) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed",
        user_id="caller",
        user_role=LitellmUserRoles.INTERNAL_USER,
        models=models,
    )


# ── fallback model-name extraction ───────────────────────────────────────────


def test_fallback_model_names_router_config_shape():
    """Router-config shape: ``[{primary: [fallback_list]}]``."""
    assert _fallback_model_names(
        [{"gpt-3.5-turbo": ["gpt-4", "claude-3"]}, {"gpt-4o": ["o1"]}]
    ) == ["gpt-4", "claude-3", "o1"]


def test_fallback_model_names_simple_string_shape():
    """Simple top-level shape: list of strings."""
    assert _fallback_model_names(["gpt-4", "claude-3"]) == ["gpt-4", "claude-3"]


def test_fallback_model_names_client_side_shape():
    """ClientSideFallbackModel shape: ``[{"model": "..."}]``."""
    assert _fallback_model_names([{"model": "gpt-4"}, {"model": "claude-3"}]) == [
        "gpt-4",
        "claude-3",
    ]


def test_fallback_model_names_nested_deployment_fallbacks():
    """A deployment target's own nested fallback field is unrolled too."""
    assert _fallback_model_names(
        [{"primary": [{"model": "gpt-4", "fallbacks": [{"gpt-4": ["deepseek-chat"]}]}]}]
    ) == ["gpt-4", "deepseek-chat"]


def test_fallback_model_names_empty_or_none():
    assert _fallback_model_names(None) == []
    assert _fallback_model_names([]) == []
    assert _fallback_model_names("not a list") == []


# ── _enforce_key_and_fallback_model_access ────────────────────────────────────


@pytest.mark.asyncio
async def test_router_override_fallbacks_validated_against_key_allowlist():
    """A fallback nested inside ``router_settings_override`` is validated
    against the API key's allowed models — not just the top-level
    ``fallbacks`` field."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        "router_settings_override": {
            "fallbacks": [{"gpt-3.5-turbo": ["unauthorized-model"]}],
        },
    }

    seen_models: List[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen_models.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await _enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    # Both the primary model and the override-nested fallback must be
    # checked against the API key's allowlist.
    assert "gpt-3.5-turbo" in seen_models
    assert "unauthorized-model" in seen_models


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallback_field",
    [
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
    ],
)
async def test_router_override_all_fallback_fields_validated(fallback_field):
    """All three fallback fields the router accepts as per-request kwargs
    are validated — context_window_fallbacks and content_policy_fallbacks
    are promoted in route_llm_request.py too."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        "router_settings_override": {
            fallback_field: [{"gpt-3.5-turbo": ["smuggled-model"]}],
        },
    }

    seen: List[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await _enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    assert "smuggled-model" in seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallback_field",
    [
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
    ],
)
async def test_top_level_fallback_fields_validated(fallback_field):
    """All three top-level fallback fields are forwarded to the router as
    per-request kwargs, so all three must be validated against the API
    key's allowlist. Greptile P1 follow-up: previously only the
    ``fallbacks`` field was walked at the top level."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        fallback_field: [{"gpt-3.5-turbo": ["top-level-smuggled"]}],
    }

    seen: List[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await _enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    assert "top-level-smuggled" in seen


@pytest.mark.asyncio
async def test_nested_deployment_fallback_inner_model_validated():
    """A model name nested several fallback rounds deep, inside a deployment
    target's own ``fallbacks``, is extracted and passed to can_key_call_model."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        "fallbacks": [
            {
                "gpt-3.5-turbo": [
                    {
                        "model": "gpt-3.5-turbo",
                        "fallbacks": [{"gpt-3.5-turbo": ["deep-smuggled-model"]}],
                    }
                ]
            }
        ],
    }

    seen: List[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await _enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    assert "deep-smuggled-model" in seen


@pytest.mark.asyncio
async def test_model_less_fallback_dict_is_skipped_never_passed_as_none():
    """A fallback target dict without a ``model`` key is skipped, never passed
    as ``None`` into can_key_call_model / is_valid_fallback_model."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        "fallbacks": [
            {
                "gpt-3.5-turbo": [
                    {"model": "real-fallback"},
                    {"api_base": "http://attacker"},
                    "string-fallback",
                ]
            }
        ],
    }

    seen: List[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await _enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    assert None not in seen
    assert seen == ["gpt-3.5-turbo", "real-fallback", "string-fallback"]


@pytest.mark.asyncio
async def test_router_override_without_fallbacks_does_not_break_auth():
    """``router_settings_override`` set without any fallback fields is a
    no-op for the auth check — only the primary model is validated."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        "router_settings_override": {"num_retries": 3, "timeout": 30},
    }

    seen: List[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await _enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    assert seen == ["gpt-3.5-turbo"]
