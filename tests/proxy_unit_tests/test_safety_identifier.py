import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing


def test_enforce_safety_identifier_hashes_authenticated_user(monkeypatch):
    monkeypatch.setenv("LITELLM_ENFORCE_SAFETY_IDENTIFIER", "true")

    result = ProxyBaseLLMRequestProcessing._enforce_safety_identifier(
        data={"safety_identifier": "caller-value"},
        route_type="acompletion",
        user_api_key_dict=UserAPIKeyAuth(user_id="user-123"),
    )

    assert result["safety_identifier"] == hashlib.sha256(b"user-123").hexdigest()


@pytest.mark.parametrize("setting", [None, "false"])
def test_enforce_safety_identifier_is_opt_in(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("LITELLM_ENFORCE_SAFETY_IDENTIFIER", raising=False)
    else:
        monkeypatch.setenv("LITELLM_ENFORCE_SAFETY_IDENTIFIER", setting)
    data = {"safety_identifier": "caller-value"}

    result = ProxyBaseLLMRequestProcessing._enforce_safety_identifier(
        data=data,
        route_type="acompletion",
        user_api_key_dict=UserAPIKeyAuth(user_id="user-123"),
    )

    assert result == data


def test_enforce_safety_identifier_skips_missing_user_id(monkeypatch):
    monkeypatch.setenv("LITELLM_ENFORCE_SAFETY_IDENTIFIER", "true")
    data = {"safety_identifier": "caller-value"}

    result = ProxyBaseLLMRequestProcessing._enforce_safety_identifier(
        data=data,
        route_type="aresponses",
        user_api_key_dict=UserAPIKeyAuth(user_id=None),
    )

    assert result == data


def test_enforce_safety_identifier_only_applies_to_openai_generation_routes(monkeypatch):
    monkeypatch.setenv("LITELLM_ENFORCE_SAFETY_IDENTIFIER", "true")
    data = {"safety_identifier": "caller-value"}

    result = ProxyBaseLLMRequestProcessing._enforce_safety_identifier(
        data=data,
        route_type="aembedding",
        user_api_key_dict=UserAPIKeyAuth(user_id="user-123"),
    )

    assert result == data


@pytest.mark.parametrize(
    ("provider", "model"),
    [("anthropic", "claude-3-5-sonnet-20241022"), ("gemini", "gemini-2.0-flash")],
)
def test_unsupported_safety_identifier_is_dropped_by_provider_translation(provider, model):
    result = litellm.get_optional_params(
        model=model,
        custom_llm_provider=provider,
        safety_identifier="trusted-value",
    )

    assert "safety_identifier" not in result


def test_supported_safety_identifier_is_preserved_by_provider_translation():
    result = litellm.get_optional_params(
        model="gpt-4o",
        custom_llm_provider="openai",
        safety_identifier="trusted-value",
    )

    assert result["safety_identifier"] == "trusted-value"


@pytest.mark.asyncio
@pytest.mark.parametrize("route_type", ["acompletion", "aresponses"])
async def test_pre_call_hook_cannot_override_enforced_safety_identifier(monkeypatch, route_type):
    monkeypatch.setenv("LITELLM_ENFORCE_SAFETY_IDENTIFIER", "true")
    request = MagicMock(spec=Request)
    request.headers.get.return_value = "call-id"
    logging_obj = MagicMock()
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.pre_call_hook = AsyncMock(return_value={"model": "gpt-5", "safety_identifier": "hook-value"})
    user_api_key_dict = UserAPIKeyAuth(user_id="user-123")
    processor = ProxyBaseLLMRequestProcessing(data={"model": "gpt-5", "safety_identifier": "caller-value"})

    with (
        patch(  # test-quality-ok: isolate shared pre-call ordering without making an upstream request
            "litellm.proxy.common_request_processing.add_litellm_data_to_request",
            new=AsyncMock(side_effect=lambda **kwargs: kwargs["data"]),
        ),
        patch(  # test-quality-ok: isolate shared pre-call ordering without initializing logging callbacks
            "litellm.proxy.common_request_processing.litellm.utils.function_setup",
            return_value=(logging_obj, processor.data),
        ),
        patch(  # test-quality-ok: isolate shared pre-call ordering from router configuration
            "litellm.proxy.common_request_processing._check_and_merge_model_level_guardrails",
            side_effect=lambda **kwargs: kwargs["data"],
        ),
        patch(  # test-quality-ok: isolate shared pre-call ordering from optional compression hooks
            "litellm.proxy.common_request_processing._arm_auto_router_compression",
            new=AsyncMock(),
        ),
    ):
        result, _ = await processor.common_processing_pre_call_logic(
            request=request,
            general_settings={},
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=MagicMock(),
            route_type=route_type,
            version="test",
        )

    assert result["safety_identifier"] == hashlib.sha256(b"user-123").hexdigest()
