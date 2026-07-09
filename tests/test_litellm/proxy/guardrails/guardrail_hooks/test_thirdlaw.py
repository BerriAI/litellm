import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.thirdlaw import (
    ThirdlawGuardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.thirdlaw.thirdlaw import (
    ThirdlawGuardrailMissingConfig,
)
from litellm.types.guardrails import LitellmParams, SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.thirdlaw import (
    ThirdlawGuardrailConfigModel,
)

_ENDPOINT = "https://thirdlaw.test/evaluate"


def _mock_response(body: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("POST", _ENDPOINT),
    )


def _make_guardrail(
    *,
    unreachable_fallback="fail_closed",
    api_base=_ENDPOINT,
    api_key="thirdlaw_secret",
    guardrail_name="thirdlaw-guard",
    additional_headers=None,
) -> ThirdlawGuardrail:
    return ThirdlawGuardrail(
        api_base=api_base,
        api_key=api_key,
        unreachable_fallback=unreachable_fallback,
        guardrail_name=guardrail_name,
        event_hook="pre_call",
        default_on=True,
        additional_headers=additional_headers,
    )


def test_requires_api_base(monkeypatch):
    monkeypatch.delenv("THIRDLAW_API_BASE", raising=False)
    with pytest.raises(ThirdlawGuardrailMissingConfig):
        ThirdlawGuardrail(api_key="k")


def test_appends_generic_guardrail_path():
    g = _make_guardrail(api_base="https://thirdlaw.test/evaluate")
    assert (
        g.api_base == "https://thirdlaw.test/evaluate/beta/litellm_basic_guardrail_api"
    )


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("THIRDLAW_API_BASE", "https://env.thirdlaw.test/evaluate")
    monkeypatch.setenv("THIRDLAW_API_KEY", "env_key")
    g = ThirdlawGuardrail(
        guardrail_name="thirdlaw",
        event_hook="pre_call",
        default_on=True,
    )
    assert (
        g.api_base
        == "https://env.thirdlaw.test/evaluate/beta/litellm_basic_guardrail_api"
    )
    assert g.headers["x-api-key"] == "env_key"


def test_additional_headers():
    g = _make_guardrail(additional_headers="x-request-id ,x-correlation-id")
    assert g.extra_headers == ["x-request-id", "x-correlation-id"]


async def test_none_action_passthrough():
    g = _make_guardrail()
    mock_response = _mock_response({"action": "NONE"})
    with patch.object(
        g.async_handler, "post", new_callable=AsyncMock, return_value=mock_response
    ):
        out = await g.apply_guardrail(
            inputs={"texts": ["hello"]}, request_data={}, input_type="request"
        )
    assert out["texts"] == ["hello"]


async def test_guardrail_intervened_replaces_texts():
    g = _make_guardrail()
    mock_response = _mock_response(
        {"action": "GUARDRAIL_INTERVENED", "texts": ["[REDACTED]"]}
    )
    with patch.object(
        g.async_handler, "post", new_callable=AsyncMock, return_value=mock_response
    ):
        out = await g.apply_guardrail(
            inputs={"texts": ["my ssn is 123"]}, request_data={}, input_type="request"
        )
    assert out["texts"] == ["[REDACTED]"]


async def test_blocked_raises():
    g = _make_guardrail()
    mock_response = _mock_response(
        {"action": "BLOCKED", "blocked_reason": "policy violation"}
    )
    with patch.object(
        g.async_handler, "post", new_callable=AsyncMock, return_value=mock_response
    ):
        with pytest.raises(GuardrailRaisedException) as exc_info:
            await g.apply_guardrail(
                inputs={"texts": ["bad"]}, request_data={}, input_type="request"
            )
    assert exc_info.value.status_code == 400


async def test_posts_to_configured_endpoint_with_bearer_auth():
    g = _make_guardrail(api_base="https://thirdlaw.test/evaluate", api_key="secret")
    mock_response = _mock_response({"action": "NONE"})
    with patch.object(
        g.async_handler, "post", new_callable=AsyncMock, return_value=mock_response
    ) as mock_post:
        await g.apply_guardrail(
            inputs={"texts": ["hello"], "model": "gpt-4o"},
            request_data={"metadata": {"user_api_key_user_id": "u1"}},
            input_type="request",
        )
    call_kwargs = mock_post.call_args.kwargs
    assert (
        call_kwargs["url"]
        == "https://thirdlaw.test/evaluate/beta/litellm_basic_guardrail_api"
    )
    assert "x-api-key" in call_kwargs["headers"]
    assert call_kwargs["headers"]["x-api-key"] == "secret"
    assert call_kwargs["json"]["texts"] == ["hello"]
    assert call_kwargs["json"]["input_type"] == "request"
    assert call_kwargs["json"]["model"] == "gpt-4o"
    assert call_kwargs["json"]["request_data"]["user_api_key_user_id"] == "u1"
    assert "secret" not in json.dumps(call_kwargs["json"])


def test_apply_guardrail_is_defined_on_class_for_routing():
    assert "apply_guardrail" in ThirdlawGuardrail.__dict__


def test_enum_value():
    assert SupportedGuardrailIntegrations.THIRDLAW.value == "thirdlaw"


def test_config_model_ui_name():
    assert ThirdlawGuardrailConfigModel.ui_friendly_name() == "ThirdLaw"


def test_registries_expose_initializer_and_class():
    assert "thirdlaw" in guardrail_initializer_registry
    assert guardrail_class_registry["thirdlaw"] is ThirdlawGuardrail


def test_config_driven_initialization_creates_callback():
    lp = LitellmParams(
        guardrail="thirdlaw",
        mode="pre_call",
        api_base="https://thirdlaw.test/evaluate",
        api_key="k",
    )
    cb = initialize_guardrail(lp, {"guardrail_name": "thirdlaw-guard"})
    assert isinstance(cb, ThirdlawGuardrail)
    assert cb.unreachable_fallback == "fail_closed"
    assert (
        cb.api_base == "https://thirdlaw.test/evaluate/beta/litellm_basic_guardrail_api"
    )


def test_streaming_defaults_to_end_of_stream_only():
    g = _make_guardrail()
    assert g.streaming_end_of_stream_only is True
    assert g.streaming_sampling_rate == 5


def test_config_model_streaming_defaults():
    model = ThirdlawGuardrailConfigModel(api_base=_ENDPOINT)
    assert model.streaming_end_of_stream_only is True
    assert model.streaming_sampling_rate == 5


def test_config_driven_initialization_propagates_streaming_overrides():
    lp = LitellmParams(
        guardrail="thirdlaw",
        mode="pre_call",
        api_base="https://thirdlaw.test/evaluate",
        api_key="k",
        streaming_end_of_stream_only=False,
        streaming_sampling_rate=10,
    )
    cb = initialize_guardrail(lp, {"guardrail_name": "thirdlaw-guard"})
    assert cb.streaming_end_of_stream_only is False
    assert cb.streaming_sampling_rate == 10
