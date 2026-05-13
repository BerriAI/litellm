import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.utils import ProxyLogging


def test_has_during_call_guardrails_ignores_non_guardrail_callbacks(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [CustomLogger(), "prometheus"])

    assert ProxyLogging.has_during_call_guardrails() is False


def test_has_during_call_guardrails_detects_guardrail_callbacks(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [CustomGuardrail()])

    assert ProxyLogging.has_during_call_guardrails() is True


def test_has_post_call_response_headers_callbacks_ignores_empty_callbacks(
    monkeypatch,
):
    monkeypatch.setattr(litellm, "callbacks", [])

    assert ProxyLogging.has_post_call_response_headers_callbacks() is False


def test_has_post_call_response_headers_callbacks_detects_custom_loggers(
    monkeypatch,
):
    monkeypatch.setattr(litellm, "callbacks", [CustomLogger()])

    assert ProxyLogging.has_post_call_response_headers_callbacks() is True


@pytest.mark.asyncio
async def test_post_call_response_headers_hook_returns_early_without_callbacks(
    monkeypatch,
):
    monkeypatch.setattr(litellm, "callbacks", [])
    proxy_logging_obj = ProxyLogging(user_api_key_cache={})  # type: ignore[arg-type]

    result = await proxy_logging_obj.post_call_response_headers_hook(
        data={},
        user_api_key_dict=None,  # type: ignore[arg-type]
        response=None,
        request_headers={},
    )

    assert result == {}
