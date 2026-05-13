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
