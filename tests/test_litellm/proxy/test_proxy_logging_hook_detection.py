import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.utils import ProxyLogging


def test_has_post_call_response_headers_callbacks_ignores_empty_callbacks(
    monkeypatch,
):
    monkeypatch.setattr(litellm, "callbacks", [])

    assert ProxyLogging.has_post_call_response_headers_callbacks() is False


def test_has_post_call_response_headers_callbacks_requires_override(
    monkeypatch,
):
    """A vanilla ``CustomLogger`` inherits the no-op response-headers hook;
    the capability flag must stay False so the proxy can skip the headers
    loop entirely.  Only callbacks that *override* the hook should flip it."""
    monkeypatch.setattr(litellm, "callbacks", [CustomLogger()])
    assert ProxyLogging.has_post_call_response_headers_callbacks() is False

    class _AddsHeaders(CustomLogger):
        async def async_post_call_response_headers_hook(self, **kwargs):
            return {"x-custom": "1"}

    monkeypatch.setattr(litellm, "callbacks", [_AddsHeaders()])
    assert ProxyLogging.has_post_call_response_headers_callbacks() is True


def test_has_streaming_callbacks_uses_custom_logger_detection(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])
    assert ProxyLogging.has_streaming_callbacks() is False

    monkeypatch.setattr(litellm, "callbacks", [CustomLogger()])
    assert ProxyLogging.has_streaming_callbacks() is False

    class StreamingLogger(CustomLogger):
        async def async_post_call_streaming_hook(self, **kwargs):
            return kwargs.get("response")

    monkeypatch.setattr(litellm, "callbacks", [StreamingLogger()])
    assert ProxyLogging.has_streaming_callbacks() is True


def test_has_streaming_callbacks_detects_guardrails(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [CustomGuardrail()])
    assert ProxyLogging.has_streaming_callbacks() is True


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


def test_callback_capabilities_skips_default_custom_logger(monkeypatch):
    """
    Internal proxy hooks (e.g. _PROXY_MaxBudgetLimiter, ManagedFiles) inherit
    the default ``async_post_call_streaming_iterator_hook`` body.  The
    capability scanner must NOT report them as iterator overrides — wrapping
    the chunk stream through every no-op layer was responsible for ~10x
    streaming overhead on default deployments.
    """

    class _InternalNoopHook(CustomLogger):
        pass

    monkeypatch.setattr(litellm, "callbacks", [_InternalNoopHook()])

    caps = ProxyLogging._callback_capabilities()
    # Subclass inherits the base no-op for every hook — every capability flag
    # must stay False so the proxy short-circuits the corresponding loops.
    assert caps.has_post_call_response_headers is False
    assert caps.iterator_overrides == ()
    assert caps.has_iterator_override is False
    assert caps.has_streaming_chunk_override is False
    assert caps.has_guardrail is False


def test_callback_capabilities_captures_iterator_override(monkeypatch):
    class _OverridesIterator(CustomLogger):
        async def async_post_call_streaming_iterator_hook(  # type: ignore[override]
            self, user_api_key_dict, response, request_data
        ):
            async for item in response:
                yield item

    override = _OverridesIterator()
    monkeypatch.setattr(litellm, "callbacks", [override])

    caps = ProxyLogging._callback_capabilities()
    assert caps.has_iterator_override is True
    assert len(caps.iterator_overrides) == 1
    resolved, kind = caps.iterator_overrides[0]
    assert resolved is override
    assert kind == "override"


def test_callback_capabilities_cache_invalidates_on_list_change(monkeypatch):
    """The cache key includes (length, id-of-each-callback).  Mutating the
    callback list must produce a fresh capability snapshot."""
    monkeypatch.setattr(litellm, "callbacks", [])
    assert ProxyLogging._callback_capabilities().resolved_callbacks == ()

    class _OverridesPreCall(CustomLogger):
        async def async_pre_call_hook(self, *args, **kwargs):
            return kwargs.get("data")

    pre = _OverridesPreCall()
    monkeypatch.setattr(litellm, "callbacks", [pre])
    caps = ProxyLogging._callback_capabilities()
    assert caps.has_pre_call_override is True
    assert pre in caps.resolved_callbacks
