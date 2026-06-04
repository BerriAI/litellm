"""Pin the ``ProxyLogging`` capability-probe family.

Covers ``_callback_capabilities`` (the cached deriver),
``has_post_call_response_headers_callbacks``, ``has_streaming_callbacks``,
``has_streaming_chunk_hook_overrides``, ``needs_iterator_wrap``,
``needs_per_chunk_streaming_hook``, ``has_during_call_guardrails``, and
``get_combined_callback_list``.
"""

from __future__ import annotations

from typing import Any

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.utils import ProxyLogging, _CallbackCapabilities


class _PlainLogger(CustomLogger):
    pass


class _OverridesResponseHeaders(CustomLogger):
    async def async_post_call_response_headers_hook(self, *args, **kwargs):  # type: ignore[override]
        return None


class _OverridesIterator(CustomLogger):
    async def async_post_call_streaming_iterator_hook(self, *args, **kwargs):  # type: ignore[override]
        return None


class _OverridesPerChunk(CustomLogger):
    async def async_post_call_streaming_hook(self, *args, **kwargs):  # type: ignore[override]
        return None


class _OverridesPreCall(CustomLogger):
    async def async_pre_call_hook(self, *args, **kwargs):  # type: ignore[override]
        return None


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


def test_callback_capabilities_with_no_callbacks_returns_defaults(mock_callbacks_disabled):
    caps = ProxyLogging._callback_capabilities()
    snapshot = {
        "headers": caps.has_post_call_response_headers,
        "iterator": caps.has_iterator_override,
        "chunk": caps.has_streaming_chunk_override,
        "guardrail": caps.has_guardrail,
        "pre_call": caps.has_pre_call_override,
        "callbacks": caps.resolved_callbacks,
        "overrides": caps.iterator_overrides,
    }
    assert snapshot == {
        "headers": False,
        "iterator": False,
        "chunk": False,
        "guardrail": False,
        "pre_call": False,
        "callbacks": (),
        "overrides": (),
    }


def test_callback_capabilities_detects_overrides(monkeypatch):
    cb1 = _OverridesResponseHeaders()
    cb2 = _OverridesIterator()
    cb3 = _OverridesPerChunk()
    cb4 = _OverridesPreCall()
    monkeypatch.setattr(litellm, "callbacks", [cb1, cb2, cb3, cb4])

    caps = ProxyLogging._callback_capabilities()
    snapshot = {
        "headers": caps.has_post_call_response_headers,
        "iterator": caps.has_iterator_override,
        "chunk": caps.has_streaming_chunk_override,
        "pre_call": caps.has_pre_call_override,
    }
    assert snapshot == {
        "headers": True,
        "iterator": True,
        "chunk": True,
        "pre_call": True,
    }


def test_callback_capabilities_caches_result(monkeypatch):
    cb = _OverridesResponseHeaders()
    monkeypatch.setattr(litellm, "callbacks", [cb])
    first = ProxyLogging._callback_capabilities()
    second = ProxyLogging._callback_capabilities()
    assert first is second


def test_callback_capabilities_invalidates_on_change(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [_OverridesResponseHeaders()])
    first = ProxyLogging._callback_capabilities()
    monkeypatch.setattr(litellm, "callbacks", [_OverridesIterator()])
    second = ProxyLogging._callback_capabilities()
    assert first is not second
    assert first.has_post_call_response_headers is True
    assert second.has_post_call_response_headers is False
    assert second.has_iterator_override is True


def test_callback_capabilities_callback_resolution_error_raises(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", ["unknown-string"])
    monkeypatch.setattr(
        litellm.litellm_core_utils.litellm_logging,
        "get_custom_logger_compatible_class",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    with pytest.raises(RuntimeError):
        ProxyLogging._callback_capabilities()


# ---------------------------------------------------------------------------
# Individual capability probes
# ---------------------------------------------------------------------------


def test_has_post_call_response_headers_callbacks_truth_table(monkeypatch, mock_callbacks_disabled):
    """One snapshot covering true + false + cache invalidation."""
    snapshot = {
        "empty_returns_false": ProxyLogging.has_post_call_response_headers_callbacks(),
    }
    monkeypatch.setattr(litellm, "callbacks", [_OverridesResponseHeaders()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["override_returns_true"] = ProxyLogging.has_post_call_response_headers_callbacks()
    monkeypatch.setattr(litellm, "callbacks", [_PlainLogger()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["plain_logger_false"] = ProxyLogging.has_post_call_response_headers_callbacks()
    assert snapshot == {
        "empty_returns_false": False,
        "override_returns_true": True,
        "plain_logger_false": False,
    }


def test_has_post_call_response_headers_callbacks_error_when_bad_callback(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", ["x"])
    monkeypatch.setattr(
        litellm.litellm_core_utils.litellm_logging,
        "get_custom_logger_compatible_class",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    with pytest.raises(RuntimeError):
        ProxyLogging.has_post_call_response_headers_callbacks()


def test_has_streaming_callbacks_truth_table(monkeypatch, mock_callbacks_disabled):
    snapshot = {
        "empty_false": ProxyLogging.has_streaming_callbacks(),
    }
    monkeypatch.setattr(litellm, "callbacks", [_OverridesIterator()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["iterator_override_true"] = ProxyLogging.has_streaming_callbacks()
    monkeypatch.setattr(litellm, "callbacks", [_OverridesPerChunk()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["per_chunk_override_true"] = ProxyLogging.has_streaming_callbacks()
    assert snapshot == {
        "empty_false": False,
        "iterator_override_true": True,
        "per_chunk_override_true": True,
    }


def test_has_streaming_callbacks_error_when_resolution_fails(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", ["x"])
    monkeypatch.setattr(
        litellm.litellm_core_utils.litellm_logging,
        "get_custom_logger_compatible_class",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("nope")),
    )
    with pytest.raises(ValueError):
        ProxyLogging.has_streaming_callbacks()


def test_has_streaming_chunk_hook_overrides_truth_table(monkeypatch, mock_callbacks_disabled):
    snapshot = {
        "empty_false": ProxyLogging.has_streaming_chunk_hook_overrides(),
    }
    monkeypatch.setattr(litellm, "callbacks", [_OverridesPerChunk()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["per_chunk_override_true"] = ProxyLogging.has_streaming_chunk_hook_overrides()
    monkeypatch.setattr(litellm, "callbacks", [_OverridesIterator()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["only_iterator_false"] = ProxyLogging.has_streaming_chunk_hook_overrides()
    assert snapshot == {
        "empty_false": False,
        "per_chunk_override_true": True,
        "only_iterator_false": False,
    }


def test_has_streaming_chunk_hook_overrides_error_raises(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", ["x"])
    monkeypatch.setattr(
        litellm.litellm_core_utils.litellm_logging,
        "get_custom_logger_compatible_class",
        lambda *a, **kw: (_ for _ in ()).throw(TypeError("nope")),
    )
    with pytest.raises(TypeError):
        ProxyLogging.has_streaming_chunk_hook_overrides()


def test_needs_iterator_wrap_truth_table(proxy_logging, monkeypatch, mock_callbacks_disabled):
    snapshot = {
        "empty_false": proxy_logging.needs_iterator_wrap(),
    }
    monkeypatch.setattr(litellm, "callbacks", [_OverridesIterator()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["with_iter_override_true"] = proxy_logging.needs_iterator_wrap()
    monkeypatch.setattr(litellm, "callbacks", [_OverridesPerChunk()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["only_per_chunk_false"] = proxy_logging.needs_iterator_wrap()
    assert snapshot == {
        "empty_false": False,
        "with_iter_override_true": True,
        "only_per_chunk_false": False,
    }


def test_needs_iterator_wrap_error_raises(proxy_logging, monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", ["x"])
    monkeypatch.setattr(
        litellm.litellm_core_utils.litellm_logging,
        "get_custom_logger_compatible_class",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("oops")),
    )
    with pytest.raises(RuntimeError):
        proxy_logging.needs_iterator_wrap()


def test_needs_per_chunk_streaming_hook_truth_table(proxy_logging, monkeypatch, mock_callbacks_disabled):
    snapshot = {
        "empty_false": proxy_logging.needs_per_chunk_streaming_hook(),
    }
    monkeypatch.setattr(litellm, "callbacks", [_OverridesPerChunk()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["per_chunk_override_true"] = proxy_logging.needs_per_chunk_streaming_hook()
    monkeypatch.setattr(litellm, "callbacks", [_OverridesIterator()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["only_iter_override_false"] = proxy_logging.needs_per_chunk_streaming_hook()
    assert snapshot == {
        "empty_false": False,
        "per_chunk_override_true": True,
        "only_iter_override_false": False,
    }


def test_needs_per_chunk_streaming_hook_error_raises(proxy_logging, monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", ["x"])
    monkeypatch.setattr(
        litellm.litellm_core_utils.litellm_logging,
        "get_custom_logger_compatible_class",
        lambda *a, **kw: (_ for _ in ()).throw(KeyError("oops")),
    )
    with pytest.raises(KeyError):
        proxy_logging.needs_per_chunk_streaming_hook()


def test_has_during_call_guardrails_truth_table(monkeypatch, mock_callbacks_disabled):
    from litellm.integrations.custom_guardrail import CustomGuardrail

    class _G(CustomGuardrail):
        def __init__(self):
            super().__init__(guardrail_name="g", event_hook="pre_call")

    snapshot = {
        "empty_false": ProxyLogging.has_during_call_guardrails(),
    }
    monkeypatch.setattr(litellm, "callbacks", [_G()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["with_guardrail_true"] = ProxyLogging.has_during_call_guardrails()
    monkeypatch.setattr(litellm, "callbacks", [_PlainLogger()])
    ProxyLogging._callback_capabilities_cache.clear()
    snapshot["only_plain_logger_false"] = ProxyLogging.has_during_call_guardrails()
    assert snapshot == {
        "empty_false": False,
        "with_guardrail_true": True,
        "only_plain_logger_false": False,
    }


def test_has_during_call_guardrails_resolution_error_raises(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", ["x"])
    monkeypatch.setattr(
        litellm.litellm_core_utils.litellm_logging,
        "get_custom_logger_compatible_class",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("oops")),
    )
    with pytest.raises(RuntimeError):
        ProxyLogging.has_during_call_guardrails()


# ---------------------------------------------------------------------------
# get_combined_callback_list
# ---------------------------------------------------------------------------


def test_get_combined_callback_list_matrix(proxy_logging):
    snapshot = {
        "merge_dedupes_shared": sorted(
            proxy_logging.get_combined_callback_list(
                dynamic_success_callbacks=["dyn-1", "shared"],
                global_callbacks=["glob-1", "shared"],
            )
        ),
        "none_dynamic_returns_global_copy": proxy_logging.get_combined_callback_list(
            dynamic_success_callbacks=None, global_callbacks=["a", "b", "c"]
        ),
        "empty_both": proxy_logging.get_combined_callback_list(
            dynamic_success_callbacks=[], global_callbacks=[]
        ),
    }
    assert snapshot == {
        "merge_dedupes_shared": ["dyn-1", "glob-1", "shared"],
        "none_dynamic_returns_global_copy": ["a", "b", "c"],
        "empty_both": [],
    }


def test_get_combined_callback_list_unhashable_dynamic_raises(proxy_logging):
    with pytest.raises(TypeError):
        proxy_logging.get_combined_callback_list(
            dynamic_success_callbacks=[{"unhashable": True}],
            global_callbacks=[],
        )
