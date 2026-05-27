"""Regression tests: Langfuse trace sending fails with closed client.

Bug: ``LangFuseLogger`` held only the inner ``httpx.Client`` and not the
wrapping ``HTTPHandler``. The HTTPHandler was kept alive only by
``litellm.in_memory_llm_clients_cache``. When that cache entry expired the
HTTPHandler was garbage collected, its ``__del__`` called
``self.client.close()``, and Langfuse’s background flush thread surfaced
the now-closed client as
``RuntimeError: Cannot send a request, as the client has been closed.``

The fix stores a strong reference to the wrapping HTTPHandler on the
LangFuseLogger instance.
"""
from __future__ import annotations

import gc
import sys
import types

import httpx
import pytest


def _stub_langfuse_module() -> None:
    """Stub the ``langfuse`` package so ``LangFuseLogger.__init__`` can run
    without a real langfuse SDK install."""
    fake = types.ModuleType("langfuse")

    class _FakeLangfuse:
        def __init__(self, **kwargs):
            self._init_kwargs = kwargs

        def trace(self, **kwargs):  # pragma: no cover - not exercised here
            return None

    fake.Langfuse = _FakeLangfuse  # type: ignore[attr-defined]
    fake.client = types.SimpleNamespace()  # type: ignore[attr-defined]
    fake.version = types.SimpleNamespace(__version__="2.59.7")  # type: ignore[attr-defined]
    sys.modules["langfuse"] = fake
    sys.modules["langfuse.client"] = fake.client
    sys.modules["langfuse.version"] = fake.version


@pytest.fixture
def langfuse_env(monkeypatch):
    _stub_langfuse_module()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://example.invalid")
    yield


def _evict_httpx_client_cache() -> None:
    """Drop all entries from ``litellm.in_memory_llm_clients_cache``."""
    import litellm

    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is None:
        return
    for fn in ("flush_cache", "clear_cache"):
        try:
            getattr(cache, fn)()
        except Exception:
            pass
    for attr in ("cache_dict",):
        try:
            getattr(cache, attr).clear()
        except Exception:
            pass
    try:
        cache.in_memory_cache.cache_dict.clear()
    except Exception:
        pass


def test_langfuse_logger_keeps_http_handler_reference(langfuse_env):
    """LangFuseLogger must keep a strong reference to the wrapping HTTPHandler
    so its __del__ cannot close the underlying httpx.Client out from under the
    Langfuse SDK while this logger is alive."""
    from litellm.integrations.langfuse.langfuse import LangFuseLogger
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    logger = LangFuseLogger()
    handler = getattr(logger, "_langfuse_http_handler", None)
    assert isinstance(handler, HTTPHandler), (
        "LangFuseLogger must hold a strong reference to the wrapping "
        "HTTPHandler (attribute `_langfuse_http_handler`)."
    )
    assert handler.client is logger.langfuse_client


def test_langfuse_client_survives_cache_eviction_and_gc(langfuse_env):
    """End-to-end lifecycle test: evict the litellm HTTPHandler cache and
    force GC. The httpx.Client Langfuse holds must remain open and usable."""
    from litellm.integrations.langfuse.langfuse import LangFuseLogger

    logger = LangFuseLogger()
    client = logger.langfuse_client
    assert client.is_closed is False, "httpx.Client should start open"

    _evict_httpx_client_cache()
    gc.collect()
    gc.collect()

    assert client.is_closed is False, (
        "httpx.Client was closed after the litellm HTTPHandler cache was "
        "evicted and garbage collected. LangFuseLogger must hold a strong "
        "reference to the wrapping HTTPHandler."
    )

    req = httpx.Request("POST", "http://127.0.0.1:1/")
    try:
        client.send(req)
    except RuntimeError as e:
        pytest.fail(
            f"closed-client RuntimeError after cache eviction + GC: {e}"
        )
    except httpx.HTTPError:
        # Any transport-layer error is fine; we only guard against the
        # closed-client RuntimeError.
        pass
