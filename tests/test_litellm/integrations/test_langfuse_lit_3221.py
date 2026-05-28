"""
Regression tests for LIT-3221: Langfuse trace sending fails with closed client.

Root cause: LangFuseLogger.__init__ extracted only the inner httpx.Client from
the cached HTTPHandler and discarded the wrapping HTTPHandler reference. When
the in-memory LLM-clients cache expired (1-hour TTL), the HTTPHandler became
eligible for GC; its __del__ closed the same httpx.Client that the Langfuse
SDK background flush thread was still using, surfacing as:

    RuntimeError: Cannot send a request, as the client has been closed.

Fix: store the HTTPHandler on `self._http_handler` so it (and its underlying
httpx.Client) outlive the cache eviction.
"""

import gc
import os
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.integrations.langfuse import langfuse as langfuse_module
from litellm.integrations.langfuse.langfuse import LangFuseLogger


@pytest.fixture(autouse=True)
def _langfuse_env():
    env = {
        "LANGFUSE_PUBLIC_KEY": "pk-lf-lit-3221",
        "LANGFUSE_SECRET_KEY": "sk-lf-lit-3221",
        "LANGFUSE_HOST": "http://127.0.0.1:9999",
    }
    # Track original litellm counter to restore
    orig = litellm.initialized_langfuse_clients
    with patch.dict(os.environ, env, clear=False):
        yield
    litellm.initialized_langfuse_clients = orig


def _safe_init_logger():
    """Build a LangFuseLogger without contacting Langfuse on init."""
    mock_lf_client = MagicMock()
    mock_lf_client.client = MagicMock()
    mock_lf_client.client.projects.get.return_value.data = [MagicMock(id="proj-1")]
    with patch.object(
        LangFuseLogger,
        "safe_init_langfuse_client",
        return_value=mock_lf_client,
    ):
        return LangFuseLogger()


def test_langfuse_logger_holds_http_handler_reference():
    """
    Regression for LIT-3221: LangFuseLogger MUST keep a strong reference to the
    HTTPHandler so the cached httpx.Client cannot be closed by GC after the
    LLM-clients cache TTL elapses.
    """
    lf = _safe_init_logger()
    assert hasattr(lf, "_http_handler"), (
        "LangFuseLogger must retain a reference to the HTTPHandler "
        "(LIT-3221) — without this attribute, the underlying httpx.Client "
        "is closed by HTTPHandler.__del__ after the cache TTL evicts it."
    )
    assert lf._http_handler is not None
    # The exposed langfuse_client must be the same httpx.Client held by the
    # HTTPHandler — sharing the connection pool with the rest of the proxy.
    assert lf.langfuse_client is lf._http_handler.client


def test_langfuse_httpx_client_survives_cache_eviction_and_gc():
    """
    End-to-end regression: after evicting the HTTPHandler from the in-memory
    LLM-clients cache AND forcing GC, the httpx.Client held by the LangFuseLogger
    must still be usable (i.e. not closed).
    """
    lf = _safe_init_logger()
    client = lf.langfuse_client
    assert client.is_closed is False, "fresh client must not be closed"

    # Evict every cached HTTPHandler entry — same effect as TTL expiry.
    cache = litellm.in_memory_llm_clients_cache
    assert hasattr(cache, "cache_dict"), "cache shape changed — update test"
    for k in list(cache.cache_dict.keys()):
        cache.cache_dict.pop(k, None)
    gc.collect()
    gc.collect()

    assert client.is_closed is False, (
        "httpx.Client was closed after cache eviction + GC — the "
        "LangFuseLogger lost its reference to the HTTPHandler (LIT-3221)."
    )


def test_langfuse_mock_mode_does_not_set_http_handler():
    """
    The mock-mode path must not call _get_httpx_client. We assert by checking
    that _http_handler is not set when langfuse-mock mode is active.
    """
    with patch.object(langfuse_module, "should_use_langfuse_mock", return_value=True),          patch.object(
             langfuse_module,
             "create_mock_langfuse_client",
             return_value=MagicMock(),
         ),          patch.object(
             LangFuseLogger,
             "safe_init_langfuse_client",
             return_value=MagicMock(client=MagicMock(projects=MagicMock(get=MagicMock(return_value=MagicMock(data=[MagicMock(id="m")]))))),
         ):
        lf = LangFuseLogger()
    assert lf.is_mock_mode is True
    assert getattr(lf, "_http_handler", None) is None
