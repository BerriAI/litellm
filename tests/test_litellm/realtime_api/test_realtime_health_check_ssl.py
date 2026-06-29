"""
Regression test for realtime health-check SSL handling.

Issue: https://github.com/BerriAI/litellm/issues/31613

The ``websockets`` library raises "ssl argument is incompatible with a ws:// URI"
when an ``ssl=`` value is passed for a plain ``ws://`` URL. ``_realtime_health_check``
previously passed ``ssl=get_shared_realtime_ssl_context()`` unconditionally, which
broke health checks against OpenAI-compatible realtime servers exposed over plain
HTTP (e.g. a self-hosted vLLM instance with an ``http://`` api_base).

It must mirror ``OpenAIRealtime._get_ssl_config``: only ``wss://`` gets an SSL
context; ``ws://`` gets ``None``.
"""

from unittest.mock import patch

import pytest

from litellm.realtime_api.main import _realtime_health_check


class _FakeWSConnect:
    """Captures kwargs passed to websockets.connect and acts as an async context manager."""

    def __init__(self, store):
        self._store = store

    def __call__(self, url, **kwargs):
        self._store["url"] = url
        self._store["ssl"] = kwargs.get("ssl", "<<not passed>>")
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


async def _capture_health_check(api_base):
    store = {}
    with patch("websockets.connect", new=_FakeWSConnect(store)):
        await _realtime_health_check(
            model="some-model",
            custom_llm_provider="openai",
            api_key="dummy",
            api_base=api_base,
        )
    return store


@pytest.mark.asyncio
async def test_realtime_health_check_no_ssl_for_ws():
    """A ws:// URL (from an http:// api_base) must NOT receive an ssl= argument."""
    store = await _capture_health_check("http://localhost:8030/v1")
    assert store["url"].startswith("ws://")
    assert store["ssl"] is None


@pytest.mark.asyncio
async def test_realtime_health_check_keeps_ssl_for_wss():
    """A wss:// URL (from an https:// api_base) must keep its SSL context."""
    store = await _capture_health_check("https://api.openai.com/")
    assert store["url"].startswith("wss://")
    assert store["ssl"] is not None
    assert store["ssl"] != "<<not passed>>"
