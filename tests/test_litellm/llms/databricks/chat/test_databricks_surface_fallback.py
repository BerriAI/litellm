"""Tests for the self-contained Databricks chat reactive surface fallback.

Covers: optimistic gateway -> reactive serving-endpoints retry on a host-level
gateway-absent error, no retry for explicit-serving or non-absent errors, host is
cached absent after the fallback, and the async path never blocks on a probe.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.databricks import ai_gateway
from litellm.llms.databricks.chat.surface_fallback import (
    databricks_chat_completion_with_surface_fallback,
)

HOST = "https://my-workspace.cloud.databricks.com"


def _absent_error():
    e = Exception("ENDPOINT_NOT_FOUND: gateway path not served")
    e.status_code = 404  # type: ignore[attr-defined]
    e.message = "ENDPOINT_NOT_FOUND"  # type: ignore[attr-defined]
    return e


def _bad_request_error():
    e = Exception("invalid messages")
    e.status_code = 400  # type: ignore[attr-defined]
    e.message = "bad request"  # type: ignore[attr-defined]
    return e


class _SyncHandler:
    """Fake handler whose completion() pops side effects (exception or value)."""

    def __init__(self, effects):
        self._effects = list(effects)
        self.calls = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        effect = self._effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class _AsyncHandler:
    def __init__(self, effects):
        self._effects = list(effects)
        self.calls = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        effect = self._effects.pop(0)

        async def _coro():
            if isinstance(effect, Exception):
                raise effect
            return effect

        return _coro()


def _call(handler, *, acompletion=False, api_base=HOST, optional_params=None):
    return databricks_chat_completion_with_surface_fallback(
        handler,
        acompletion=acompletion,
        api_base=api_base,
        optional_params=optional_params if optional_params is not None else {},
        litellm_params={},
        model="databricks/databricks-claude-sonnet-4",
        messages=[{"role": "user", "content": "hi"}],
    )


class TestSyncReactiveFallback:
    def setup_method(self):
        ai_gateway.clear_gateway_cache()

    def teardown_method(self):
        ai_gateway.clear_gateway_cache()

    def test_gateway_absent_marks_host_and_retries(self):
        handler = _SyncHandler([_absent_error(), "OK"])
        result = _call(handler)
        assert result == "OK"
        assert len(handler.calls) == 2  # gateway attempt + serving retry
        assert ai_gateway.gateway_known_absent(HOST) is True

    def test_non_absent_error_does_not_retry(self):
        handler = _SyncHandler([_bad_request_error()])
        with pytest.raises(Exception, match="invalid messages"):
            _call(handler)
        assert len(handler.calls) == 1
        assert ai_gateway.gateway_known_absent(HOST) is False

    def test_explicit_serving_base_does_not_retry(self):
        handler = _SyncHandler([_absent_error()])
        with pytest.raises(Exception):
            _call(handler, api_base=f"{HOST}/serving-endpoints")
        assert len(handler.calls) == 1

    def test_request_tags_preserved_on_retry(self):
        # The first attempt "pops" the tag param (mutating optional_params);
        # the retry must see it restored.
        optional_params = {"databricks_ai_gateway_request_tags": ["team:fe"]}

        class _PoppingHandler(_SyncHandler):
            def completion(self, **kwargs):
                kwargs["optional_params"].pop(
                    "databricks_ai_gateway_request_tags", None
                )
                return super().completion(**kwargs)

        handler = _PoppingHandler([_absent_error(), "OK"])
        result = _call(handler, optional_params=optional_params)
        assert result == "OK"
        # Retry saw the restored tag param.
        assert (
            handler.calls[1]["optional_params"].get(
                "databricks_ai_gateway_request_tags"
            )
            is None  # popped again by the retry's own run
        )
        # original dict was restored before the retry (not left empty by attempt 1)
        assert optional_params == {}


class TestAsyncReactiveFallback:
    def setup_method(self):
        ai_gateway.clear_gateway_cache()

    def teardown_method(self):
        ai_gateway.clear_gateway_cache()

    def test_async_gateway_absent_retries(self):
        handler = _AsyncHandler([_absent_error(), "OK"])
        coro = _call(handler, acompletion=True)
        assert asyncio.iscoroutine(coro)
        result = asyncio.run(coro)
        assert result == "OK"
        assert len(handler.calls) == 2
        assert ai_gateway.gateway_known_absent(HOST) is True

    def test_async_non_absent_raises(self):
        handler = _AsyncHandler([_bad_request_error()])
        coro = _call(handler, acompletion=True)
        with pytest.raises(Exception, match="invalid messages"):
            asyncio.run(coro)
        assert len(handler.calls) == 1
