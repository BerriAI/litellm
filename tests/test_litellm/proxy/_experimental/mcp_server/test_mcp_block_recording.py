"""Seam test for the guardrail-block recording fix in
``litellm.proxy._experimental.mcp_server.server.call_mcp_tool``.

A pre-call MCP guardrail block *raises* into ``call_mcp_tool``'s
``except Exception``. The failure spend-log row (the one the Guardrails Monitor
counts) is written by ``proxy_logging_obj.post_call_failure_hook`` ->
get_logging_payload, which reads ``guardrail_information`` off the logging obj's
``standard_logging_object``. That object is only populated once
``failure_handler`` / ``async_failure_handler`` run. So the failure handlers MUST
run *before* post_call_failure_hook, otherwise the row persists with
``guardrail_information=None`` and the block is never counted (totalBlocked stays
0). This test locks that ordering in.

``call_mcp_tool`` is wrapped by ``@client`` (litellm.utils.client), which uses
``functools.wraps`` and therefore exposes the raw undecorated coroutine as
``__wrapped__``. The tests drive ``__wrapped__`` directly so the except-block
ordering is observed in isolation, without the wrapper's own post-raise logging
firing. This mirrors the original config-repo seam test, which stubbed ``@client``
to an identity decorator for the same reason; like the original, it does not
exercise the wrapper's dedup path.

``proxy_logging_obj`` is imported lazily inside the except block via
``from litellm.proxy.proxy_server import proxy_logging_obj``; the real
proxy_server module is heavy (boto3 etc.), so a fake module is injected into
``sys.modules`` to satisfy that lazy import without loading it.
"""

import asyncio
import contextlib
import sys
import types
from unittest import mock

from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server import server


class _RecordingLoggingObj:
    """Stands in for LiteLLMLoggingObj; records the failure-flush calls the fix
    makes so the test can assert they happen before post_call_failure_hook."""

    def __init__(self, order):
        self._order = order
        self.failure_calls = 0
        self.async_failure_calls = 0

    def has_run_logging(self, event_type):
        self._order.append(("has_run_logging", event_type))

    def failure_handler(self, *args, **kwargs):
        self.failure_calls += 1
        self._order.append("failure_handler")

    async def async_failure_handler(self, *args, **kwargs):
        self.async_failure_calls += 1
        self._order.append("async_failure_handler")


def _call_block(logging_obj, order):
    """Drive call_mcp_tool into its except path via arguments=None (raises
    HTTPException before any server-manager call) and return once it re-raises.

    A fake ``litellm.proxy.proxy_server`` module is injected so the except
    block's lazy ``from litellm.proxy.proxy_server import proxy_logging_obj``
    resolves to a recording stand-in instead of importing the real (heavy)
    module."""

    async def _record_post_call_failure_hook(**kwargs):
        order.append("post_call_failure_hook")

    proxy_logging_obj = mock.MagicMock()
    proxy_logging_obj.post_call_failure_hook.side_effect = _record_post_call_failure_hook

    fake_proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    fake_proxy_server.proxy_logging_obj = proxy_logging_obj

    with mock.patch.dict(sys.modules, {"litellm.proxy.proxy_server": fake_proxy_server}):
        with contextlib.suppress(HTTPException):
            asyncio.run(
                server.call_mcp_tool.__wrapped__(
                    name="t",
                    arguments=None,  # -> raise HTTPException(400) -> except Exception
                    user_api_key_auth=mock.MagicMock(),  # truthy: gate the hook
                    litellm_logging_obj=logging_obj,
                )
            )


def test_block_flushes_failure_before_post_call_failure_hook():
    order = []
    obj = _RecordingLoggingObj(order)
    _call_block(obj, order)

    assert "post_call_failure_hook" in order, order
    assert "failure_handler" in order, order
    assert "async_failure_handler" in order, order
    hook_at = order.index("post_call_failure_hook")
    assert order.index("failure_handler") < hook_at, order
    assert order.index("async_failure_handler") < hook_at, order


def test_block_flushes_each_handler_exactly_once():
    order = []
    obj = _RecordingLoggingObj(order)
    _call_block(obj, order)
    assert obj.failure_calls == 1, obj.failure_calls
    assert obj.async_failure_calls == 1, obj.async_failure_calls


def test_absent_logging_obj_still_calls_hook_and_skips_flush():
    # Guarded by `if litellm_logging_obj is not None`: without a logging obj the
    # failure handlers are skipped (no crash) but post_call_failure_hook still
    # fires. Byte-equivalent to stock behavior for that branch, so it passes on
    # baseline too; its value is as a mutation-killer for that guard.
    order = []
    _call_block(None, order)
    assert order == ["post_call_failure_hook"], order
