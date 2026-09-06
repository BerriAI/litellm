"""Tests for guardrail-block recording in
``litellm.proxy._experimental.mcp_server.server.call_mcp_tool``.

A pre-call MCP guardrail block *raises* into ``call_mcp_tool``'s
``except Exception``. The failure spend-log row that the Guardrails Monitor's
"Total Blocked" counts is written by ``_ProxyDBLogger.async_post_call_failure_hook``
(reached via ``proxy_logging_obj.post_call_failure_hook``), which reads
``standard_logging_object`` off the request's logging obj -- and that only exists
once ``failure_handler`` / ``async_failure_handler`` have run. So the failure
handlers must run *before* ``post_call_failure_hook``, otherwise the row persists
with ``guardrail_information=None`` and the block is never counted. These tests
pin that ordering.

``call_mcp_tool`` is wrapped by ``@client`` (``litellm.utils.client``), which uses
``functools.wraps`` and therefore exposes the raw undecorated coroutine as
``__wrapped__``. The tests drive ``__wrapped__`` directly so the except-block
ordering is observed in isolation, without the wrapper's own post-raise logging
firing. Note that this means they do not exercise the wrapper's dedup path; that
dedup rests on ``should_run_logging("sync_failure")`` / ``("async_failure")``,
which has its own coverage in the logging tests.

``proxy_logging_obj`` is imported lazily inside the except block via
``from litellm.proxy.proxy_server import proxy_logging_obj``; the real
``proxy_server`` module is heavy, so a fake module is injected into ``sys.modules``
to satisfy that lazy import without loading it.
"""

import contextlib
import sys
import types
from unittest import mock

import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server import server


class _RecordingLoggingObj:
    """Stands in for ``LiteLLMLoggingObj``, recording the failure flush the fix
    makes so the test can assert it happens before ``post_call_failure_hook``."""

    def __init__(self, order: list) -> None:
        self._order = order
        self.failure_calls = 0
        self.async_failure_calls = 0

    def failure_handler(self, *_args, **_kwargs) -> None:
        self.failure_calls += 1
        self._order.append("failure_handler")

    async def async_failure_handler(self, *_args, **_kwargs) -> None:
        self.async_failure_calls += 1
        self._order.append("async_failure_handler")


async def _call_block(logging_obj, order: list, *, user_api_key_auth=mock.sentinel.auth):
    """Drive ``call_mcp_tool`` into its except path via ``arguments=None``, which
    raises ``HTTPException(400)`` before any server-manager call, and return once it
    re-raises."""

    async def _record_post_call_failure_hook(**_kwargs) -> None:
        order.append("post_call_failure_hook")

    proxy_logging_obj = mock.MagicMock()
    proxy_logging_obj.post_call_failure_hook.side_effect = _record_post_call_failure_hook

    fake_proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    fake_proxy_server.proxy_logging_obj = proxy_logging_obj  # pyright: ignore[reportAttributeAccessIssue]

    with mock.patch.dict(sys.modules, {"litellm.proxy.proxy_server": fake_proxy_server}):
        with contextlib.suppress(HTTPException):
            await server.call_mcp_tool.__wrapped__(
                name="t",
                arguments=None,
                user_api_key_auth=user_api_key_auth,
                litellm_logging_obj=logging_obj,
            )


@pytest.mark.asyncio
async def test_block_flushes_failure_before_post_call_failure_hook():
    order: list = []
    await _call_block(_RecordingLoggingObj(order), order)

    assert order == ["failure_handler", "async_failure_handler", "post_call_failure_hook"], order


@pytest.mark.asyncio
async def test_block_flushes_each_handler_exactly_once():
    """Each handler runs once, so the block yields exactly one counted row rather
    than double-counting on the shared logging obj."""
    order: list = []
    obj = _RecordingLoggingObj(order)
    await _call_block(obj, order)

    assert (obj.failure_calls, obj.async_failure_calls) == (1, 1)


@pytest.mark.asyncio
async def test_block_flushes_failure_for_anonymous_calls():
    """With no ``user_api_key_auth`` the failure handlers still run, so OTel and the
    other failure sinks see the block.

    ``post_call_failure_hook`` stays gated on auth, matching the pre-existing
    contract: SpendLogs rows are attributable billing/audit records and the
    downstream DB logger dereferences authenticated key, budget, and route data.
    Counting anonymous MCP blocks needs a counter that does not live in SpendLogs,
    which is a separate design change, not part of this fix.
    """
    order: list = []
    obj = _RecordingLoggingObj(order)
    await _call_block(obj, order, user_api_key_auth=None)

    assert order == ["failure_handler", "async_failure_handler"], order


@pytest.mark.asyncio
async def test_absent_logging_obj_still_calls_hook_and_skips_flush():
    """Without a logging obj the flush is skipped (no crash) but
    ``post_call_failure_hook`` still fires. Byte-equivalent to stock behavior for
    that branch; its value is as a mutation-killer for the ``is not None`` guard."""
    order: list = []
    await _call_block(None, order)

    assert order == ["post_call_failure_hook"], order
