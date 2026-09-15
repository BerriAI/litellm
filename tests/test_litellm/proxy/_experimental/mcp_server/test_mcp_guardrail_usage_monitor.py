"""Tests for MCP guardrail evaluations reaching the Guardrails Monitor.

MCP tool calls run their guardrails against a throwaway LLM-shaped dict built by
``ProxyLogging._convert_mcp_to_llm_format``, not against the dict the tool call is
logged from. ``@log_guardrail_information`` therefore appends
``standard_logging_guardrail_information`` to that throwaway dict's metadata
bucket, where ``get_standard_logging_object_payload`` never sees it, so the
Guardrails Monitor reported zero evaluations and zero blocks for MCP traffic.

``pre_call_tool_check`` and ``_create_during_hook_task`` now take the request's
``litellm_logging_obj`` and bridge those records onto it. These tests pin both the
seeding (which unified guardrails consume off ``data["litellm_logging_obj"]``) and
the bridge (which native guardrails depend on), including on the block path.
"""

import asyncio
import datetime
from typing import Any
from unittest import mock

import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy._experimental.mcp_server import mcp_server_manager as MOD


class _FakeLoggingObj:
    """Minimal stand-in for ``LiteLLMLoggingObj``.

    ``_sync_guardrail_info_to_logging_obj`` reads exactly these two attributes,
    and the spend-log payload is built from ``litellm_params["metadata"]``, so a
    real ``Logging`` instance would add setup cost without adding coverage.
    """

    def __init__(self) -> None:
        self.litellm_params: dict[str, Any] = {"metadata": {}}
        self.model_call_details: dict[str, Any] = {"litellm_params": self.litellm_params}

    @property
    def recorded_guardrails(self) -> list:
        return self.litellm_params["metadata"].get("standard_logging_guardrail_information", [])


def _bare_manager() -> MOD.MCPServerManager:
    """An ``MCPServerManager`` without running ``__init__``.

    The authorization/validation helpers on the path are stubbed out so the test
    reaches the guardrail hooks; they have their own coverage elsewhere.
    """
    mgr = MOD.MCPServerManager.__new__(MOD.MCPServerManager)
    mgr.check_allowed_or_banned_tools = lambda name, server: True
    mgr.validate_allowed_params = lambda tool_name, arguments, server: None

    async def _ok(*_args, **_kwargs) -> None:
        return None

    mgr.check_tool_permission_for_key_team = _ok
    return mgr


def _fake_proxy_logging(capture: dict, *, guardrail_effect=None):
    """A ``proxy_logging_obj`` double whose hooks capture the data they receive.

    ``guardrail_effect`` stands in for a guardrail: it is handed the synthetic
    request dict so it can append a guardrail record (and optionally raise, the
    way a blocking guardrail does).
    """
    plo = mock.MagicMock()
    plo._create_mcp_request_object_from_kwargs.return_value = mock.MagicMock()
    # Mirror the real conversion's metadata bucket so a test can prove it survives.
    plo._convert_mcp_to_llm_format.side_effect = lambda *_a, **_k: {
        "metadata": {"headers": {"x-forwarded-for": "1.2.3.4"}}
    }

    async def _hook(*, user_api_key_dict, data, call_type) -> None:
        del user_api_key_dict  # captured shape is what matters, not the auth double
        capture["data"] = data
        capture["call_type"] = call_type
        if guardrail_effect is not None:
            guardrail_effect(data)

    plo.pre_call_hook.side_effect = _hook
    plo.during_call_hook.side_effect = _hook
    return plo


def _record_guardrail(status: str = "success"):
    """Write a guardrail record the way ``@log_guardrail_information`` does."""

    def _effect(data: dict) -> None:
        data.setdefault("metadata", {}).setdefault("standard_logging_guardrail_information", []).append(
            {"guardrail_name": "test-guardrail", "guardrail_status": status}
        )

    return _effect


def _blocking_guardrail():
    record = _record_guardrail(status="guardrail_intervened")

    def _effect(data: dict) -> None:
        record(data)
        raise GuardrailRaisedException(guardrail_name="test-guardrail", message="blocked")

    return _effect


async def _run_pre_call(mgr, plo, logging_obj) -> dict:
    return await mgr.pre_call_tool_check(
        name="t",
        arguments={},
        server_name="s",
        user_api_key_auth=None,
        proxy_logging_obj=plo,
        server=mock.MagicMock(),
        raw_headers={},
        litellm_logging_obj=logging_obj,
    )


@pytest.mark.asyncio
async def test_pre_call_seeds_request_logging_obj_for_unified_guardrails():
    """Unified guardrails read ``data["litellm_logging_obj"]`` and pass it into
    ``apply_guardrail``, whose ``@log_guardrail_information`` wrapper bridges the
    evaluation onto that logger itself. Drop the seed and that path records
    nothing."""
    capture: dict = {}
    logging_obj = _FakeLoggingObj()
    await _run_pre_call(_bare_manager(), _fake_proxy_logging(capture), logging_obj)

    assert capture["data"]["litellm_logging_obj"] is logging_obj


@pytest.mark.asyncio
async def test_pre_call_keeps_synthetic_request_headers_metadata():
    """The seed must not clobber the metadata bucket ``_convert_mcp_to_llm_format``
    builds: guardrails such as ``MCPJWTSigner`` read ``metadata["headers"]`` off
    it."""
    capture: dict = {}
    await _run_pre_call(_bare_manager(), _fake_proxy_logging(capture), _FakeLoggingObj())

    assert capture["data"]["metadata"]["headers"] == {"x-forwarded-for": "1.2.3.4"}


@pytest.mark.asyncio
async def test_pre_call_bridges_allowed_evaluation_onto_request_logger():
    """An allowed ``pre_mcp_call`` evaluation must land on the request logger, which
    is what the monitor's "Total Evaluations" counts."""
    capture: dict = {}
    logging_obj = _FakeLoggingObj()
    plo = _fake_proxy_logging(capture, guardrail_effect=_record_guardrail())

    await _run_pre_call(_bare_manager(), plo, logging_obj)

    assert logging_obj.recorded_guardrails == [{"guardrail_name": "test-guardrail", "guardrail_status": "success"}]


@pytest.mark.asyncio
async def test_pre_call_bridges_blocked_evaluation_before_reraising():
    """A block raises straight out of ``pre_call_tool_check``, and the failure
    spend-log row that "Total Blocked" counts is built from this logger further up
    the stack. So the record has to be attached before the exception leaves the
    frame -- hence the bridge lives in a ``finally``."""
    capture: dict = {}
    logging_obj = _FakeLoggingObj()
    plo = _fake_proxy_logging(capture, guardrail_effect=_blocking_guardrail())

    with pytest.raises(GuardrailRaisedException):
        await _run_pre_call(_bare_manager(), plo, logging_obj)

    assert logging_obj.recorded_guardrails == [
        {"guardrail_name": "test-guardrail", "guardrail_status": "guardrail_intervened"}
    ]


@pytest.mark.asyncio
async def test_pre_call_without_logging_obj_is_unchanged():
    """Callers that thread no logger are unaffected: the seed is an explicit
    ``None`` (which every consumer reads via ``.get``) and nothing is bridged.
    Guards against the bridge assuming a logger exists."""
    capture: dict = {}
    plo = _fake_proxy_logging(capture, guardrail_effect=_record_guardrail())
    mgr = _bare_manager()

    result = await mgr.pre_call_tool_check(
        name="t",
        arguments={},
        server_name="s",
        user_api_key_auth=None,
        proxy_logging_obj=plo,
        server=mock.MagicMock(),
        raw_headers={},
    )

    assert result == {}
    assert capture["data"]["litellm_logging_obj"] is None


@pytest.mark.asyncio
async def test_during_hook_seeds_and_bridges_onto_request_logger():
    """``during_mcp_call`` evaluations need the same treatment. The task is awaited
    before the tool call's success logging runs, so the record is serialized with
    that call."""
    capture: dict = {}
    logging_obj = _FakeLoggingObj()
    plo = _fake_proxy_logging(capture, guardrail_effect=_record_guardrail())

    await _bare_manager()._create_during_hook_task(
        name="t",
        arguments={},
        server_name_from_prefix="s",
        user_api_key_auth=None,
        proxy_logging_obj=plo,
        start_time=datetime.datetime(2026, 7, 14),
        litellm_logging_obj=logging_obj,
    )

    assert capture["data"]["litellm_logging_obj"] is logging_obj
    assert logging_obj.recorded_guardrails == [{"guardrail_name": "test-guardrail", "guardrail_status": "success"}]


@pytest.mark.asyncio
async def test_during_hook_bridges_even_when_hook_raises():
    """A during-call guardrail block must still be recorded before the task's
    exception propagates to the ``asyncio.gather`` in ``call_tool``."""
    capture: dict = {}
    logging_obj = _FakeLoggingObj()
    plo = _fake_proxy_logging(capture, guardrail_effect=_blocking_guardrail())

    task = _bare_manager()._create_during_hook_task(
        name="t",
        arguments={},
        server_name_from_prefix="s",
        user_api_key_auth=None,
        proxy_logging_obj=plo,
        start_time=datetime.datetime(2026, 7, 14),
        litellm_logging_obj=logging_obj,
    )
    with pytest.raises(GuardrailRaisedException):
        await task

    assert logging_obj.recorded_guardrails == [
        {"guardrail_name": "test-guardrail", "guardrail_status": "guardrail_intervened"}
    ]


@pytest.mark.asyncio
async def test_bridge_failure_does_not_mask_a_guardrail_block():
    """Recording is best-effort bookkeeping. If the bridge itself raises, the guardrail's
    block must still be what the caller sees, not a bookkeeping error.

    The bridge is forced to fail by making the logger's ``model_call_details`` raise, and
    the swallow is asserted (not just the surviving exception type) so the test cannot go
    vacuous if a refactor stops the bridge from touching that attribute.
    """
    capture: dict = {}
    plo = _fake_proxy_logging(capture, guardrail_effect=_blocking_guardrail())

    broken_logging_obj = mock.MagicMock()
    type(broken_logging_obj).model_call_details = mock.PropertyMock(side_effect=RuntimeError("boom"))

    with mock.patch.object(MOD.verbose_logger, "warning") as warn:
        with pytest.raises(GuardrailRaisedException):
            await _run_pre_call(_bare_manager(), plo, broken_logging_obj)

    assert warn.call_count == 1, "the bridge did not actually fail, so this test proves nothing"
    assert "boom" in str(warn.call_args)


@pytest.mark.asyncio
async def test_call_tool_threads_logging_obj_into_both_hooks():
    """``call_tool`` is the single entry point every MCP dispatch route funnels
    through, so it must hand the logger to both guardrail hook sites."""
    mgr = _bare_manager()
    logging_obj = _FakeLoggingObj()
    seen: dict = {}

    async def _fake_pre_call_tool_check(**kwargs):
        seen["pre_call"] = kwargs.get("litellm_logging_obj")
        return {}

    def _fake_during_hook_task(**kwargs):
        seen["during_call"] = kwargs.get("litellm_logging_obj")
        return asyncio.get_running_loop().create_future()

    mgr.pre_call_tool_check = _fake_pre_call_tool_check
    mgr._create_during_hook_task = _fake_during_hook_task
    mgr._resolve_mcp_server_for_tool_call = lambda server_name, name: mock.MagicMock(spec_path=None)
    mgr._resolve_oauth2_headers_for_tool_call = mock.AsyncMock(return_value=None)
    mgr._call_regular_mcp_tool = mock.AsyncMock(return_value=mock.MagicMock())

    with mock.patch.object(MOD, "_resolve_byok_mcp_auth_header", mock.AsyncMock(return_value=None)):
        await mgr.call_tool(
            server_name="s",
            name="t",
            arguments={},
            proxy_logging_obj=mock.MagicMock(),
            litellm_logging_obj=logging_obj,
        )

    assert seen == {"pre_call": logging_obj, "during_call": logging_obj}
