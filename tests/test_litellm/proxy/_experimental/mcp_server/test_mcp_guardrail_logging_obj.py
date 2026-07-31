"""Seam test for the MCP guardrail logging-obj threading in
``litellm.proxy._experimental.mcp_server.mcp_server_manager``.

It proves the fix's crux: ``pre_call_tool_check`` and ``_create_during_hook_task``
seed the real ``litellm_logging_obj`` (and its shared ``metadata`` dict) onto the
synthetic dict handed to the guardrail hooks (``data["litellm_logging_obj"]`` /
``data["metadata"]``). That is exactly what the custom_guardrail bridge reads to
record the evaluation into the spend log the monitor counts.

The real ``MCPServerManager`` is used via ``__new__`` (skipping ``__init__``); the
three pre-call guard/validation helpers on the path are stubbed, and a fake
``proxy_logging_obj`` is dependency-injected to capture the data each hook
receives. ``_create_during_hook_task`` does lazy in-function imports of
``litellm.types.*`` (HiddenParams, MCPDuringCallRequestObject) at call time;
those resolve against the installed litellm package.
"""

import asyncio
import datetime
from unittest import mock

from litellm.proxy._experimental.mcp_server import mcp_server_manager as MOD

SENTINEL = mock.MagicMock()  # stands in for the real LiteLLMLoggingObj
SENTINEL.litellm_params = {"metadata": {}}


def _bare_manager():
    """An MCPServerManager instance without running __init__; stub the pre-call
    guard/validation helpers this test's path touches so it reaches the seed."""
    mgr = MOD.MCPServerManager.__new__(MOD.MCPServerManager)
    mgr.check_allowed_or_banned_tools = lambda name, server: True
    mgr.validate_allowed_params = lambda tool_name, arguments, server: None

    async def _ok(*a, **k):
        return None

    mgr.check_tool_permission_for_key_team = _ok
    return mgr


def _fake_proxy_logging(capture):
    """A proxy_logging_obj whose _convert_mcp_to_llm_format returns a fresh dict
    and whose pre_call_hook/during_call_hook capture the data they receive."""
    plo = mock.MagicMock()
    plo._create_mcp_request_object_from_kwargs.return_value = mock.MagicMock()
    plo._convert_mcp_to_llm_format.side_effect = lambda *a, **k: {}

    async def _pre_call_hook(*, user_api_key_dict, data, call_type):
        capture["pre"] = data
        return None

    async def _during_call_hook(*, user_api_key_dict, data, call_type):
        capture["during"] = data
        return None

    plo.pre_call_hook.side_effect = _pre_call_hook
    plo.during_call_hook.side_effect = _during_call_hook
    return plo


def test_pre_call_seeds_real_logging_obj():
    capture = {}
    mgr = _bare_manager()
    plo = _fake_proxy_logging(capture)
    server = mock.MagicMock()
    asyncio.run(
        mgr.pre_call_tool_check(
            name="t",
            arguments={},
            server_name="s",
            user_api_key_auth=None,
            proxy_logging_obj=plo,
            server=server,
            raw_headers={},
            litellm_logging_obj=SENTINEL,
        )
    )
    assert capture["pre"]["litellm_logging_obj"] is SENTINEL
    assert capture["pre"]["metadata"] is SENTINEL.litellm_params["metadata"]


def test_during_hook_seeds_real_logging_obj():
    capture = {}
    mgr = _bare_manager()
    plo = _fake_proxy_logging(capture)

    async def _run():
        task = mgr._create_during_hook_task(
            name="t",
            arguments={},
            server_name_from_prefix="s",
            user_api_key_auth=None,
            proxy_logging_obj=plo,
            start_time=datetime.datetime(2026, 7, 14),
            litellm_logging_obj=SENTINEL,
        )
        await task

    asyncio.run(_run())
    assert capture["during"]["litellm_logging_obj"] is SENTINEL
    assert capture["during"]["metadata"] is SENTINEL.litellm_params["metadata"]


def test_absent_logging_obj_defaults_to_none():
    # When the caller does not thread a logging obj, the seed is None. Byte
    # equivalent to stock behavior (key absent -> .get returns None).
    capture = {}
    mgr = _bare_manager()
    plo = _fake_proxy_logging(capture)
    asyncio.run(
        mgr.pre_call_tool_check(
            name="t",
            arguments={},
            server_name="s",
            user_api_key_auth=None,
            proxy_logging_obj=plo,
            server=mock.MagicMock(),
            raw_headers={},
        )
    )
    assert capture["pre"]["litellm_logging_obj"] is None
