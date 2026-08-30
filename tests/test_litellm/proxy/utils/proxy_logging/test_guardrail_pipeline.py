"""Pin ProxyLogging guardrail pipeline helpers.

Covers ``_should_use_guardrail_load_balancing``, ``_execute_guardrail_hook``,
``_execute_guardrail_with_load_balancing``, ``_process_guardrail_callback``,
``_process_prompt_template``, ``_process_guardrail_metadata``,
``_maybe_execute_pipelines``, ``_handle_pipeline_result``,
``_run_guardrail_with_metrics``, ``_emit_guardrail_metrics``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.exceptions import SensitiveDataRouteException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import add_guardrail_to_applied_guardrails_header
from litellm.proxy.utils import ProxyLogging, _raise_for_streaming_post_call_pipelines
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.types.guardrails import BlockedWord, ContentFilterAction, GuardrailEventHooks
from litellm.types.proxy.policy_engine.pipeline_types import (
    GuardrailPipeline,
    PipelineStep,
)


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


# ---------------------------------------------------------------------------
# _should_use_guardrail_load_balancing
# ---------------------------------------------------------------------------


def test_should_use_guardrail_load_balancing_truth_table(proxy_logging):
    snapshot = {}
    router = MagicMock()
    router.guardrail_list = [{"guardrail_name": "g1"}, {"guardrail_name": "g1"}]
    with patch("litellm.proxy.proxy_server.llm_router", router):
        snapshot["multiple_deployments"] = proxy_logging._should_use_guardrail_load_balancing("g1")
    router.guardrail_list = [{"guardrail_name": "g1"}]
    with patch("litellm.proxy.proxy_server.llm_router", router):
        snapshot["single_deployment"] = proxy_logging._should_use_guardrail_load_balancing("g1")
    with patch("litellm.proxy.proxy_server.llm_router", None):
        snapshot["no_router"] = proxy_logging._should_use_guardrail_load_balancing("g1")
    router.guardrail_list = [{"guardrail_name": "other"}, {"guardrail_name": "other"}]
    with patch("litellm.proxy.proxy_server.llm_router", router):
        snapshot["unmatched_name"] = proxy_logging._should_use_guardrail_load_balancing("g1")
    assert snapshot == {
        "multiple_deployments": True,
        "single_deployment": False,
        "no_router": False,
        "unmatched_name": False,
    }


def test_should_use_guardrail_load_balancing_error_on_bad_guardrail_list(proxy_logging):
    router = MagicMock()
    router.guardrail_list = "not a list"
    with patch("litellm.proxy.proxy_server.llm_router", router):
        with pytest.raises((TypeError, AttributeError)):
            proxy_logging._should_use_guardrail_load_balancing("g1")


# ---------------------------------------------------------------------------
# _execute_guardrail_hook
# ---------------------------------------------------------------------------


def _make_guardrail():
    cb = MagicMock(spec=CustomGuardrail)
    cb.__class__ = CustomGuardrail
    cb.guardrail_name = "g"
    cb.event_hook = GuardrailEventHooks.pre_call
    cb.use_native_during_call_hook = False
    cb.async_pre_call_hook = AsyncMock(return_value={"a": 1, "b": 2, "c": 3})
    cb.async_moderation_hook = AsyncMock(return_value={"x": 1, "y": 2, "z": 3})
    cb.async_post_call_success_hook = AsyncMock(return_value={"p": 1, "q": 2, "r": 3})
    return cb


@pytest.mark.asyncio
async def test_execute_guardrail_hook_pre_call(proxy_logging, make_user_api_key_auth):
    cb = _make_guardrail()
    out = await proxy_logging._execute_guardrail_hook(
        callback=cb,
        hook_type="pre_call",
        data={"model": "m"},
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
    )
    assert out == {"a": 1, "b": 2, "c": 3}


@pytest.mark.asyncio
async def test_execute_guardrail_hook_during_call(proxy_logging, make_user_api_key_auth):
    cb = _make_guardrail()
    out = await proxy_logging._execute_guardrail_hook(
        callback=cb,
        hook_type="during_call",
        data={"model": "m"},
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
    )
    assert out == {"x": 1, "y": 2, "z": 3}


@pytest.mark.asyncio
async def test_execute_guardrail_hook_post_call(proxy_logging, make_user_api_key_auth):
    cb = _make_guardrail()
    out = await proxy_logging._execute_guardrail_hook(
        callback=cb,
        hook_type="post_call",
        data={"model": "m"},
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
        response={"original": True},
    )
    assert out == {"p": 1, "q": 2, "r": 3}


@pytest.mark.asyncio
async def test_execute_guardrail_hook_unknown_hook_type_raises(proxy_logging, make_user_api_key_auth):
    cb = _make_guardrail()
    with pytest.raises(ValueError, match="Unknown hook_type"):
        await proxy_logging._execute_guardrail_hook(
            callback=cb,
            hook_type="weird",  # type: ignore[arg-type]
            data={},
            user_api_key_dict=make_user_api_key_auth(),
            call_type="completion",
        )


# ---------------------------------------------------------------------------
# _execute_guardrail_with_load_balancing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_guardrail_with_load_balancing_routes_through_router(
    proxy_logging, make_user_api_key_auth
):
    cb = _make_guardrail()
    router = MagicMock()
    router.get_available_guardrail = MagicMock(return_value={"callback": cb})
    with patch("litellm.proxy.proxy_server.llm_router", router):
        out = await proxy_logging._execute_guardrail_with_load_balancing(
            guardrail_name="g",
            hook_type="pre_call",
            data={"model": "m"},
            user_api_key_dict=make_user_api_key_auth(),
            call_type="completion",
        )
    assert out == {"a": 1, "b": 2, "c": 3}


@pytest.mark.asyncio
async def test_execute_guardrail_with_load_balancing_router_none_raises(
    proxy_logging, make_user_api_key_auth
):
    with patch("litellm.proxy.proxy_server.llm_router", None):
        with pytest.raises(ValueError, match="Router not initialized"):
            await proxy_logging._execute_guardrail_with_load_balancing(
                guardrail_name="g",
                hook_type="pre_call",
                data={},
                user_api_key_dict=make_user_api_key_auth(),
                call_type="completion",
            )


@pytest.mark.asyncio
async def test_execute_guardrail_with_load_balancing_no_callback_raises(
    proxy_logging, make_user_api_key_auth
):
    router = MagicMock()
    router.get_available_guardrail = MagicMock(return_value={"callback": None})
    with patch("litellm.proxy.proxy_server.llm_router", router):
        with pytest.raises(ValueError, match="No callback found"):
            await proxy_logging._execute_guardrail_with_load_balancing(
                guardrail_name="g",
                hook_type="pre_call",
                data={},
                user_api_key_dict=make_user_api_key_auth(),
                call_type="completion",
            )


# ---------------------------------------------------------------------------
# _process_guardrail_callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_guardrail_callback_skipped_when_should_run_false(
    proxy_logging, make_user_api_key_auth
):
    cb = _make_guardrail()
    cb.should_run_guardrail = MagicMock(return_value=False)
    out = await proxy_logging._process_guardrail_callback(
        callback=cb,
        data={"model": "m"},
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
        event_type=GuardrailEventHooks.pre_call,
    )
    assert out is None


@pytest.mark.asyncio
async def test_process_guardrail_callback_returns_data_on_success(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    cb = _make_guardrail()
    cb.should_run_guardrail = MagicMock(return_value=True)
    proxy_logging._should_use_guardrail_load_balancing = MagicMock(return_value=False)
    out = await proxy_logging._process_guardrail_callback(
        callback=cb,
        data={"model": "m", "messages": [{"role": "user"}], "temperature": 0.1},
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
        event_type=GuardrailEventHooks.pre_call,
    )
    assert out == {"a": 1, "b": 2, "c": 3}


@pytest.mark.asyncio
async def test_process_guardrail_callback_enriches_and_reraises_http_exception(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    cb = _make_guardrail()
    cb.should_run_guardrail = MagicMock(return_value=True)
    detail = {"error": "blocked"}
    cb.async_pre_call_hook = AsyncMock(side_effect=HTTPException(status_code=400, detail=detail))
    cb.event_hook = "pre_call"
    proxy_logging._should_use_guardrail_load_balancing = MagicMock(return_value=False)

    with pytest.raises(HTTPException):
        await proxy_logging._process_guardrail_callback(
            callback=cb,
            data={"model": "m"},
            user_api_key_dict=make_user_api_key_auth(),
            call_type="completion",
            event_type=GuardrailEventHooks.pre_call,
        )
    assert detail["guardrail_name"] == "g"


# ---------------------------------------------------------------------------
# _process_guardrail_metadata
# ---------------------------------------------------------------------------


def test_process_guardrail_metadata_calls_header_helper(proxy_logging, monkeypatch):
    calls: List[Dict[str, Any]] = []

    def fake_add(request_data, guardrail_name):
        calls.append({"data": request_data, "name": guardrail_name})

    from litellm.proxy.common_utils import callback_utils

    monkeypatch.setattr(callback_utils, "add_guardrail_to_applied_guardrails_header", fake_add)
    data = {"metadata": {"guardrails": ["g1", "g2"]}}
    proxy_logging._process_guardrail_metadata(data)
    snapshot = {
        "call_count": len(calls),
        "first_name": calls[0]["name"],
        "second_name": calls[1]["name"],
        "data_passed_is_input": all(c["data"] is data for c in calls),
    }
    assert snapshot == {
        "call_count": 2,
        "first_name": "g1",
        "second_name": "g2",
        "data_passed_is_input": True,
    }


def test_process_guardrail_metadata_skips_already_applied(proxy_logging, monkeypatch):
    calls: List[str] = []

    def fake_add(request_data, guardrail_name):
        calls.append(guardrail_name)

    from litellm.proxy.common_utils import callback_utils

    monkeypatch.setattr(callback_utils, "add_guardrail_to_applied_guardrails_header", fake_add)
    data = {"metadata": {"guardrails": ["g1", "g2"], "applied_guardrails": ["g1"]}}
    proxy_logging._process_guardrail_metadata(data)
    assert calls == ["g2"]


def test_process_guardrail_metadata_no_metadata_is_noop(proxy_logging, monkeypatch):
    from litellm.proxy.common_utils import callback_utils

    monkeypatch.setattr(
        callback_utils,
        "add_guardrail_to_applied_guardrails_header",
        MagicMock(side_effect=AssertionError("should not be called")),
    )
    proxy_logging._process_guardrail_metadata({})


def test_process_guardrail_metadata_invalid_data_raises(proxy_logging):
    with pytest.raises(AttributeError):
        proxy_logging._process_guardrail_metadata(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _maybe_execute_pipelines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_execute_pipelines_no_pipelines_returns_data(proxy_logging, make_user_api_key_auth):
    data = {"messages": [{"role": "user"}], "model": "m", "temperature": 0.1}
    out, replacement = await proxy_logging._maybe_execute_pipelines(
        data=data,
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
        event_hook="pre_call",
    )
    assert out == {"messages": [{"role": "user"}], "model": "m", "temperature": 0.1}
    assert replacement is None


@pytest.mark.asyncio
async def test_maybe_execute_pipelines_skips_pipelines_with_other_mode(proxy_logging, make_user_api_key_auth, monkeypatch):
    pipeline = MagicMock()
    pipeline.mode = "post_call"  # not pre_call
    data = {"metadata": {"_guardrail_pipelines": [("p1", pipeline)]}, "model": "m", "messages": []}
    executed = MagicMock()
    monkeypatch.setattr(
        "litellm.proxy.policy_engine.pipeline_executor.PipelineExecutor.execute_steps", executed
    )
    out, replacement = await proxy_logging._maybe_execute_pipelines(
        data=data,
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
        event_hook="pre_call",
    )
    executed.assert_not_called()
    assert out is data
    assert replacement is None


@pytest.mark.parametrize(
    ("policy_state_key", "caller_metadata_key", "call_type"),
    [
        ("litellm_metadata", "metadata", "anthropic_messages"),
        ("metadata", "litellm_metadata", "completion"),
    ],
)
@pytest.mark.asyncio
async def test_maybe_execute_pipelines_finds_policy_state_when_caller_sends_own_metadata(
    proxy_logging, make_user_api_key_auth, monkeypatch, policy_state_key, caller_metadata_key, call_type
):
    """The route picks the bucket the policy engine writes to (``litellm_metadata`` on
    /v1/messages, ``metadata`` on chat completions), and the caller can populate the other
    one, e.g. Claude Code sending ``metadata.user_id``. The pipeline must still run and block."""

    class BlockingGuardrail(CustomGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            raise HTTPException(status_code=400, detail={"error": "blocked by pipeline"})

    monkeypatch.setattr(litellm, "callbacks", [BlockingGuardrail(guardrail_name="gr-1")])
    pipeline = GuardrailPipeline(mode="pre_call", steps=[PipelineStep(guardrail="gr-1", on_fail="block")])
    data = {
        caller_metadata_key: {"user_id": "user_abc"},
        policy_state_key: {"_guardrail_pipelines": [("policy-1", pipeline)]},
        "messages": [],
        "model": "m",
    }

    with pytest.raises(HTTPException) as exc_info:
        await proxy_logging._maybe_execute_pipelines(
            data=data,
            user_api_key_dict=make_user_api_key_auth(),
            call_type=call_type,
            event_hook="pre_call",
        )
    assert exc_info.value.detail["error"] == "blocked by pipeline"
    assert exc_info.value.detail["guardrail_name"] == "gr-1"


@pytest.mark.asyncio
async def test_maybe_execute_pipelines_blocks_on_block_terminal_action_raises(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    pipeline = MagicMock()
    pipeline.mode = "pre_call"
    pipeline.steps = []
    fake_result = MagicMock()
    fake_result.terminal_action = "block"
    fake_result.step_results = []
    fake_result.original_exception = None
    data = {"metadata": {"_guardrail_pipelines": [("policy-1", pipeline)]}, "messages": [], "model": "m"}

    async def fake_execute_steps(**kwargs):
        return fake_result

    monkeypatch.setattr(
        "litellm.proxy.policy_engine.pipeline_executor.PipelineExecutor.execute_steps",
        fake_execute_steps,
    )
    with pytest.raises(HTTPException):
        await proxy_logging._maybe_execute_pipelines(
            data=data,
            user_api_key_dict=make_user_api_key_auth(),
            call_type="completion",
            event_hook="pre_call",
        )


@pytest.mark.asyncio
async def test_maybe_execute_pipelines_reraises_original_guardrail_exception(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A policy-wrapped guardrail block must surface the guardrail's own
    exception verbatim, identical to the direct-attachment path."""
    pipeline = MagicMock()
    pipeline.mode = "pre_call"
    pipeline.steps = []
    original = HTTPException(
        status_code=400,
        detail={"error": "Violated OpenAI moderation policy", "moderation_result": {"x": 1}},
    )
    fake_result = MagicMock()
    fake_result.terminal_action = "block"
    fake_result.step_results = []
    fake_result.original_exception = original
    data = {"metadata": {"_guardrail_pipelines": [("policy-1", pipeline)]}, "messages": [], "model": "m"}

    async def fake_execute_steps(**kwargs):
        return fake_result

    monkeypatch.setattr(
        "litellm.proxy.policy_engine.pipeline_executor.PipelineExecutor.execute_steps",
        fake_execute_steps,
    )
    with pytest.raises(HTTPException) as info:
        await proxy_logging._maybe_execute_pipelines(
            data=data,
            user_api_key_dict=make_user_api_key_auth(),
            call_type="completion",
            event_hook="pre_call",
        )
    assert info.value is original


# ---------------------------------------------------------------------------
# _handle_pipeline_result
# ---------------------------------------------------------------------------


def test_handle_pipeline_result_allow_with_modifications():
    data = {"a": 1}
    result = MagicMock()
    result.terminal_action = "allow"
    result.modified_data = {"b": 2, "c": 3}
    out = ProxyLogging._handle_pipeline_result(result=result, data=data, policy_name="p")
    assert out == {"a": 1, "b": 2, "c": 3}


def test_handle_pipeline_result_block_falls_back_to_generic_when_no_exception():
    result = MagicMock()
    result.terminal_action = "block"
    result.step_results = []
    result.original_exception = None
    with pytest.raises(HTTPException) as info:
        ProxyLogging._handle_pipeline_result(result=result, data={"model": "m"}, policy_name="p")
    detail = info.value.detail
    snapshot = {
        "is_dict": isinstance(detail, dict),
        "error_type": detail["error"]["type"],
        "policy": detail["error"]["pipeline_context"]["policy"],
    }
    assert snapshot == {
        "is_dict": True,
        "error_type": "guardrail_pipeline_error",
        "policy": "p",
    }


def test_handle_pipeline_result_block_reraises_original_guardrail_exception():
    """The policy path must re-raise the guardrail's own exception untouched,
    not wrap it in a generic ``guardrail_pipeline_error``; this is what makes
    the response and trace span identical to the direct-attachment path."""
    original = HTTPException(
        status_code=400,
        detail={
            "error": "Violated OpenAI moderation policy",
            "moderation_result": {"violated_categories": ["harassment"]},
        },
    )
    result = MagicMock()
    result.terminal_action = "block"
    result.step_results = []
    result.original_exception = original
    with pytest.raises(HTTPException) as info:
        ProxyLogging._handle_pipeline_result(result=result, data={"model": "m"}, policy_name="p")
    assert info.value is original
    assert info.value.detail == {
        "error": "Violated OpenAI moderation policy",
        "moderation_result": {"violated_categories": ["harassment"]},
    }


def test_handle_pipeline_result_block_enriches_with_guardrail_name_and_mode():
    """The re-raised exception must gain the blocking guardrail's name and mode,
    matching the enrichment the direct-attachment path applies."""
    cb = _make_guardrail()  # guardrail_name="g", event_hook=pre_call
    original = HTTPException(status_code=400, detail={"error": "blocked"})
    result = MagicMock()
    result.terminal_action = "block"
    result.step_results = [MagicMock(guardrail_name="g")]
    result.original_exception = original

    saved = litellm.callbacks
    litellm.callbacks = [cb]
    try:
        with pytest.raises(HTTPException) as info:
            ProxyLogging._handle_pipeline_result(
                result=result, data={"model": "m"}, policy_name="p"
            )
    finally:
        litellm.callbacks = saved

    assert info.value is original
    assert info.value.detail["guardrail_name"] == "g"
    assert info.value.detail["guardrail_mode"] == GuardrailEventHooks.pre_call


def test_handle_pipeline_result_block_does_not_reraise_sensitive_data_route():
    """A step configured to block must enforce the block even when the guardrail
    raised a reroute exception; re-raising it verbatim would route the request to
    an alternate model instead of blocking, bypassing the configured policy."""
    original = SensitiveDataRouteException(
        route_to_model="on-prem-model",
        session_id="sess-1",
        guardrail_name="pii-router",
    )
    result = MagicMock()
    result.terminal_action = "block"
    result.step_results = [MagicMock(guardrail_name="pii-router")]
    result.original_exception = original
    with pytest.raises(HTTPException) as info:
        ProxyLogging._handle_pipeline_result(result=result, data={"model": "m"}, policy_name="p")
    assert info.value.status_code == 400
    assert info.value.detail["error"]["type"] == "guardrail_pipeline_error"


def test_handle_pipeline_result_block_does_not_reraise_modify_response():
    """A step configured to block must enforce the block even when the guardrail
    raised a passthrough/modify-response exception; re-raising it verbatim would
    return the guardrail's synthetic response instead of blocking."""
    original = ModifyResponseException(
        message="redacted",
        model="m",
        request_data={"model": "m"},
        guardrail_name="masker",
    )
    result = MagicMock()
    result.terminal_action = "block"
    result.step_results = [MagicMock(guardrail_name="masker")]
    result.original_exception = original
    with pytest.raises(HTTPException) as info:
        ProxyLogging._handle_pipeline_result(result=result, data={"model": "m"}, policy_name="p")
    assert info.value.status_code == 400
    assert info.value.detail["error"]["type"] == "guardrail_pipeline_error"


def test_handle_pipeline_result_modify_response_raises_modify_exception():
    result = MagicMock()
    result.terminal_action = "modify_response"
    result.modify_response_message = "filtered"
    with pytest.raises(ModifyResponseException):
        ProxyLogging._handle_pipeline_result(result=result, data={"model": "m"}, policy_name="p")


def test_handle_pipeline_result_unknown_action_returns_data():
    data = {"a": 1, "b": 2, "c": 3}
    result = MagicMock()
    result.terminal_action = "something_else"
    assert ProxyLogging._handle_pipeline_result(result=result, data=data, policy_name="p") is data


# ---------------------------------------------------------------------------
# _run_guardrail_with_metrics
# ---------------------------------------------------------------------------


def _prometheus_callback() -> MagicMock:
    """Stand-in PrometheusLogger that records ``_record_guardrail_metrics`` calls.

    ``MagicMock(spec=PrometheusLogger)`` passes the ``isinstance`` check inside
    ``_emit_guardrail_metrics`` while letting us capture the recorded labels.
    """
    return MagicMock(spec=PrometheusLogger)


@pytest.mark.asyncio
async def test_run_guardrail_with_metrics_passes_result_and_records_success(monkeypatch):
    async def task():
        return {"a": 1, "b": 2, "c": 3}

    prom = _prometheus_callback()
    monkeypatch.setattr(litellm, "callbacks", [prom])

    out = await ProxyLogging._run_guardrail_with_metrics(
        callback=MagicMock(guardrail_name="g"), coro=task(), hook_type="during_call"
    )

    assert out == {"a": 1, "b": 2, "c": 3}
    recorded = prom._record_guardrail_metrics.call_args.kwargs
    assert recorded["guardrail_name"] == "g"
    assert recorded["status"] == "success"
    assert recorded["error_type"] is None
    assert recorded["hook_type"] == "during_call"
    assert recorded["latency_seconds"] >= 0


@pytest.mark.asyncio
async def test_run_guardrail_with_metrics_records_error_and_enriches(monkeypatch):
    detail = {"error": "blocked"}

    async def task():
        raise HTTPException(status_code=400, detail=detail)

    cb = MagicMock()
    cb.guardrail_name = "presidio"
    cb.event_hook = "pre_call"
    prom = _prometheus_callback()
    monkeypatch.setattr(litellm, "callbacks", [prom])

    with pytest.raises(HTTPException):
        await ProxyLogging._run_guardrail_with_metrics(
            callback=cb, coro=task(), hook_type="post_call"
        )

    assert detail["guardrail_name"] == "presidio"
    recorded = prom._record_guardrail_metrics.call_args.kwargs
    assert recorded["status"] == "error"
    assert recorded["error_type"] == "HTTPException"
    assert recorded["hook_type"] == "post_call"


# ---------------------------------------------------------------------------
# during_call / post_call phases emit the latency metric (LIT-3999 regression)
# ---------------------------------------------------------------------------


def _moderation_guardrail() -> MagicMock:
    cb = MagicMock(spec=CustomGuardrail)
    cb.__class__ = CustomGuardrail
    cb.guardrail_name = "g"
    cb.event_hook = GuardrailEventHooks.during_call
    cb.use_native_during_call_hook = False
    cb.should_run_guardrail = MagicMock(return_value=True)
    cb.async_moderation_hook = AsyncMock(return_value=None)
    cb.async_post_call_success_hook = AsyncMock(return_value=None)
    cb.run_in_parallel = False
    return cb


@pytest.mark.asyncio
async def test_during_call_hook_records_latency_metric(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    cb = _moderation_guardrail()
    prom = _prometheus_callback()
    monkeypatch.setattr(litellm, "callbacks", [prom, cb])

    await proxy_logging.during_call_hook(
        data={"model": "m"},
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
    )

    cb.async_moderation_hook.assert_awaited_once()
    recorded = prom._record_guardrail_metrics.call_args.kwargs
    assert recorded["hook_type"] == "during_call"
    assert recorded["guardrail_name"] == "g"
    assert recorded["status"] == "success"


@pytest.mark.asyncio
async def test_post_call_success_hook_records_latency_metric(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    cb = _moderation_guardrail()
    prom = _prometheus_callback()
    monkeypatch.setattr(litellm, "callbacks", [prom, cb])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)

    await proxy_logging.post_call_success_hook(
        data={"model": "m"},
        response=litellm.ModelResponse(),
        user_api_key_dict=make_user_api_key_auth(),
    )

    cb.async_post_call_success_hook.assert_awaited_once()
    recorded = prom._record_guardrail_metrics.call_args.kwargs
    assert recorded["hook_type"] == "post_call"
    assert recorded["guardrail_name"] == "g"
    assert recorded["status"] == "success"


# ---------------------------------------------------------------------------
# _process_prompt_template
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_prompt_template_no_op_when_no_prompt_spec(proxy_logging, monkeypatch):
    from litellm.proxy.prompts import prompt_registry

    monkeypatch.setattr(
        prompt_registry.IN_MEMORY_PROMPT_REGISTRY, "get_prompt_callback_by_id", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        prompt_registry.IN_MEMORY_PROMPT_REGISTRY, "get_prompt_by_id", lambda *a, **kw: None
    )
    data: Dict[str, Any] = {"messages": [{"role": "user"}], "model": "m", "temperature": 0.1}
    await proxy_logging._process_prompt_template(
        data=data,
        litellm_logging_obj=MagicMock(),
        prompt_id="some-id",
        prompt_version=1,
        call_type="completion",
    )
    assert data == {"messages": [{"role": "user"}], "model": "m", "temperature": 0.1}


@pytest.mark.asyncio
async def test_process_prompt_template_applies_when_spec_resolves(proxy_logging, monkeypatch):
    from litellm.proxy.prompts import prompt_registry

    custom_logger = MagicMock()
    prompt_spec = MagicMock()
    prompt_spec.litellm_params = MagicMock(prompt_id="resolved-id")

    monkeypatch.setattr(
        prompt_registry.IN_MEMORY_PROMPT_REGISTRY,
        "get_prompt_callback_by_id",
        lambda *a, **kw: custom_logger,
    )
    monkeypatch.setattr(
        prompt_registry.IN_MEMORY_PROMPT_REGISTRY, "get_prompt_by_id", lambda *a, **kw: prompt_spec
    )

    logging_obj = MagicMock()
    logging_obj.async_get_chat_completion_prompt = AsyncMock(
        return_value=(
            "model-out",
            [{"role": "user", "content": "rendered"}],
            {"temperature": 0.5, "top_p": 1},
        )
    )
    data: Dict[str, Any] = {
        "messages": [{"role": "user", "content": "orig"}],
        "model": "m",
        "prompt_id": "x",
    }
    await proxy_logging._process_prompt_template(
        data=data,
        litellm_logging_obj=logging_obj,
        prompt_id="x",
        prompt_version=None,
        call_type="completion",
    )
    snapshot = {
        "model": data["model"],
        "messages": data["messages"],
        "temperature": data["temperature"],
        "top_p": data["top_p"],
    }
    assert snapshot == {
        "model": "model-out",
        "messages": [{"role": "user", "content": "rendered"}],
        "temperature": 0.5,
        "top_p": 1,
    }


@pytest.mark.asyncio
async def test_process_prompt_template_async_get_prompt_error_raises(proxy_logging, monkeypatch):
    from litellm.proxy.prompts import prompt_registry

    custom_logger = MagicMock()
    prompt_spec = MagicMock()
    prompt_spec.litellm_params = MagicMock(prompt_id="x")
    monkeypatch.setattr(
        prompt_registry.IN_MEMORY_PROMPT_REGISTRY,
        "get_prompt_callback_by_id",
        lambda *a, **kw: custom_logger,
    )
    monkeypatch.setattr(
        prompt_registry.IN_MEMORY_PROMPT_REGISTRY, "get_prompt_by_id", lambda *a, **kw: prompt_spec
    )
    logging_obj = MagicMock()
    logging_obj.async_get_chat_completion_prompt = AsyncMock(side_effect=RuntimeError("bad prompt"))
    with pytest.raises(RuntimeError):
        await proxy_logging._process_prompt_template(
            data={"messages": [], "model": "m", "prompt_id": "x"},
            litellm_logging_obj=logging_obj,
            prompt_id="x",
            prompt_version=None,
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_process_prompt_template_aresponses_swaps_model_and_merges_input(proxy_logging, monkeypatch):
    from litellm.proxy.prompts import prompt_registry

    custom_logger = MagicMock()
    prompt_spec = MagicMock()
    prompt_spec.litellm_params = MagicMock(prompt_id="resolved-id")
    monkeypatch.setattr(
        prompt_registry.IN_MEMORY_PROMPT_REGISTRY,
        "get_prompt_callback_by_id",
        lambda *a, **kw: custom_logger,
    )
    monkeypatch.setattr(
        prompt_registry.IN_MEMORY_PROMPT_REGISTRY, "get_prompt_by_id", lambda *a, **kw: prompt_spec
    )

    logging_obj = MagicMock()
    logging_obj.async_get_chat_completion_prompt = AsyncMock(
        return_value=(
            "gpt-4o-mini",
            [
                {"role": "user", "content": "You are a pirate."},
                {"role": "user", "content": "Who are you?"},
            ],
            {},
        )
    )
    data: dict[str, object] = {"input": "Who are you?", "model": "anthropic-haiku-4-5", "prompt_id": "x"}
    await proxy_logging._process_prompt_template(
        data=data,
        litellm_logging_obj=logging_obj,
        prompt_id="x",
        prompt_version=None,
        call_type="aresponses",
    )
    assert data["model"] == "gpt-4o-mini"
    assert data["input"] == [
        {"role": "user", "content": "You are a pirate."},
        {"role": "user", "content": "Who are you?"},
    ]
    assert "messages" not in data
    assert "prompt_id" not in data
    hook_kwargs = logging_obj.async_get_chat_completion_prompt.await_args.kwargs
    assert hook_kwargs["messages"] == [{"role": "user", "content": "Who are you?"}]
    assert hook_kwargs["prompt_spec"] is prompt_spec


# ---------------------------------------------------------------------------
# post_call pipeline execution (LIT-6410)
# ---------------------------------------------------------------------------


def _post_call_pipeline_data(
    guardrail: str = "gr-post", step: PipelineStep | None = None, **extra: Any
) -> Dict[str, Any]:
    pipeline = GuardrailPipeline(
        mode="post_call",
        steps=[step or PipelineStep(guardrail=guardrail, on_pass="allow", on_fail="block")],
    )
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {
            "_guardrail_pipelines": [("response-governance", pipeline)],
            "_pipeline_managed_guardrails": {guardrail},
        },
        **extra,
    }


@pytest.mark.asyncio
async def test_post_call_success_hook_runs_post_call_pipeline_and_reraises_block(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {}

    class OutputBlockingGuardrail(CustomGuardrail):
        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            seen["response"] = response
            raise HTTPException(status_code=400, detail={"error": "output blocked"})

    monkeypatch.setattr(
        litellm,
        "callbacks",
        [OutputBlockingGuardrail(guardrail_name="gr-post", event_hook=GuardrailEventHooks.post_call, default_on=False)],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data()
    response = litellm.ModelResponse()

    with pytest.raises(HTTPException) as info:
        await proxy_logging.post_call_success_hook(
            data=data, response=response, user_api_key_dict=make_user_api_key_auth()
        )

    assert info.value.detail["error"] == "output blocked"
    assert seen["response"] is response


@pytest.mark.asyncio
async def test_post_call_pipeline_pass_runs_once_and_leaves_request_data_untouched(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {"count": 0}

    class RecordingGuardrail(CustomGuardrail):
        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            seen["count"] += 1
            seen["response"] = response
            return None

    monkeypatch.setattr(
        litellm,
        "callbacks",
        [RecordingGuardrail(guardrail_name="gr-post", event_hook=GuardrailEventHooks.post_call, default_on=False)],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data()
    response = litellm.ModelResponse()

    out = await proxy_logging.post_call_success_hook(
        data=data, response=response, user_api_key_dict=make_user_api_key_auth()
    )

    assert out is response
    assert seen["response"] is response
    assert seen["count"] == 1
    assert "response" not in data
    assert "guardrails" not in data["metadata"]


@pytest.mark.asyncio
async def test_post_call_pipeline_managed_default_on_guardrail_runs_exactly_once(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {"count": 0}

    class CountingGuardrail(CustomGuardrail):
        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            seen["count"] += 1
            return None

    monkeypatch.setattr(
        litellm,
        "callbacks",
        [CountingGuardrail(guardrail_name="gr-post", event_hook=GuardrailEventHooks.post_call, default_on=True)],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data()

    await proxy_logging.post_call_success_hook(
        data=data, response=litellm.ModelResponse(), user_api_key_dict=make_user_api_key_auth()
    )

    assert seen["count"] == 1


@pytest.mark.asyncio
async def test_post_call_hook_still_runs_guardrail_managed_only_by_pre_call_pipeline(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {"count": 0}

    class DualStageGuardrail(CustomGuardrail):
        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            seen["count"] += 1
            return None

    pre_call_pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[PipelineStep(guardrail="gr-dual", on_pass="allow", on_fail="block")],
    )
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [DualStageGuardrail(guardrail_name="gr-dual", event_hook=["pre_call", "post_call"], default_on=True)],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {
            "_guardrail_pipelines": [("request-governance", pre_call_pipeline)],
            "_pipeline_managed_guardrails": {"gr-dual"},
        },
    }

    await proxy_logging.post_call_success_hook(
        data=data, response=litellm.ModelResponse(), user_api_key_dict=make_user_api_key_auth()
    )

    assert seen["count"] == 1


@pytest.mark.asyncio
async def test_pre_call_hook_still_runs_guardrail_managed_only_by_post_call_pipeline(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {"count": 0}

    class DualStageGuardrail(CustomGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            seen["count"] += 1
            return data

    post_call_pipeline = GuardrailPipeline(
        mode="post_call",
        steps=[PipelineStep(guardrail="gr-dual", on_pass="allow", on_fail="block")],
    )
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [DualStageGuardrail(guardrail_name="gr-dual", event_hook=["pre_call", "post_call"], default_on=True)],
    )
    data = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {
            "_guardrail_pipelines": [("response-governance", post_call_pipeline)],
            "_pipeline_managed_guardrails": {"gr-dual"},
        },
    }

    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(), data=data, call_type="completion"
    )

    assert seen["count"] == 1


@pytest.mark.asyncio
async def test_post_call_pipeline_replacement_response_reaches_caller(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    masked = litellm.ModelResponse()

    class MaskingGuardrail(CustomGuardrail):
        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            return masked

    monkeypatch.setattr(
        litellm,
        "callbacks",
        [MaskingGuardrail(guardrail_name="gr-post", event_hook=GuardrailEventHooks.post_call, default_on=False)],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data()

    out = await proxy_logging.post_call_success_hook(
        data=data, response=litellm.ModelResponse(), user_api_key_dict=make_user_api_key_auth()
    )

    assert out is masked
    assert "response" not in data


@pytest.mark.asyncio
async def test_post_call_pipeline_replacement_chains_to_next_step_without_pass_data(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    masked = litellm.ModelResponse()
    seen: Dict[str, Any] = {}

    class MaskingGuardrail(CustomGuardrail):
        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            return masked

    class RecordingGuardrail(CustomGuardrail):
        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            seen["response"] = response
            return None

    pipeline = GuardrailPipeline(
        mode="post_call",
        steps=[
            PipelineStep(guardrail="gr-mask", on_pass="next", on_fail="block"),
            PipelineStep(guardrail="gr-audit", on_pass="allow", on_fail="block"),
        ],
    )
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [
            MaskingGuardrail(guardrail_name="gr-mask", event_hook=GuardrailEventHooks.post_call, default_on=False),
            RecordingGuardrail(guardrail_name="gr-audit", event_hook=GuardrailEventHooks.post_call, default_on=False),
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {
            "_guardrail_pipelines": [("response-governance", pipeline)],
            "_pipeline_managed_guardrails": {"gr-mask", "gr-audit"},
        },
    }

    out = await proxy_logging.post_call_success_hook(
        data=data, response=litellm.ModelResponse(), user_api_key_dict=make_user_api_key_auth()
    )

    assert out is masked
    assert seen["response"] is masked


def test_handle_pipeline_result_modify_response_carries_original_response():
    result = MagicMock()
    result.terminal_action = "modify_response"
    result.modify_response_message = "filtered"
    response = litellm.ModelResponse()

    with pytest.raises(ModifyResponseException) as info:
        ProxyLogging._handle_pipeline_result(
            result=result, data={"model": "m"}, policy_name="p", original_response=response
        )

    assert info.value.original_response is response


def test_handle_pipeline_result_allow_on_post_call_keeps_metadata_writes_only():
    data = {"a": 1, "metadata": {"guardrails": ["other"]}}
    result = MagicMock()
    result.terminal_action = "allow"
    result.modified_data = {
        "a": 2,
        "metadata": {"guardrails": ["other"], "applied_guardrails": ["gr-post"]},
        "response": object(),
    }

    out = ProxyLogging._handle_pipeline_result(
        result=result, data=data, policy_name="p", original_response=litellm.ModelResponse()
    )

    assert out is data
    assert data["a"] == 1
    assert "response" not in data
    assert data["metadata"] == {"guardrails": ["other"], "applied_guardrails": ["gr-post"]}


@pytest.mark.asyncio
async def test_post_call_pipeline_guardrail_metadata_writes_reach_request_data(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    class HeaderWritingGuardrail(CustomGuardrail):
        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name="gr-post")
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response={"verdict": "pass"},
                request_data=data,
                guardrail_status="success",
            )
            return None

    monkeypatch.setattr(
        litellm,
        "callbacks",
        [HeaderWritingGuardrail(guardrail_name="gr-post", event_hook=GuardrailEventHooks.post_call, default_on=False)],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data()

    await proxy_logging.post_call_success_hook(
        data=data, response=litellm.ModelResponse(), user_api_key_dict=make_user_api_key_auth()
    )

    assert data["metadata"]["applied_guardrails"] == ["gr-post"]
    slg_entries = data["metadata"]["standard_logging_guardrail_information"]
    assert len(slg_entries) == 1
    assert slg_entries[0]["guardrail_name"] == "gr-post"


@pytest.mark.asyncio
async def test_post_call_pipeline_managed_parallel_guardrail_runs_exactly_once(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {"count": 0}

    class CountingGuardrail(CustomGuardrail):
        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            seen["count"] += 1
            return None

    monkeypatch.setattr(
        litellm,
        "callbacks",
        [
            CountingGuardrail(
                guardrail_name="gr-post",
                event_hook=GuardrailEventHooks.post_call,
                default_on=True,
                run_in_parallel=True,
            )
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data()

    await proxy_logging.post_call_success_hook(
        data=data, response=litellm.ModelResponse(), user_api_key_dict=make_user_api_key_auth()
    )

    assert seen["count"] == 1


@pytest.mark.asyncio
async def test_pre_call_pipeline_managed_parallel_guardrail_runs_exactly_once(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {"count": 0}

    class CountingGuardrail(CustomGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            seen["count"] += 1
            return data

    pre_call_pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[PipelineStep(guardrail="gr-pre", on_pass="allow", on_fail="block")],
    )
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [
            CountingGuardrail(
                guardrail_name="gr-pre",
                event_hook=GuardrailEventHooks.pre_call,
                default_on=True,
                run_in_parallel=True,
            )
        ],
    )
    data = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {
            "_guardrail_pipelines": [("request-governance", pre_call_pipeline)],
            "_pipeline_managed_guardrails": {"gr-pre"},
        },
    }

    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(), data=data, call_type="completion"
    )

    assert seen["count"] == 1


@pytest.mark.asyncio
async def test_pre_call_hook_rejects_streaming_request_with_post_call_pipeline(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [])
    data = _post_call_pipeline_data(stream=True)

    with pytest.raises(HTTPException) as info:
        await proxy_logging.pre_call_hook(
            user_api_key_dict=make_user_api_key_auth(),
            data=data,
            call_type="completion",
            guardrails_only=True,
        )

    assert info.value.status_code == 400
    assert info.value.detail["error"]["policies"] == ("response-governance",)
    assert info.value.detail["error"]["guardrails"] == ("gr-post",)
    assert "stream=false" in info.value.detail["error"]["message"]


@pytest.mark.asyncio
async def test_pre_call_hook_rejects_background_request_with_post_call_pipeline(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [])
    data = _post_call_pipeline_data(background=True)

    with pytest.raises(HTTPException) as info:
        await proxy_logging.pre_call_hook(
            user_api_key_dict=make_user_api_key_auth(),
            data=data,
            call_type="aresponses",
            guardrails_only=True,
        )

    assert info.value.status_code == 400
    assert info.value.detail["error"]["policies"] == ("response-governance",)
    assert "background=false" in info.value.detail["error"]["message"]


def test_raise_for_streaming_post_call_pipelines_ignores_non_streaming_and_pre_call(make_user_api_key_auth):
    post_call = GuardrailPipeline(mode="post_call", steps=[PipelineStep(guardrail="g", on_fail="block")])
    pre_call = GuardrailPipeline(mode="pre_call", steps=[PipelineStep(guardrail="g", on_fail="block")])
    auth = make_user_api_key_auth(request_route="/custom/stream")

    assert (
        _raise_for_streaming_post_call_pipelines(
            {"stream": False, "metadata": {"_guardrail_pipelines": [("p", post_call)]}}, auth
        )
        is None
    )
    assert (
        _raise_for_streaming_post_call_pipelines(
            {"background": False, "metadata": {"_guardrail_pipelines": [("p", post_call)]}}, auth
        )
        is None
    )
    assert (
        _raise_for_streaming_post_call_pipelines({"metadata": {"_guardrail_pipelines": [("p", post_call)]}}, auth)
        is None
    )
    assert (
        _raise_for_streaming_post_call_pipelines(
            {"stream": True, "metadata": {"_guardrail_pipelines": [("p", pre_call)]}}, auth
        )
        is None
    )
    assert _raise_for_streaming_post_call_pipelines({"stream": True}, auth) is None
    assert _raise_for_streaming_post_call_pipelines({"background": True}, auth) is None


# ---------------------------------------------------------------------------
# post_call pipelines on streaming responses
# ---------------------------------------------------------------------------


def _unified_stream_guardrail(seen: Dict[str, Any], block: bool = False) -> CustomGuardrail:
    class UnifiedStreamGuardrail(CustomGuardrail):
        async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
            seen["count"] = seen.get("count", 0) + 1
            seen["input_type"] = input_type
            if block:
                raise HTTPException(status_code=400, detail={"error": "output blocked"})
            return inputs

    return UnifiedStreamGuardrail(guardrail_name="gr-post", event_hook=GuardrailEventHooks.post_call, default_on=False)


def _stream_chunks() -> List[Any]:
    return [
        litellm.ModelResponseStream(choices=[{"index": 0, "delta": {"content": "hello "}, "finish_reason": None}]),
        litellm.ModelResponseStream(choices=[{"index": 0, "delta": {"content": "world"}, "finish_reason": "stop"}]),
    ]


async def _async_chunk_iter(chunks: List[Any]):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("request_route", [None, "/v1/chat/completions"])
async def test_pre_call_hook_allows_streaming_when_pipeline_guardrail_supports_unified(
    proxy_logging, make_user_api_key_auth, monkeypatch, request_route
):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(litellm, "callbacks", [_unified_stream_guardrail(seen)])
    data = _post_call_pipeline_data(stream=True)

    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(request_route=request_route),
        data=data,
        call_type="completion",
        guardrails_only=True,
    )

    assert out is not None
    assert out.get("stream") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("native_lifecycle", [False, True])
async def test_pre_call_hook_rejects_streaming_when_pipeline_guardrail_lacks_unified_support(
    proxy_logging, make_user_api_key_auth, monkeypatch, native_lifecycle
):
    if native_lifecycle:

        class NativeOnlyGuardrail(CustomGuardrail):
            use_native_lifecycle_hooks = True

            async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
                return inputs

    else:

        class NativeOnlyGuardrail(CustomGuardrail):
            pass

    monkeypatch.setattr(
        litellm,
        "callbacks",
        [NativeOnlyGuardrail(guardrail_name="gr-post", event_hook=GuardrailEventHooks.post_call, default_on=False)],
    )
    data = _post_call_pipeline_data(stream=True)

    with pytest.raises(HTTPException) as info:
        await proxy_logging.pre_call_hook(
            user_api_key_dict=make_user_api_key_auth(),
            data=data,
            call_type="completion",
            guardrails_only=True,
        )

    assert info.value.status_code == 400
    assert info.value.detail["error"]["guardrails"] == ("gr-post",)
    assert "apply_guardrail" in info.value.detail["error"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rewrite_attribute, value",
    [
        ("mask_response_content", True),
        ("streaming_transform_mode", "incremental_diff"),
        ("guardrail_config", {"streaming_transform_mode": "incremental_diff"}),
    ],
)
async def test_pre_call_hook_allows_streaming_when_pipeline_guardrail_rewrites_streamed_content(
    proxy_logging, make_user_api_key_auth, monkeypatch, rewrite_attribute, value
):
    seen: Dict[str, Any] = {}
    guardrail = _unified_stream_guardrail(seen)
    setattr(guardrail, rewrite_attribute, value)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    data = _post_call_pipeline_data(stream=True)

    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(request_route="/v1/chat/completions"),
        data=data,
        call_type="completion",
        guardrails_only=True,
    )

    assert out is not None
    assert out.get("stream") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [ContentFilterAction.MASK, ContentFilterAction.BLOCK])
async def test_pre_call_hook_allows_streaming_when_content_filter_step_masks_or_blocks(
    proxy_logging, make_user_api_key_auth, monkeypatch, action
):
    guardrail = ContentFilterGuardrail(
        guardrail_name="gr-post",
        event_hook=GuardrailEventHooks.post_call,
        blocked_words=[BlockedWord(keyword="persimmon", action=action)],
    )
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    data = _post_call_pipeline_data(stream=True)
    user_api_key_dict = make_user_api_key_auth(request_route="/v1/chat/completions")

    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=user_api_key_dict, data=data, call_type="completion", guardrails_only=True
    )

    assert out is not None and out.get("stream") is True


@pytest.mark.asyncio
async def test_pre_call_hook_allows_streaming_when_content_filter_category_masks(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    guardrail = ContentFilterGuardrail(
        guardrail_name="gr-post",
        event_hook=GuardrailEventHooks.post_call,
        categories=[{"category": "bias_gender", "enabled": True, "action": "MASK"}],
    )
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    data = _post_call_pipeline_data(stream=True)
    user_api_key_dict = make_user_api_key_auth(request_route="/v1/chat/completions")

    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=user_api_key_dict, data=data, call_type="completion", guardrails_only=True
    )

    assert out is not None and out.get("stream") is True


@pytest.mark.asyncio
async def test_pre_call_hook_rejects_streaming_when_route_has_no_guardrail_translation(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(litellm, "callbacks", [_unified_stream_guardrail(seen)])
    data = _post_call_pipeline_data(stream=True)

    with pytest.raises(HTTPException) as info:
        await proxy_logging.pre_call_hook(
            user_api_key_dict=make_user_api_key_auth(request_route="/custom/stream"),
            data=data,
            call_type="completion",
            guardrails_only=True,
        )

    assert info.value.status_code == 400
    assert info.value.detail["error"]["policies"] == ("response-governance",)
    assert "/custom/stream" in info.value.detail["error"]["message"]
    assert seen.get("count") is None


@pytest.mark.asyncio
async def test_streaming_iterator_hook_pipeline_allow_releases_buffered_chunks(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(litellm, "callbacks", [_unified_stream_guardrail(seen)])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data(stream=True)
    chunks = _stream_chunks()

    delivered = [
        item
        async for item in proxy_logging.async_post_call_streaming_iterator_hook(
            user_api_key_dict=make_user_api_key_auth(request_route="/v1/chat/completions"),
            response=_async_chunk_iter(chunks),
            request_data=data,
        )
    ]

    assert [id(item) for item in delivered] == [id(chunk) for chunk in chunks]
    assert seen["count"] == 1
    assert seen["input_type"] == "response"


@pytest.mark.asyncio
async def test_streaming_iterator_hook_pipeline_block_withholds_all_chunks(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(litellm, "callbacks", [_unified_stream_guardrail(seen, block=True)])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data(stream=True)
    delivered: List[Any] = []

    async def _drain() -> None:
        async for item in proxy_logging.async_post_call_streaming_iterator_hook(
            user_api_key_dict=make_user_api_key_auth(request_route="/v1/chat/completions"),
            response=_async_chunk_iter(_stream_chunks()),
            request_data=data,
        ):
            delivered.append(item)

    with pytest.raises(HTTPException) as info:
        await _drain()

    assert delivered == []
    assert info.value.status_code == 400
    assert "output blocked" in str(info.value.detail)


def _rewriting_stream_guardrail(transform: Callable[[Dict[str, Any]], Dict[str, Any]]) -> CustomGuardrail:
    class RewritingStreamGuardrail(CustomGuardrail):
        async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
            return {**inputs, **transform(inputs)}

    return RewritingStreamGuardrail(guardrail_name="gr-post", event_hook=GuardrailEventHooks.post_call, default_on=False)


def _tool_call_stream_chunks() -> List[Any]:
    tool_call = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"ssn": "123"}'},
    }
    return [
        litellm.ModelResponseStream(
            choices=[{"index": 0, "delta": {"tool_calls": [tool_call]}, "finish_reason": None}]
        ),
        litellm.ModelResponseStream(choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
    ]


def _echoed_tool_call_dicts(arguments: str) -> List[Dict[str, Any]]:
    return [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": arguments}}]


@pytest.mark.asyncio
@pytest.mark.parametrize("on_fail, on_error", [("block", None), ("next", "next")])
async def test_streaming_iterator_hook_pipeline_withholds_runtime_tool_call_rewrite(
    proxy_logging, make_user_api_key_auth, monkeypatch, on_fail, on_error
):
    transform = lambda inputs: {"tool_calls": _echoed_tool_call_dicts('{"ssn": "[MASKED]"}')}  # noqa: E731
    monkeypatch.setattr(litellm, "callbacks", [_rewriting_stream_guardrail(transform)])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    step = PipelineStep(guardrail="gr-post", on_pass="allow", on_fail=on_fail, on_error=on_error)
    data = _post_call_pipeline_data(step=step, stream=True)
    delivered: List[Any] = []

    async def _drain() -> None:
        async for item in proxy_logging.async_post_call_streaming_iterator_hook(
            user_api_key_dict=make_user_api_key_auth(request_route="/v1/chat/completions"),
            response=_async_chunk_iter(_tool_call_stream_chunks()),
            request_data=data,
        ):
            delivered.append(item)

    with pytest.raises(HTTPException) as info:
        await _drain()

    error = info.value.detail["error"]
    assert delivered == []
    assert info.value.status_code == 400
    assert error["type"] == "guardrail_pipeline_error"
    assert error["policies"] == ("response-governance",)
    assert error["guardrails"] == ("gr-post",)
    assert "stream=false" in error["message"]


@pytest.mark.asyncio
async def test_streaming_iterator_hook_pipeline_delivers_runtime_text_rewrite(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    transform = lambda inputs: {"texts": ["hello [MASKED]"]}  # noqa: E731
    monkeypatch.setattr(litellm, "callbacks", [_rewriting_stream_guardrail(transform)])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data(stream=True)
    chunks = _stream_chunks()

    delivered = [
        item
        async for item in proxy_logging.async_post_call_streaming_iterator_hook(
            user_api_key_dict=make_user_api_key_auth(request_route="/v1/chat/completions"),
            response=_async_chunk_iter(chunks),
            request_data=data,
        )
    ]

    assert [id(item) for item in delivered] == [id(chunk) for chunk in chunks]
    assert delivered[0].choices[0].delta.content == "hello [MASKED]"
    assert delivered[1].choices[0].delta.content in (None, "")
    assert delivered[1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_streaming_iterator_hook_pipeline_chains_text_rewrites_across_steps(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    second_step_saw: Dict[str, Any] = {}

    class FirstMask(CustomGuardrail):
        async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
            return {**inputs, "texts": [text.replace("world", "[MASKED]") for text in inputs["texts"]]}

    class SecondMask(CustomGuardrail):
        async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
            second_step_saw["texts"] = list(inputs["texts"])
            return {**inputs, "texts": [text.replace("hello", "[GREETING]") for text in inputs["texts"]]}

    monkeypatch.setattr(
        litellm,
        "callbacks",
        [
            FirstMask(guardrail_name="gr-first", event_hook=GuardrailEventHooks.post_call, default_on=False),
            SecondMask(guardrail_name="gr-second", event_hook=GuardrailEventHooks.post_call, default_on=False),
        ],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    pipeline = GuardrailPipeline(
        mode="post_call",
        steps=[
            PipelineStep(guardrail="gr-first", on_pass="next", on_fail="block"),
            PipelineStep(guardrail="gr-second", on_pass="allow", on_fail="block"),
        ],
    )
    data = _post_call_pipeline_data(stream=True)
    data["metadata"]["_guardrail_pipelines"] = [("response-governance", pipeline)]
    chunks = _stream_chunks()

    delivered = [
        item
        async for item in proxy_logging.async_post_call_streaming_iterator_hook(
            user_api_key_dict=make_user_api_key_auth(request_route="/v1/chat/completions"),
            response=_async_chunk_iter(chunks),
            request_data=data,
        )
    ]

    assert second_step_saw["texts"] == ["hello [MASKED]"]
    assert delivered[0].choices[0].delta.content == "[GREETING] [MASKED]"
    assert delivered[1].choices[0].delta.content in (None, "")
    assert delivered[1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_chunks, transform",
    [
        (_stream_chunks, lambda inputs: {"texts": tuple(inputs["texts"])}),
        (_tool_call_stream_chunks, lambda inputs: {"tool_calls": _echoed_tool_call_dicts('{"ssn": "123"}')}),
    ],
    ids=["texts_as_tuple", "tool_calls_as_dicts"],
)
async def test_streaming_iterator_hook_pipeline_releases_stream_echoed_in_another_shape(
    proxy_logging, make_user_api_key_auth, monkeypatch, make_chunks, transform
):
    monkeypatch.setattr(litellm, "callbacks", [_rewriting_stream_guardrail(transform)])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data(stream=True)
    chunks = make_chunks()

    delivered = [
        item
        async for item in proxy_logging.async_post_call_streaming_iterator_hook(
            user_api_key_dict=make_user_api_key_auth(request_route="/v1/chat/completions"),
            response=_async_chunk_iter(chunks),
            request_data=data,
        )
    ]

    assert [id(item) for item in delivered] == [id(chunk) for chunk in chunks]


@pytest.mark.asyncio
async def test_streaming_iterator_hook_pipeline_withholds_unresolvable_response_shape(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(litellm, "callbacks", [_unified_stream_guardrail(seen)])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data(stream=True)
    delivered: List[Any] = []

    async def _drain() -> None:
        async for item in proxy_logging.async_post_call_streaming_iterator_hook(
            user_api_key_dict=make_user_api_key_auth(),
            response=_async_chunk_iter([object(), object()]),
            request_data=data,
        ):
            delivered.append(item)

    with pytest.raises(ProxyException) as info:
        await _drain()

    assert delivered == []
    assert info.value.code == "500"
    assert "withheld" in info.value.message
    assert seen.get("count") is None


def _anthropic_sse_chunks() -> List[bytes]:
    events = [
        ("message_start", {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "m", "content": [], "stop_reason": None, "usage": {"input_tokens": 1, "output_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello world"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 2}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return [f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in events]


@pytest.mark.asyncio
async def test_streaming_iterator_hook_pipeline_modify_response_emits_translated_block(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(litellm, "callbacks", [_unified_stream_guardrail(seen, block=True)])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    pipeline = GuardrailPipeline(
        mode="post_call",
        steps=[
            PipelineStep(
                guardrail="gr-post",
                on_pass="allow",
                on_fail="modify_response",
                modify_response_message="content policy block",
            )
        ],
    )
    data = _post_call_pipeline_data(stream=True)
    data["metadata"]["_guardrail_pipelines"] = [("response-governance", pipeline)]
    chunks = _anthropic_sse_chunks()

    delivered = [
        item
        async for item in proxy_logging.async_post_call_streaming_iterator_hook(
            user_api_key_dict=make_user_api_key_auth(request_route="/v1/messages"),
            response=_async_chunk_iter(chunks),
            request_data=data,
        )
    ]

    raw = b"".join(delivered).decode()
    assert seen["count"] == 1
    assert "content policy block" in raw
    assert "hello world" not in raw
    assert not any(item is chunk for item in delivered for chunk in chunks)


@pytest.mark.asyncio
async def test_streaming_iterator_hook_pipeline_delivers_text_rewrite_on_anthropic_sse(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    transform = lambda inputs: {"texts": ["hello [MASKED]"]}  # noqa: E731
    monkeypatch.setattr(litellm, "callbacks", [_rewriting_stream_guardrail(transform)])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data(stream=True)
    chunks = _anthropic_sse_chunks()

    delivered = [
        item
        async for item in proxy_logging.async_post_call_streaming_iterator_hook(
            user_api_key_dict=make_user_api_key_auth(request_route="/v1/messages"),
            response=_async_chunk_iter(chunks),
            request_data=data,
        )
    ]

    raw = b"".join(delivered).decode()
    assert "hello [MASKED]" in raw
    assert "hello world" not in raw
    assert raw.count("event: content_block_delta") == 1
    for expected_event in ("message_start", "content_block_start", "content_block_stop", "message_delta", "message_stop"):
        assert f"event: {expected_event}" in raw


@pytest.mark.asyncio
async def test_pipeline_executor_withholds_text_rewrite_when_translation_lacks_write_back(monkeypatch):
    from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
    from litellm.proxy.policy_engine.pipeline_executor import PipelineExecutor, UndeliverableStreamRewrite

    class NoWriteBackTranslation(BaseTranslation):
        async def process_input_messages(self, data, guardrail_to_apply, litellm_logging_obj):
            return data

        async def process_output_response(self, response, guardrail_to_apply, litellm_logging_obj, **kwargs):
            return response

        async def process_output_streaming_response(
            self,
            responses_so_far,
            guardrail_to_apply,
            litellm_logging_obj,
            user_api_key_dict=None,
            request_data=None,
            stream_transform_sink=None,
            deliver_ended_stream_rewrites=False,
        ):
            assert deliver_ended_stream_rewrites is False
            await guardrail_to_apply.apply_guardrail(
                inputs={"texts": ["hello world"]},
                request_data=request_data or {},
                input_type="response",
            )
            return responses_so_far

    transform = lambda inputs: {"texts": ["hello [MASKED]"]}  # noqa: E731
    monkeypatch.setattr(litellm, "callbacks", [_rewriting_stream_guardrail(transform)])

    with pytest.raises(UndeliverableStreamRewrite):
        await PipelineExecutor.execute_steps(
            steps=[PipelineStep(guardrail="gr-post", on_pass="allow", on_fail="block")],
            mode="post_call",
            data={"metadata": {}},
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
            call_type="acompletion",
            policy_name="response-governance",
            streaming_chunks=_stream_chunks(),
            endpoint_translation=NoWriteBackTranslation(),
        )


@pytest.mark.asyncio
async def test_streaming_iterator_hook_pipeline_gates_without_iterator_overrides(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data(stream=True)
    delivered: List[Any] = []

    async def _drain() -> None:
        async for item in proxy_logging.async_post_call_streaming_iterator_hook(
            user_api_key_dict=make_user_api_key_auth(request_route="/v1/chat/completions"),
            response=_async_chunk_iter(_stream_chunks()),
            request_data=data,
        ):
            delivered.append(item)

    with pytest.raises(HTTPException) as info:
        await _drain()

    assert delivered == []
    assert info.value.status_code == 400
    assert info.value.detail["error"]["pipeline_context"]["step_results"] == [
        {"guardrail": "gr-post", "outcome": "error", "action": "block"}
    ]


@pytest.mark.asyncio
async def test_per_chunk_streaming_hook_skips_pipeline_managed_guardrail(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    seen: Dict[str, Any] = {}

    class RecordingGuardrail(CustomGuardrail):
        async def async_post_call_streaming_hook(self, user_api_key_dict, response):
            seen[self.guardrail_name] = seen.get(self.guardrail_name, 0) + 1
            return None

    managed = RecordingGuardrail(
        guardrail_name="gr-post", event_hook=GuardrailEventHooks.post_call, default_on=True
    )
    free = RecordingGuardrail(
        guardrail_name="gr-free", event_hook=GuardrailEventHooks.post_call, default_on=True
    )
    monkeypatch.setattr(litellm, "callbacks", [managed, free])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)
    data = _post_call_pipeline_data(stream=True)

    result = await proxy_logging.async_post_call_streaming_hook(
        data=data,
        response=_stream_chunks()[0],
        user_api_key_dict=make_user_api_key_auth(request_route="/v1/chat/completions"),
    )

    assert result is not None
    assert seen.get("gr-post") is None
    assert seen["gr-free"] == 1
