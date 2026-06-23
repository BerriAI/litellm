"""Pin ProxyLogging guardrail pipeline helpers.

Covers ``_should_use_guardrail_load_balancing``, ``_execute_guardrail_hook``,
``_execute_guardrail_with_load_balancing``, ``_process_guardrail_callback``,
``_process_prompt_template``, ``_process_guardrail_metadata``,
``_maybe_execute_pipelines``, ``_handle_pipeline_result``,
``_run_guardrail_task_with_enrichment``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks


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
    out = await proxy_logging._maybe_execute_pipelines(
        data=data,
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
        event_hook="pre_call",
    )
    assert out == {"messages": [{"role": "user"}], "model": "m", "temperature": 0.1}


@pytest.mark.asyncio
async def test_maybe_execute_pipelines_skips_pipelines_with_other_mode(proxy_logging, make_user_api_key_auth, monkeypatch):
    pipeline = MagicMock()
    pipeline.mode = "post_call"  # not pre_call
    data = {"metadata": {"_guardrail_pipelines": [("p1", pipeline)]}, "model": "m", "messages": []}
    executed = MagicMock()
    monkeypatch.setattr(
        "litellm.proxy.policy_engine.pipeline_executor.PipelineExecutor.execute_steps", executed
    )
    out = await proxy_logging._maybe_execute_pipelines(
        data=data,
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
        event_hook="pre_call",
    )
    executed.assert_not_called()
    assert out is data


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


def test_handle_pipeline_result_block_raises_http_exception():
    result = MagicMock()
    result.terminal_action = "block"
    result.step_results = []
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
# _run_guardrail_task_with_enrichment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_guardrail_task_with_enrichment_passes_result():
    async def task():
        return {"a": 1, "b": 2, "c": 3}

    out = await ProxyLogging._run_guardrail_task_with_enrichment(
        callback=MagicMock(guardrail_name="g"), coro=task()
    )
    assert out == {"a": 1, "b": 2, "c": 3}


@pytest.mark.asyncio
async def test_run_guardrail_task_with_enrichment_enriches_http_exception_raises():
    detail = {"error": "blocked"}

    async def task():
        raise HTTPException(status_code=400, detail=detail)

    cb = MagicMock()
    cb.guardrail_name = "presidio"
    cb.event_hook = "pre_call"
    with pytest.raises(HTTPException):
        await ProxyLogging._run_guardrail_task_with_enrichment(callback=cb, coro=task())
    assert detail["guardrail_name"] == "presidio"


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
