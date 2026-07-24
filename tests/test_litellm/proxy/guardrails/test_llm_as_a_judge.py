"""Unit tests for the LLM-as-a-Judge guardrail hook."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge import (
    LLMAsAJudgeGuardrail,
    _build_judge_prompt,
    _extract_text_from_content,
    initialize_guardrail,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CRITERIA_100 = [
    {"name": "Accuracy", "weight": 60, "description": "Is it accurate?"},
    {"name": "Safety", "weight": 40, "description": "Is it safe?"},
]


def _make_guardrail(**overrides) -> LLMAsAJudgeGuardrail:
    kwargs = dict(
        guardrail_name="test_judge",
        judge_model="gpt-4o-mini",
        criteria=CRITERIA_100,
        overall_threshold=80.0,
        on_failure="block",
    )
    kwargs.update(overrides)
    return LLMAsAJudgeGuardrail(**kwargs)


def _make_verdict_response(overall_score: float) -> dict:
    return {
        "verdicts": [
            {"criterion_name": "Accuracy", "score": overall_score, "reasoning": "ok", "passed": True, "weight": 60},
            {"criterion_name": "Safety", "score": overall_score, "reasoning": "ok", "passed": True, "weight": 40},
        ],
        "overall_score": overall_score,
    }


# ---------------------------------------------------------------------------
# _extract_text_from_content
# ---------------------------------------------------------------------------


def test_extract_text_str():
    assert _extract_text_from_content("hello") == "hello"


def test_extract_text_multimodal_list():
    content = [{"type": "text", "text": "hello"}, {"type": "image_url", "url": "x"}]
    assert _extract_text_from_content(content) == "hello"


def test_extract_text_unknown_type():
    assert _extract_text_from_content(42) == ""


# ---------------------------------------------------------------------------
# _build_judge_prompt
# ---------------------------------------------------------------------------


def test_build_judge_prompt_contains_criteria():
    prompt = _build_judge_prompt(CRITERIA_100, [], "response text")
    assert "Accuracy" in prompt
    assert "60%" in prompt
    assert "Safety" in prompt
    assert "response text" in prompt


def test_build_judge_prompt_missing_name_and_weight():
    criteria = [{"description": "check it"}]
    prompt = _build_judge_prompt(criteria, [], "resp")
    assert "0%" in prompt


# ---------------------------------------------------------------------------
# initialize_guardrail — validation
# ---------------------------------------------------------------------------


def _make_litellm_params(**overrides):
    params = MagicMock()
    for attr in ("guardrail_name", "judge_model", "criteria", "on_failure", "overall_threshold", "mode", "default_on"):
        setattr(params, attr, None)
    for k, v in overrides.items():
        setattr(params, k, v)
    return params


def _make_guardrail_dict(name="g", **litellm_params_overrides):
    raw = {"judge_model": "gpt-4o-mini", "criteria": CRITERIA_100, "on_failure": "block", "overall_threshold": 80.0}
    raw.update(litellm_params_overrides)
    return {"guardrail_name": name, "litellm_params": raw}


@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.logging_callback_manager")
def test_initialize_guardrail_ok(mock_mgr):
    lp = _make_litellm_params()
    g = _make_guardrail_dict()
    instance = initialize_guardrail(lp, g)
    assert isinstance(instance, LLMAsAJudgeGuardrail)
    mock_mgr.add_litellm_callback.assert_called_once_with(instance)


def test_initialize_guardrail_missing_judge_model():
    lp = _make_litellm_params()
    g = _make_guardrail_dict(judge_model=None)
    g["litellm_params"].pop("judge_model")
    with pytest.raises(ValueError, match="judge_model"):
        initialize_guardrail(lp, g)


def test_initialize_guardrail_weight_sum_not_100():
    lp = _make_litellm_params()
    bad_criteria = [{"name": "A", "weight": 50, "description": "d"}]
    g = _make_guardrail_dict(criteria=bad_criteria)
    with pytest.raises(ValueError, match="100"):
        initialize_guardrail(lp, g)


def test_initialize_guardrail_invalid_on_failure():
    lp = _make_litellm_params()
    g = _make_guardrail_dict(on_failure="explode")
    with pytest.raises(ValueError, match="on_failure"):
        initialize_guardrail(lp, g)


# ---------------------------------------------------------------------------
# apply_guardrail — enforcement paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_pre_call_passthrough():
    guardrail = _make_guardrail()
    inputs = {"texts": ["some text"]}
    result = await guardrail.apply_guardrail(inputs, {}, "request")
    assert result is inputs


@pytest.mark.asyncio
async def test_apply_guardrail_empty_response_passthrough():
    guardrail = _make_guardrail()
    inputs = {"texts": []}
    result = await guardrail.apply_guardrail(inputs, {}, "response")
    assert result is inputs


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_apply_guardrail_passes_above_threshold(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(_make_verdict_response(90.0))))]
    )
    guardrail = _make_guardrail(overall_threshold=80.0)
    inputs = {"texts": ["good response"]}
    request_data: dict = {"messages": [{"role": "user", "content": "hi"}], "metadata": {}}
    result = await guardrail.apply_guardrail(inputs, request_data, "response")
    assert result is inputs
    assert request_data["metadata"]["eval_information"]["passed"] is True


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_apply_guardrail_blocks_below_threshold(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(_make_verdict_response(50.0))))]
    )
    guardrail = _make_guardrail(overall_threshold=80.0, on_failure="block")
    inputs = {"texts": ["bad response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(inputs, request_data, "response")
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_apply_guardrail_log_mode_does_not_block(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(_make_verdict_response(50.0))))]
    )
    guardrail = _make_guardrail(overall_threshold=80.0, on_failure="log")
    inputs = {"texts": ["bad response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    result = await guardrail.apply_guardrail(inputs, request_data, "response")
    assert result is inputs
    assert request_data["metadata"]["eval_information"]["passed"] is False


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_apply_guardrail_judge_error_fails_open(mock_completion):
    mock_completion.side_effect = RuntimeError("judge down")
    guardrail = _make_guardrail()
    inputs = {"texts": ["response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    result = await guardrail.apply_guardrail(inputs, request_data, "response")
    assert result is inputs


# ---------------------------------------------------------------------------
# judge_model credential/provider resolution — route through the proxy Router
# ---------------------------------------------------------------------------


def _make_router(model_names):
    router = MagicMock()
    router.get_model_names.return_value = list(model_names)
    router.acompletion = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(_make_verdict_response(90.0))))]
        )
    )
    return router


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion", new_callable=AsyncMock)
async def test_judge_call_routes_through_router_for_configured_model(mock_sdk_completion):
    """A judge_model that is a configured proxy deployment must resolve its
    credentials via the Router, not the SDK (which cannot see deployment creds)."""
    router = _make_router({"my-judge-alias"})
    guardrail = _make_guardrail(judge_model="my-judge-alias", llm_router=router)
    inputs = {"texts": ["good response"]}
    request_data: dict = {"messages": [{"role": "user", "content": "hi"}], "metadata": {}}

    result = await guardrail.apply_guardrail(inputs, request_data, "response")

    assert result is inputs
    router.acompletion.assert_awaited_once()
    assert router.acompletion.await_args.kwargs["model"] == "my-judge-alias"
    mock_sdk_completion.assert_not_called()


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion", new_callable=AsyncMock)
async def test_judge_call_falls_back_to_sdk_when_model_not_in_router(mock_sdk_completion):
    """A judge_model that is not a configured deployment (e.g. a raw provider
    model resolved from the environment) must fall back to the SDK."""
    mock_sdk_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(_make_verdict_response(90.0))))]
    )
    router = _make_router({"some-other-model"})
    guardrail = _make_guardrail(judge_model="gpt-4o-mini", llm_router=router)
    inputs = {"texts": ["good response"]}
    request_data: dict = {"messages": [], "metadata": {}}

    result = await guardrail.apply_guardrail(inputs, request_data, "response")

    assert result is inputs
    router.acompletion.assert_not_called()
    mock_sdk_completion.assert_awaited_once()
    assert mock_sdk_completion.await_args.kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion", new_callable=AsyncMock)
async def test_judge_call_uses_sdk_when_no_router(mock_sdk_completion):
    mock_sdk_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(_make_verdict_response(90.0))))]
    )
    guardrail = _make_guardrail(judge_model="gpt-4o-mini")
    inputs = {"texts": ["good response"]}
    request_data: dict = {"messages": [], "metadata": {}}

    result = await guardrail.apply_guardrail(inputs, request_data, "response")

    assert result is inputs
    mock_sdk_completion.assert_awaited_once()


@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.logging_callback_manager")
def test_initialize_guardrail_wires_llm_router(mock_mgr):
    router = _make_router({"my-judge-alias"})
    lp = _make_litellm_params()
    g = _make_guardrail_dict(judge_model="my-judge-alias")
    instance = initialize_guardrail(lp, g, router)
    assert instance.llm_router is router


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_apply_guardrail_clamps_score(mock_completion):
    response_payload = {"verdicts": [], "overall_score": 150}
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(response_payload)))]
    )
    guardrail = _make_guardrail(overall_threshold=80.0)
    inputs = {"texts": ["response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    result = await guardrail.apply_guardrail(inputs, request_data, "response")
    assert result is inputs
    assert request_data["metadata"]["eval_information"]["overall_score"] == 100.0
