"""Unit tests for the LLM-as-a-Judge guardrail hook."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge import (
    LLMAsAJudgeGuardrail,
    _build_judge_prompt,
    _extract_text_from_content,
    _parse_judge_verdict,
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


# ---------------------------------------------------------------------------
# _parse_judge_verdict — tolerate fenced/prose-wrapped JSON
# ---------------------------------------------------------------------------


def test_parse_judge_verdict_plain_json():
    assert _parse_judge_verdict('{"overall_score": 90}')["overall_score"] == 90


def test_parse_judge_verdict_strips_json_fence_and_prose():
    raw = 'Here is my verdict:\n```json\n{"overall_score": 42}\n```\nHope that helps'
    assert _parse_judge_verdict(raw)["overall_score"] == 42


def test_parse_judge_verdict_strips_bare_fence():
    raw = '```\n{"overall_score": 7}\n```'
    assert _parse_judge_verdict(raw)["overall_score"] == 7


def test_parse_judge_verdict_extracts_json_from_surrounding_prose():
    raw = 'Sure, here it is: {"overall_score": 55} let me know'
    assert _parse_judge_verdict(raw)["overall_score"] == 55


def test_parse_judge_verdict_reraises_when_no_json():
    with pytest.raises(json.JSONDecodeError):
        _parse_judge_verdict("no json here")


def test_parse_judge_verdict_rejects_json_non_object():
    """Valid JSON that is not an object (e.g. a bare list) raises ValueError."""
    with pytest.raises(ValueError):
        _parse_judge_verdict("[1, 2, 3]")


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_apply_guardrail_enforces_fenced_verdict(mock_completion):
    """A failing verdict wrapped in a code fence blocks with a 422."""
    fenced = "```json\n" + json.dumps(_make_verdict_response(50.0)) + "\n```"
    mock_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content=fenced))])
    guardrail = _make_guardrail(overall_threshold=80.0, on_failure="block", router_provider=lambda: None)
    inputs = {"texts": ["bad response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(inputs, request_data, "response")
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_apply_guardrail_non_object_verdict_fails_open_with_status(mock_completion):
    """A non-object verdict fails open and logs guardrail_failed_to_respond."""
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='[{"overall_score": 50}]'))]
    )
    guardrail = _make_guardrail(overall_threshold=80.0, on_failure="block", router_provider=lambda: None)
    inputs = {"texts": ["response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    result = await guardrail.apply_guardrail(inputs, request_data, "response")
    assert result is inputs
    logged = request_data["metadata"]["standard_logging_guardrail_information"]
    assert logged[0]["guardrail_status"] == "guardrail_failed_to_respond"


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_apply_guardrail_parses_fenced_json_verdict(mock_completion):
    """Fencing-prone judge models wrap the verdict in a ```json fence; the guardrail
    must parse it and evaluate rather than failing open on json.loads."""
    fenced = "```json\n" + json.dumps(_make_verdict_response(90.0)) + "\n```"
    mock_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content=fenced))])
    guardrail = _make_guardrail(overall_threshold=80.0)
    inputs = {"texts": ["good response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    result = await guardrail.apply_guardrail(inputs, request_data, "response")
    assert result is inputs
    assert request_data["metadata"]["eval_information"]["passed"] is True


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


def _judge_response_mock() -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(_make_verdict_response(90.0))))])


def _real_router(model_list, **router_kwargs):
    """Build a real Router so the router-membership decision is exercised for
    real (wildcards, model_group_alias, exact names), stubbing only the outbound
    completion so no network call is made."""
    from litellm import Router

    router = Router(model_list=model_list, **router_kwargs)
    router.acompletion = AsyncMock(return_value=_judge_response_mock())
    return router


@pytest.mark.parametrize(
    "model_list, router_kwargs, judge_model",
    [
        (
            [{"model_name": "my-judge-alias", "litellm_params": {"model": "anthropic/claude-sonnet-4-6", "api_key": "sk-ant-test"}}],
            {},
            "my-judge-alias",
        ),
        (
            [{"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*", "api_key": "sk-ant-test"}}],
            {},
            "anthropic/claude-sonnet-4-6",
        ),
        (
            [{"model_name": "backing-group", "litellm_params": {"model": "anthropic/claude-sonnet-4-6", "api_key": "sk-ant-test"}}],
            {"model_group_alias": {"my-judge-alias": "backing-group"}},
            "my-judge-alias",
        ),
        (
            [{"model_name": "backing-group", "litellm_params": {"model": "anthropic/claude-sonnet-4-6", "api_key": "sk-ant-test"}}],
            {"model_group_alias": {"my-judge-alias": {"model": "backing-group", "hidden": True}}},
            "my-judge-alias",
        ),
    ],
    ids=["plain-deployment", "wildcard-route", "model-group-alias", "hidden-model-group-alias"],
)
@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion", new_callable=AsyncMock)
async def test_judge_routes_through_router_for_router_served_model(
    mock_sdk_completion, model_list, router_kwargs, judge_model
):
    """Any judge_model the Router can serve must resolve its credentials via the
    Router. Wildcard and alias shapes regress the naive `judge_model in
    get_model_names()` check, which reports patterns/aliases literally and so
    routes a servable model to the SDK, where deployment creds do not resolve."""
    router = _real_router(model_list, **router_kwargs)
    guardrail = _make_guardrail(judge_model=judge_model, router_provider=lambda: router)
    inputs = {"texts": ["good response"]}
    request_data: dict = {"messages": [{"role": "user", "content": "hi"}], "metadata": {}}

    result = await guardrail.apply_guardrail(inputs, request_data, "response")

    assert result is inputs
    router.acompletion.assert_awaited_once()
    call_kwargs = router.acompletion.await_args.kwargs
    assert call_kwargs["model"] == judge_model
    assert call_kwargs["num_retries"] == 0
    assert call_kwargs["fallbacks"] == []
    mock_sdk_completion.assert_not_called()


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion", new_callable=AsyncMock)
async def test_judge_call_falls_back_to_sdk_when_model_not_in_router(mock_sdk_completion):
    """A judge_model the Router cannot serve (e.g. a raw provider model resolved
    from the environment) must fall back to the SDK."""
    mock_sdk_completion.return_value = _judge_response_mock()
    router = _real_router(
        [{"model_name": "some-other-model", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"}}]
    )
    guardrail = _make_guardrail(judge_model="gpt-4o-mini", router_provider=lambda: router)
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
    mock_sdk_completion.return_value = _judge_response_mock()
    guardrail = _make_guardrail(judge_model="gpt-4o-mini", router_provider=lambda: None)
    inputs = {"texts": ["good response"]}
    request_data: dict = {"messages": [], "metadata": {}}

    result = await guardrail.apply_guardrail(inputs, request_data, "response")

    assert result is inputs
    mock_sdk_completion.assert_awaited_once()


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion", new_callable=AsyncMock)
async def test_judge_resolves_router_lazily_per_call(mock_sdk_completion):
    """The Router is resolved at call time, not captured at construction. A
    guardrail built before the proxy Router exists (provider returns None) starts
    routing through the Router as soon as it is available, with no re-init. This
    regresses the config-less DB-backed startup order where the guardrail was
    created while the global Router was still None and then never recovered."""
    mock_sdk_completion.return_value = _judge_response_mock()
    holder: dict = {"router": None}
    guardrail = _make_guardrail(judge_model="my-judge-alias", router_provider=lambda: holder["router"])

    await guardrail.apply_guardrail({"texts": ["r"]}, {"messages": [], "metadata": {}}, "response")
    mock_sdk_completion.assert_awaited_once()

    holder["router"] = _real_router(
        [{"model_name": "my-judge-alias", "litellm_params": {"model": "anthropic/claude-sonnet-4-6", "api_key": "sk-ant-test"}}]
    )
    await guardrail.apply_guardrail({"texts": ["r"]}, {"messages": [], "metadata": {}}, "response")
    holder["router"].acompletion.assert_awaited_once()
    mock_sdk_completion.assert_awaited_once()


def test_default_router_provider_returns_none_when_proxy_not_importable():
    """If the proxy dependency set is not importable, the provider must return None
    so the judge falls back to the SDK rather than the ImportError being swallowed
    by the fail-open handler and the guardrail silently no-opping."""
    import sys

    from litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge import _default_router_provider

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": None}):
        assert _default_router_provider() is None


def test_default_router_provider_reads_global_router():
    """The default provider must read the live proxy global so the router is
    resolved lazily rather than captured."""
    from litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge import _default_router_provider

    sentinel = object()
    with patch("litellm.proxy.proxy_server.llm_router", sentinel):
        assert _default_router_provider() is sentinel


@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.logging_callback_manager")
def test_initialize_guardrail_uses_default_router_provider(mock_mgr):
    from litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge import _default_router_provider

    lp = _make_litellm_params()
    g = _make_guardrail_dict(judge_model="my-judge-alias")
    instance = initialize_guardrail(lp, g)
    assert instance._router_provider is _default_router_provider


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
