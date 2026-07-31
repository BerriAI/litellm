"""Unit tests for the LLM-as-a-Judge guardrail hook."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge import (
    LLMAsAJudgeGuardrail,
    _build_judge_prompt,
    _derive_overall_score,
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
    for attr in ("guardrail_name", "judge_model", "criteria", "on_failure", "overall_threshold", "fail_closed", "mode", "default_on"):
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


# ---------------------------------------------------------------------------
# _derive_overall_score — score resolution (issue #30731)
# ---------------------------------------------------------------------------


def test_derive_overall_score_prefers_top_level():
    assert _derive_overall_score({"overall_score": 73, "verdicts": []}) == 73.0


def test_derive_overall_score_clamps_top_level():
    assert _derive_overall_score({"overall_score": 150}) == 100.0
    assert _derive_overall_score({"overall_score": -10}) == 0.0


def test_derive_overall_score_weighted_average_from_verdicts():
    # Missing overall_score -> weighted mean: (2*60 + 5*40) / 100 = 3.2
    result = _derive_overall_score(
        {
            "verdicts": [
                {"criterion_name": "Safety", "score": 2, "passed": False, "weight": 60},
                {"criterion_name": "Policy", "score": 5, "passed": False, "weight": 40},
            ]
        }
    )
    assert result == pytest.approx(3.2)


def test_derive_overall_score_simple_mean_when_no_weights():
    result = _derive_overall_score(
        {"verdicts": [{"score": 2}, {"score": 5}, {"score": 8}]}
    )
    assert result == pytest.approx(5.0)


def test_derive_overall_score_falls_back_to_verdicts_when_unparseable():
    result = _derive_overall_score(
        {
            "overall_score": "not-a-number",
            "verdicts": [{"score": 90, "weight": 100}],
        }
    )
    assert result == pytest.approx(90.0)


def test_derive_overall_score_none_when_no_data():
    assert _derive_overall_score({}) is None
    assert _derive_overall_score({"verdicts": []}) is None
    assert _derive_overall_score({"verdicts": [{"reasoning": "no score"}]}) is None


def test_derive_overall_score_bad_weight_falls_back_to_simple_mean():
    # A non-numeric weight is not a fail-open vector: the score still counts, the
    # weighted path is just disabled in favour of a simple mean of all scores.
    result = _derive_overall_score(
        {
            "verdicts": [
                {"score": 40, "weight": "heavy"},  # bad weight -> ignored weight
                {"score": 60, "weight": "x"},  # bad weight -> ignored weight
            ]
        }
    )
    assert result == pytest.approx(50.0)


def test_derive_overall_score_mixed_weighted_and_unweighted_uses_simple_mean():
    # A verdict missing its weight must NOT be silently dropped from the score.
    # A(90, weight 60) + B(0, no weight) -> simple mean 45, NOT weighted 90.
    result = _derive_overall_score(
        {
            "verdicts": [
                {"criterion_name": "A", "score": 90, "weight": 60},
                {"criterion_name": "B", "score": 0},  # weight missing
            ]
        }
    )
    assert result == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# _derive_overall_score — malformed / manipulated judge output (veria-ai)
# ---------------------------------------------------------------------------


def test_derive_overall_score_rejects_non_finite_top_level():
    # NaN/Infinity must not clamp to 100. With no verdicts -> indeterminate.
    assert _derive_overall_score({"overall_score": float("nan")}) is None
    assert _derive_overall_score({"overall_score": float("inf")}) is None
    assert _derive_overall_score({"overall_score": "nan"}) is None


def test_derive_overall_score_non_finite_top_level_falls_back_to_verdicts():
    result = _derive_overall_score(
        {"overall_score": float("nan"), "verdicts": [{"score": 10, "weight": 100}]}
    )
    assert result == pytest.approx(10.0)


def test_derive_overall_score_indeterminate_on_non_finite_verdict_score():
    # A NaN verdict score makes the evaluation indeterminate: it must NOT be
    # silently skipped and the remaining verdicts averaged, because the dropped
    # verdict could be the failing one. Indeterminate -> None -> caller decides.
    result = _derive_overall_score(
        {"verdicts": [{"score": "nan", "weight": 60}, {"score": 0, "weight": 40}]}
    )
    assert result is None


def test_derive_overall_score_clamps_inflated_verdict_score():
    # An out-of-range score is clamped per verdict so it cannot mask a failure.
    # clamp(10000) = 100, simple mean(100, 0) = 50.
    result = _derive_overall_score({"verdicts": [{"score": 10000}, {"score": 0}]})
    assert result == pytest.approx(50.0)


def test_derive_overall_score_none_for_non_dict():
    assert _derive_overall_score([]) is None
    assert _derive_overall_score("not a dict") is None
    assert _derive_overall_score(123) is None


def test_derive_overall_score_indeterminate_on_unscored_failing_verdict():
    # The judge marks a criterion as failed but omits its numeric score. The
    # failing verdict must NOT be dropped while the passing one is averaged into
    # a 100 -> the whole evaluation is indeterminate (None).
    result = _derive_overall_score(
        {
            "verdicts": [
                {"criterion_name": "Accuracy", "score": 100, "weight": 60},
                {"criterion_name": "Safety", "passed": False, "weight": 40},  # no score
            ]
        }
    )
    assert result is None


def test_derive_overall_score_indeterminate_on_malformed_verdict():
    # A non-dict entry or an unparsable score among otherwise-valid verdicts also
    # makes the derivation indeterminate rather than averaging the rest.
    assert _derive_overall_score({"verdicts": [{"score": 90}, "junk"]}) is None
    assert (
        _derive_overall_score({"verdicts": [{"score": 90}, {"score": "abc"}]}) is None
    )


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_non_dict_judge_output_fail_closed_blocks(mock_completion):
    # A top-level JSON array is valid JSON but unusable -> indeterminate ->
    # fail_closed must block instead of slipping through the outer except.
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps([])))]
    )
    guardrail = _make_guardrail(fail_closed=True, on_failure="block")
    inputs = {"texts": ["response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(inputs, request_data, "response")
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_inflated_verdict_scores_still_block(mock_completion):
    # Weighted path: clamp(10000)=100, (100*60 + 0*40)/100 = 60 < 80 -> block.
    payload = {
        "verdicts": [
            {"criterion_name": "A", "score": 10000, "weight": 60},
            {"criterion_name": "B", "score": 0, "weight": 40},
        ]
    }
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
    )
    guardrail = _make_guardrail(overall_threshold=80.0, on_failure="block")
    inputs = {"texts": ["response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(inputs, request_data, "response")
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_unscored_failing_verdict_fail_closed_blocks(mock_completion):
    # End-to-end: judge fails a criterion but omits its score. The passing
    # verdict alone must not slip through -> indeterminate -> fail_closed blocks.
    payload = {
        "verdicts": [
            {"criterion_name": "Accuracy", "score": 100, "weight": 60},
            {"criterion_name": "Safety", "passed": False, "weight": 40},  # no score
        ]
    }
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
    )
    guardrail = _make_guardrail(fail_closed=True, on_failure="block")
    inputs = {"texts": ["partially unsafe response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(inputs, request_data, "response")
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_missing_weight_on_failing_verdict_still_blocks(mock_completion):
    # End-to-end: judge drops the weight on a failing criterion. The failing
    # score must still pull the derived score below threshold and block.
    payload = {
        "verdicts": [
            {"criterion_name": "Accuracy", "score": 100, "weight": 60},
            {"criterion_name": "Safety", "score": 0},  # failing, weight missing
        ]
    }
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
    )
    guardrail = _make_guardrail(overall_threshold=80.0, on_failure="block")
    inputs = {"texts": ["partially unsafe response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(inputs, request_data, "response")
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# apply_guardrail — missing overall_score (fail-open regression, issue #30731)
# ---------------------------------------------------------------------------


def _verdicts_only_response(scores_weights) -> dict:
    """Build a judge response with verdicts but NO top-level overall_score."""
    return {
        "verdicts": [
            {
                "criterion_name": name,
                "score": score,
                "reasoning": "r",
                "passed": score >= 80,
                "weight": weight,
            }
            for name, score, weight in scores_weights
        ]
    }


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_missing_overall_score_with_failing_verdicts_blocks(mock_completion):
    # The core bug: judge omits overall_score but every verdict fails. Previously
    # defaulted to 100 (pass); now derived from verdicts -> 3.2 -> blocked.
    payload = _verdicts_only_response([("Safety", 2, 60), ("Policy", 5, 40)])
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
    )
    guardrail = _make_guardrail(overall_threshold=80.0, on_failure="block")
    inputs = {"texts": ["unsafe response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(inputs, request_data, "response")
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_missing_overall_score_with_passing_verdicts_passes(mock_completion):
    payload = _verdicts_only_response([("Accuracy", 95, 60), ("Safety", 90, 40)])
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
    )
    guardrail = _make_guardrail(overall_threshold=80.0)
    inputs = {"texts": ["good response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    result = await guardrail.apply_guardrail(inputs, request_data, "response")
    assert result is inputs
    assert request_data["metadata"]["eval_information"]["passed"] is True
    assert request_data["metadata"]["eval_information"]["overall_score"] == pytest.approx(93.0)


# ---------------------------------------------------------------------------
# apply_guardrail — fail_closed configuration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_no_score_no_verdicts_fails_open_by_default(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({})))]
    )
    guardrail = _make_guardrail()  # fail_closed defaults to False
    inputs = {"texts": ["response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    result = await guardrail.apply_guardrail(inputs, request_data, "response")
    assert result is inputs


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_no_score_no_verdicts_fail_closed_blocks(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({})))]
    )
    guardrail = _make_guardrail(fail_closed=True, on_failure="block")
    inputs = {"texts": ["response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(inputs, request_data, "response")
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_fail_closed_log_mode_does_not_block(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({})))]
    )
    guardrail = _make_guardrail(fail_closed=True, on_failure="log")
    inputs = {"texts": ["response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    result = await guardrail.apply_guardrail(inputs, request_data, "response")
    assert result is inputs


@pytest.mark.asyncio
@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.acompletion")
async def test_judge_error_fail_closed_blocks(mock_completion):
    mock_completion.side_effect = RuntimeError("judge down")
    guardrail = _make_guardrail(fail_closed=True, on_failure="block")
    inputs = {"texts": ["response"]}
    request_data: dict = {"messages": [], "metadata": {}}
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(inputs, request_data, "response")
    assert exc_info.value.status_code == 422


@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.logging_callback_manager")
def test_initialize_guardrail_parses_fail_closed(mock_mgr):
    lp = _make_litellm_params()
    g = _make_guardrail_dict(fail_closed=True)
    instance = initialize_guardrail(lp, g)
    assert instance.fail_closed is True


@patch("litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge.litellm.logging_callback_manager")
def test_initialize_guardrail_fail_closed_defaults_false(mock_mgr):
    lp = _make_litellm_params()
    g = _make_guardrail_dict()
    instance = initialize_guardrail(lp, g)
    assert instance.fail_closed is False
