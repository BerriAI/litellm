import pytest

from litellm.router_utils.pre_call_checks.continuation_prefill_check import (
    MID_STREAM_CONTINUATION_KWARG,
    MID_STREAM_CONTINUATION_MARKER,
    ContinuationPrefillDeploymentCheck,
    _deployment_supports_prefill,
)

PREFILL_MODEL = "anthropic/claude-3-opus-20240229"  # supports_assistant_prefill: True in the cost map
NON_PREFILL_MODEL = "openai/gpt-4o"  # capability absent -> treated as unsupported


def _deployment(model: str, dep_id: str) -> dict:
    return {"litellm_params": {"model": model}, "model_info": {"id": dep_id}}


def test_deployment_supports_prefill_reads_capability():
    assert _deployment_supports_prefill(_deployment(PREFILL_MODEL, "a")) is True
    assert _deployment_supports_prefill(_deployment(NON_PREFILL_MODEL, "b")) is False


def test_deployment_model_info_override_wins_over_cost_map():
    # model_info True opts in a model that is not in the cost map
    assert (
        _deployment_supports_prefill(
            {"litellm_params": {"model": "vendor/custom-model"}, "model_info": {"supports_assistant_prefill": True}}
        )
        is True
    )
    # model_info False opts out a model the cost map would otherwise allow
    assert (
        _deployment_supports_prefill(
            {"litellm_params": {"model": PREFILL_MODEL}, "model_info": {"supports_assistant_prefill": False}}
        )
        is False
    )
    # model_info without the key falls through to the cost map
    assert _deployment_supports_prefill(_deployment(PREFILL_MODEL, "z")) is True


def test_deployment_supports_prefill_rejects_malformed_deployments():
    assert _deployment_supports_prefill({}) is False
    assert _deployment_supports_prefill({"litellm_params": {}}) is False
    assert _deployment_supports_prefill("not-a-dict") is False


@pytest.mark.asyncio
async def test_filter_is_noop_without_continuation_marker():
    """A normal (non-continuation) request must be passed through untouched, even
    if some deployments cannot prefill."""
    check = ContinuationPrefillDeploymentCheck()
    deployments = [_deployment(PREFILL_MODEL, "a"), _deployment(NON_PREFILL_MODEL, "b")]

    for request_kwargs in ({}, None, {MID_STREAM_CONTINUATION_KWARG: False}):
        result = await check.async_filter_deployments(
            model="group", healthy_deployments=deployments, messages=None, request_kwargs=request_kwargs
        )
        assert result == deployments


@pytest.mark.asyncio
async def test_filter_ignores_forged_client_flag():
    """A client cannot steer routing: a plain truthy value under the marker key
    (which the proxy could forward from the request body) is not the internal
    sentinel, so the filter leaves the deployment list untouched."""
    check = ContinuationPrefillDeploymentCheck()
    deployments = [_deployment(PREFILL_MODEL, "a"), _deployment(NON_PREFILL_MODEL, "b")]

    for forged in (True, "true", 1, {"any": "json"}):
        result = await check.async_filter_deployments(
            model="group",
            healthy_deployments=deployments,
            messages=None,
            request_kwargs={MID_STREAM_CONTINUATION_KWARG: forged},
        )
        assert result == deployments


@pytest.mark.asyncio
async def test_filter_keeps_only_prefill_capable_on_continuation():
    check = ContinuationPrefillDeploymentCheck()
    deployments = [_deployment(PREFILL_MODEL, "a"), _deployment(NON_PREFILL_MODEL, "b")]

    result = await check.async_filter_deployments(
        model="group",
        healthy_deployments=deployments,
        messages=None,
        request_kwargs={MID_STREAM_CONTINUATION_KWARG: MID_STREAM_CONTINUATION_MARKER},
    )
    assert [d["model_info"]["id"] for d in result] == ["a"]


@pytest.mark.asyncio
async def test_filter_empties_group_when_no_prefill_capable_deployment():
    """No prefill-capable deployment -> empty result, so the router advances the
    fallback chain and ultimately surfaces the original error."""
    check = ContinuationPrefillDeploymentCheck()
    deployments = [_deployment(NON_PREFILL_MODEL, "b"), _deployment("openai/gpt-4.1", "c")]

    result = await check.async_filter_deployments(
        model="group",
        healthy_deployments=deployments,
        messages=None,
        request_kwargs={MID_STREAM_CONTINUATION_KWARG: MID_STREAM_CONTINUATION_MARKER},
    )
    assert result == []
