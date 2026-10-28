import pytest

import litellm
from litellm.router_utils.pre_call_checks.continuation_prefill_check import (
    MID_STREAM_CONTINUATION_KWARG,
    MID_STREAM_CONTINUATION_MARKER,
    ContinuationOutput,
    ContinuationPrefillDeploymentCheck,
    continuation_output_ceilings,
    deployment_supports_prefill,
)

PREFILL_MODEL = "anthropic/prefill-capable-test-model"
NON_PREFILL_MODEL = "openai/prefill-unknown-test-model"
UNMAPPED_MODEL = "openai/unmapped-test-model"


@pytest.fixture(autouse=True)
def _synthetic_cost_map_entries() -> None:
    litellm.register_model(
        {
            PREFILL_MODEL: {"litellm_provider": "anthropic", "mode": "chat", "supports_assistant_prefill": True},
            NON_PREFILL_MODEL: {"litellm_provider": "openai", "mode": "chat"},
        },
        persist_across_reloads=False,
    )
    litellm.get_model_info.cache_clear()


def _deployment(model: str, dep_id: str) -> dict[str, object]:
    return {"litellm_params": {"model": model}, "model_info": {"id": dep_id}}


def test_deployment_supports_prefill_reads_capability():
    assert deployment_supports_prefill(_deployment(PREFILL_MODEL, "a")) is True
    assert deployment_supports_prefill(_deployment(NON_PREFILL_MODEL, "b")) is False
    assert deployment_supports_prefill({"litellm_params": {"model": PREFILL_MODEL}}) is True
    assert deployment_supports_prefill({"litellm_params": {"model": PREFILL_MODEL}, "model_info": "bogus"}) is True


def test_deployment_model_info_override_wins_over_cost_map():
    # model_info True opts in a model that is not in the cost map
    assert (
        deployment_supports_prefill(
            {"litellm_params": {"model": "vendor/custom-model"}, "model_info": {"supports_assistant_prefill": True}}
        )
        is True
    )
    # model_info False opts out a model the cost map would otherwise allow
    assert (
        deployment_supports_prefill(
            {"litellm_params": {"model": PREFILL_MODEL}, "model_info": {"supports_assistant_prefill": False}}
        )
        is False
    )
    # model_info without the key falls through to the cost map
    assert deployment_supports_prefill(_deployment(PREFILL_MODEL, "z")) is True


@pytest.mark.parametrize(
    "deployment,expected",
    [
        ({"litellm_params": {"model": "bedrock/arn:custom-import", "base_model": PREFILL_MODEL}}, True),
        ({"litellm_params": {"model": "azure/my-deployment"}, "model_info": {"base_model": PREFILL_MODEL}}, True),
        (
            {
                "litellm_params": {"model": "azure/my-deployment", "base_model": NON_PREFILL_MODEL},
                "model_info": {"base_model": PREFILL_MODEL},
            },
            True,
        ),
        (
            {
                "litellm_params": {"model": PREFILL_MODEL, "base_model": NON_PREFILL_MODEL},
                "model_info": {"base_model": NON_PREFILL_MODEL},
            },
            False,
        ),
    ],
    ids=[
        "litellm_params_base_model",
        "model_info_base_model",
        "model_info_base_model_wins",
        "base_model_wins_over_model",
    ],
)
def test_deployment_supports_prefill_reads_the_cost_map_key_of_the_base_model(deployment, expected):
    assert deployment_supports_prefill(deployment) is expected


def test_deployment_supports_prefill_rejects_malformed_deployments():
    assert deployment_supports_prefill({}) is False
    assert deployment_supports_prefill({"litellm_params": {}}) is False
    assert deployment_supports_prefill("not-a-dict") is False


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
    deployments = [_deployment(NON_PREFILL_MODEL, "b"), _deployment(UNMAPPED_MODEL, "c")]

    result = await check.async_filter_deployments(
        model="group",
        healthy_deployments=deployments,
        messages=None,
        request_kwargs={MID_STREAM_CONTINUATION_KWARG: MID_STREAM_CONTINUATION_MARKER},
    )
    assert result == []


@pytest.mark.asyncio
async def test_filter_drops_deployments_whose_own_params_constrain_the_continuation():
    check = ContinuationPrefillDeploymentCheck()
    plain = _deployment(PREFILL_MODEL, "plain")
    constrained = [
        {"litellm_params": {"model": PREFILL_MODEL, **params}, "model_info": {"id": dep_id}}
        for dep_id, params in (
            ("json", {"response_format": {"type": "json_object"}}),
            ("two_choices", {"extra_body": {"n": 2}}),
            ("forced_tool", {"tool_choice": {"type": "function", "function": {"name": "f"}}}),
            ("merged_reasoning", {"merge_reasoning_content_in_choices": True}),
        )
    ]

    result = await check.async_filter_deployments(
        model="group",
        healthy_deployments=[plain, *constrained],
        messages=None,
        request_kwargs={MID_STREAM_CONTINUATION_KWARG: MID_STREAM_CONTINUATION_MARKER},
    )
    assert [d["model_info"]["id"] for d in result] == ["plain"]


def _text_chunk(text: str) -> litellm.ModelResponseStream:
    return litellm.ModelResponseStream(choices=[{"index": 0, "delta": {"content": text}}])


@pytest.mark.parametrize(
    "prefill_content,fallback_texts,expected_texts",
    [
        ("Hello", ("", "Hel", "lo wor", "ld"), ("", "", " wor", "ld")),
        ("Hello", ("Hello",), ("",)),
        ("Hello", (" world",), (" world",)),
        ("Hello", ("Hel", "p me"), ("Hel", "p me")),
        ("Hello", ("Hel",), ("Hel",)),
        ([{"type": "text", "text": "Sure: "}, {"type": "text", "text": "Hello"}], ("Sure: Hello", " world"), ("", " world")),
    ],
    ids=[
        "echo_split_across_chunks_is_dropped",
        "exact_echo_then_end_is_dropped",
        "continuation_without_echo_passes_through",
        "near_echo_is_kept_in_order",
        "stream_ending_mid_match_is_kept",
        "content_block_prefill_echo_is_dropped",
    ],
)
@pytest.mark.asyncio
async def test_continuation_output_drops_only_a_full_echo_of_the_prefill(
    prefill_content, fallback_texts, expected_texts
):
    output = ContinuationOutput(
        "Hello", {"role": "assistant", "content": prefill_content, "prefix": True}, "group", ()
    )

    async def fallback_stream():
        for text in fallback_texts:
            yield _text_chunk(text)

    released = [chunk async for chunk in output.splice(fallback_stream())]

    assert tuple(chunk.choices[0].delta.content for chunk in released) == expected_texts


@pytest.mark.parametrize(
    "prefill,fallback_texts,expected_texts",
    [
        ("Transformers use ", ("self-attention",), ("self-attention",)),
        ("Hello ", ("Hello ", " world"), ("", "world")),
        ("Hello \n", ("\nworld",), ("\nworld",)),
        ("A.\n\n", ("\n Hi", "\nthere"), (" Hi", "\nthere")),
        ("\n\n", ("\nHi",), ("\nHi",)),
        ("\nHello\n", ("\n",), ("",)),
    ],
    ids=[
        "continuation_without_whitespace_passes_through",
        "echo_then_re_emitted_space_are_both_dropped",
        "different_whitespace_is_kept",
        "seam_ends_at_the_first_divergence",
        "whitespace_only_prefill_has_no_seam",
        "re_emitted_whitespace_at_stream_end_is_dropped",
    ],
)
@pytest.mark.asyncio
async def test_continuation_output_drops_only_whitespace_the_client_already_has(
    prefill, fallback_texts, expected_texts
):
    output = ContinuationOutput(prefill, {"role": "assistant", "content": prefill, "prefix": True}, "group", ())

    async def fallback_stream():
        for text in fallback_texts:
            yield _text_chunk(text)

    released = [chunk async for chunk in output.splice(fallback_stream())]

    assert tuple(chunk.choices[0].delta.content for chunk in released) == expected_texts


@pytest.mark.parametrize(
    "request_kwargs,expected",
    [
        ({"max_tokens": 100}, {"max_tokens": 70}),
        ({"max_tokens": 100, "max_completion_tokens": 40}, {"max_tokens": 70, "max_completion_tokens": 10}),
        ({}, {}),
        (
            {"max_tokens": 100, "extra_body": {"max_tokens": 50, "top_k": 3}},
            {"max_tokens": 70, "extra_body": {"max_tokens": 20, "top_k": 3}},
        ),
        ({"extra_body": {"max_completion_tokens": 45}}, {"extra_body": {"max_completion_tokens": 15}}),
        ({"extra_body": {"top_k": 3}}, {}),
    ],
    ids=["max_tokens", "both_ceilings", "no_ceiling", "extra_body_max_tokens", "extra_body_only", "extra_body_no_ceiling"],
)
def test_continuation_output_ceilings_take_the_emitted_tokens_off_every_ceiling_the_request_sends(
    request_kwargs, expected
):
    assert continuation_output_ceilings(request_kwargs, 30) == expected


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"max_tokens": 30},
        {"max_tokens": 20},
        {"max_tokens": 100, "max_completion_tokens": 30},
        {"max_tokens": 100, "extra_body": {"max_tokens": 30}},
    ],
    ids=["max_tokens_met", "max_tokens_exceeded", "one_of_two_met", "extra_body_met"],
)
def test_continuation_output_ceilings_decline_once_a_ceiling_has_no_tokens_left(request_kwargs):
    assert continuation_output_ceilings(request_kwargs, 30) is None
