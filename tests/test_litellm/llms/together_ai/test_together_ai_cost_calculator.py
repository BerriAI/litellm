"""Cost tests for together_ai: cached-input billing and the published rates it bills against."""

import pytest

from litellm.types.utils import Choices, Message, ModelResponse, Usage

CACHED_MODEL = "together_ai/moonshotai/Kimi-K3"
INPUT_RATE = 3e-06
CACHED_RATE = 3e-07
OUTPUT_RATE = 1.5e-05


def response_with_usage(usage: Usage, model: str = CACHED_MODEL) -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-test",
        object="chat.completion",
        created=1,
        model=model,
        choices=[Choices(index=0, message=Message(content="hi", role="assistant"), finish_reason="stop")],
        usage=usage,
    )


@pytest.mark.parametrize(
    "usage_kwargs",
    [
        {"cached_tokens": 800},
        {"prompt_tokens_details": {"cached_tokens": 800}},
    ],
    ids=["flat-cached-tokens", "nested-prompt-tokens-details"],
)
def test_cached_prefix_bills_at_the_cached_rate(local_model_cost_map, usage_kwargs):
    """Together reports cache hits in two shapes, and both have to bill at the cached rate.

    Reasoning models nest the count under `prompt_tokens_details`; some non-reasoning models
    return `cached_tokens` flat on `usage`. The flat shape used to be ignored, so a warm
    prefix was billed at the full input rate.
    """
    usage = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, **usage_kwargs)

    cost = local_model_cost_map.completion_cost(
        completion_response=response_with_usage(usage),
        model=CACHED_MODEL,
        custom_llm_provider="together_ai",
    )

    expected = 200 * INPUT_RATE + 800 * CACHED_RATE + 100 * OUTPUT_RATE
    assert cost == pytest.approx(expected)
    assert cost < 1000 * INPUT_RATE + 100 * OUTPUT_RATE


def test_uncached_call_bills_every_prompt_token_at_the_input_rate(local_model_cost_map):
    usage = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)

    cost = local_model_cost_map.completion_cost(
        completion_response=response_with_usage(usage),
        model=CACHED_MODEL,
        custom_llm_provider="together_ai",
    )

    assert cost == pytest.approx(1000 * INPUT_RATE + 100 * OUTPUT_RATE)


@pytest.mark.parametrize(
    "model",
    [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ],
    ids=["bare", "prefixed"],
)
def test_published_rate_beats_the_size_bucket(local_model_cost_map, model):
    """Together's per-token rates are size buckets only for ids the map does not price.

    Applied unconditionally the buckets shadow every entry whose name carries a parameter
    count, so this model billed at the 41.1b-80b bucket's $0.90/1M instead of its own
    published $1.04/1M, and a cached-input rate on such an entry could never apply.
    """
    usage = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)

    cost = local_model_cost_map.completion_cost(
        completion_response=response_with_usage(usage, model=model),
        model=model,
        custom_llm_provider="together_ai",
    )

    assert cost == pytest.approx(1000 * 1.04e-06 + 100 * 1.04e-06)


def test_cached_rate_applies_to_a_model_the_bucket_would_have_shadowed(local_model_cost_map):
    usage = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, cached_tokens=800)

    cost = local_model_cost_map.completion_cost(
        completion_response=response_with_usage(usage, model="meta-models/Muse-Glimmer-30B"),
        model="meta-models/Muse-Glimmer-30B",
        custom_llm_provider="together_ai",
    )

    assert cost == pytest.approx(200 * 3.5e-07 + 800 * 4e-08 + 100 * 1.5e-06)


@pytest.mark.parametrize(
    "model, bucket_rate",
    [
        # mapped, but the entry carries no rate, so the bucket is all there is
        ("Qwen/Qwen2.5-72B-Instruct-Turbo", 9e-07),
        # never mapped at all
        ("some-org/Mystery-13B-Instruct", 3e-07),
    ],
)
def test_unpriced_models_still_fall_back_to_the_size_bucket(local_model_cost_map, model, bucket_rate):
    usage = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)

    cost = local_model_cost_map.completion_cost(
        completion_response=response_with_usage(usage, model=model),
        model=model,
        custom_llm_provider="together_ai",
    )

    assert cost == pytest.approx(1100 * bucket_rate)


def test_qwen3_235b_instruct_output_rate(local_model_cost_map):
    """Regression: this entry billed output at $6.00/1M, 10x the published rate."""
    entry = local_model_cost_map.model_cost["together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"]

    assert entry["input_cost_per_token"] == 2e-07
    assert entry["output_cost_per_token"] == 6e-07


def test_llama_3_3_70b_turbo_matches_the_published_rate(local_model_cost_map):
    entry = local_model_cost_map.model_cost["together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"]

    assert entry["input_cost_per_token"] == 1.04e-06
    assert entry["output_cost_per_token"] == 1.04e-06
    assert entry["max_input_tokens"] == 131072


@pytest.mark.parametrize(
    "model, input_rate, cached_rate, output_rate, context_window",
    [
        ("together_ai/moonshotai/Kimi-K3", 3e-06, 3e-07, 1.5e-05, 1048576),
        ("together_ai/zai-org/GLM-5.2", 1.4e-06, 2.6e-07, 4.4e-06, 512000),
        ("together_ai/deepseek-ai/DeepSeek-V4-Pro", 1.74e-06, 2e-07, 3.48e-06, 512000),
        ("together_ai/MiniMaxAI/MiniMax-M3", 3e-07, 6e-08, 1.2e-06, 524288),
        ("together_ai/Qwen/Qwen3.5-9B", 1.7e-07, None, 2.5e-07, 262144),
    ],
)
def test_current_catalog_rates(local_model_cost_map, model, input_rate, cached_rate, output_rate, context_window):
    entry = local_model_cost_map.model_cost[model]

    assert entry["litellm_provider"] == "together_ai"
    assert entry["mode"] == "chat"
    assert entry["input_cost_per_token"] == input_rate
    assert entry["output_cost_per_token"] == output_rate
    assert entry["max_input_tokens"] == context_window
    assert entry.get("cache_read_input_token_cost") == cached_rate
