import math
from typing import Generator

import pytest

import litellm
from litellm.llms.volcengine.cost_calculator import (
    cost_per_token as volcengine_cost_per_token,
)
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    yield
    litellm.model_cost = original_model_cost
    litellm.get_model_info.cache_clear()


@pytest.mark.parametrize(
    ("model", "prompt_tokens", "completion_tokens", "tier_index"),
    [
        ("doubao-seed-2-0-lite-260215", 32000, 1000, 0),
        ("doubao-seed-2-0-lite-260215", 32001, 1000, 1),
        ("doubao-seed-2-0-pro-260215", 128000, 1000, 1),
        ("doubao-seed-2-0-pro-260215", 128001, 1000, 2),
    ],
)
def test_seed_2_uses_one_tier_selected_by_total_input_length(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    tier_index: int,
) -> None:
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    prompt_cost, completion_cost = volcengine_cost_per_token(model=model, usage=usage)

    model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider="volcengine",
    )
    tier = model_info["tiered_pricing"][tier_index]

    assert math.isclose(
        prompt_cost,
        prompt_tokens * tier["input_cost_per_token"],
        rel_tol=1e-12,
    )
    assert math.isclose(
        completion_cost,
        completion_tokens * tier["output_cost_per_token"],
        rel_tol=1e-12,
    )


def test_seed_1_8_applies_short_output_discount_only_through_200_tokens() -> None:
    short_usage = Usage(prompt_tokens=10000, completion_tokens=200)
    long_usage = Usage(prompt_tokens=10000, completion_tokens=201)

    _, short_output_cost = volcengine_cost_per_token(
        model="doubao-seed-1-8-251228",
        usage=short_usage,
    )
    _, long_output_cost = volcengine_cost_per_token(
        model="doubao-seed-1-8-251228",
        usage=long_usage,
    )

    model_info = litellm.get_model_info(
        model="doubao-seed-1-8-251228",
        custom_llm_provider="volcengine",
    )
    first_tier = model_info["tiered_pricing"][0]
    assert math.isclose(
        short_output_cost,
        200 * first_tier["output_cost_per_token"],
        rel_tol=1e-12,
    )
    assert math.isclose(
        long_output_cost,
        201 * first_tier["output_cost_per_token_above_200_tokens"],
        rel_tol=1e-12,
    )


@pytest.mark.parametrize(
    ("prompt_tokens", "tier_index"),
    [
        (32000, 0),
        (32001, 1),
        (128000, 1),
        (128001, 2),
    ],
)
def test_seed_1_8_input_and_output_rates_follow_input_tier(
    prompt_tokens: int,
    tier_index: int,
) -> None:
    completion_tokens = 1000
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    prompt_cost, completion_cost = volcengine_cost_per_token(
        model="doubao-seed-1-8-251228",
        usage=usage,
    )

    model_info = litellm.get_model_info(
        model="doubao-seed-1-8-251228",
        custom_llm_provider="volcengine",
    )
    tier = model_info["tiered_pricing"][tier_index]
    output_key = "output_cost_per_token_above_200_tokens" if tier_index == 0 else "output_cost_per_token"

    assert math.isclose(
        prompt_cost,
        prompt_tokens * tier["input_cost_per_token"],
        rel_tol=1e-12,
    )
    assert math.isclose(
        completion_cost,
        completion_tokens * tier[output_key],
        rel_tol=1e-12,
    )


def test_seed_1_8_cached_tokens_use_cache_rate_but_still_select_input_tier() -> None:
    usage = Usage(
        prompt_tokens=50000,
        completion_tokens=1000,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=10000),
    )
    prompt_cost, _ = volcengine_cost_per_token(
        model="doubao-seed-1-8-251228",
        usage=usage,
    )

    model_info = litellm.get_model_info(
        model="doubao-seed-1-8-251228",
        custom_llm_provider="volcengine",
    )
    second_tier = model_info["tiered_pricing"][1]
    expected = (40000 * second_tier["input_cost_per_token"]) + (10000 * model_info["cache_read_input_token_cost"])
    assert math.isclose(prompt_cost, expected, rel_tol=1e-12)


def test_output_only_usage_uses_first_pricing_tier() -> None:
    usage = Usage(prompt_tokens=0, completion_tokens=100)
    _, completion_cost = volcengine_cost_per_token(
        model="doubao-seed-2-0-lite-260215",
        usage=usage,
    )

    model_info = litellm.get_model_info(
        model="doubao-seed-2-0-lite-260215",
        custom_llm_provider="volcengine",
    )
    first_tier = model_info["tiered_pricing"][0]
    assert math.isclose(
        completion_cost,
        100 * first_tier["output_cost_per_token"],
        rel_tol=1e-12,
    )


def test_flat_pricing_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 2e-7,
    }
    monkeypatch.setattr(
        "litellm.llms.volcengine.cost_calculator.get_model_info",
        lambda **_: model_info,
    )
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=20),
    )

    prompt_cost, completion_cost = volcengine_cost_per_token(
        model="flat-priced-model",
        usage=usage,
    )

    assert math.isclose(prompt_cost, (80 * 1e-6) + (20 * 2e-7), rel_tol=1e-12)
    assert math.isclose(completion_cost, 50 * 2e-6, rel_tol=1e-12)


def test_top_level_cost_calculator_routes_volcengine_to_tiered_calculator() -> None:
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="doubao-seed-2-0-lite-260215",
        prompt_tokens=50000,
        completion_tokens=1000,
        custom_llm_provider="volcengine",
    )
    model_info = litellm.get_model_info(
        model="doubao-seed-2-0-lite-260215",
        custom_llm_provider="volcengine",
    )
    second_tier = model_info["tiered_pricing"][1]

    assert math.isclose(
        prompt_cost,
        50000 * second_tier["input_cost_per_token"],
        rel_tol=1e-12,
    )
    assert math.isclose(
        completion_cost,
        1000 * second_tier["output_cost_per_token"],
        rel_tol=1e-12,
    )
