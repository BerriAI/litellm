import json
from pathlib import Path

import pytest

import litellm
from litellm import ModelResponse, Usage

# (model, input, cache_read, cache_write, output) in USD per token, from
# https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
PRICING = (
    ("github_copilot/claude-haiku-4.5", 1e-06, 1e-07, 1.25e-06, 5e-06),
    ("github_copilot/claude-opus-5", 5e-06, 5e-07, 6.25e-06, 2.5e-05),
    ("github_copilot/gpt-5.6-luna", 2e-07, 2e-08, 2.5e-07, 1.2e-06),
    ("github_copilot/gpt-5-mini", 2.5e-07, 2.5e-08, None, 2e-06),
)


def _load_cost_map(filename: str = "model_prices_and_context_window.json") -> dict:
    with open(Path(__file__).parents[2] / filename) as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def _local_cost_map(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", _load_cost_map())


def _response(model: str, prompt_tokens: int, completion_tokens: int) -> ModelResponse:
    response = ModelResponse(model=model.split("/", 1)[1])
    response.usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return response


@pytest.mark.parametrize(("model", "input_cost", "cache_read", "cache_write", "output_cost"), PRICING)
def test_github_copilot_published_rates(model, input_cost, cache_read, cache_write, output_cost):
    entry = _load_cost_map()[model]

    assert entry["input_cost_per_token"] == input_cost
    assert entry["cache_read_input_token_cost"] == cache_read
    assert entry.get("cache_creation_input_token_cost") == cache_write
    assert entry["output_cost_per_token"] == output_cost


@pytest.mark.parametrize(("model", "input_cost", "cache_read", "cache_write", "output_cost"), PRICING)
def test_github_copilot_request_is_not_free(model, input_cost, cache_read, cache_write, output_cost):
    cost = litellm.completion_cost(
        completion_response=_response(model, 1000, 100),
        model=model,
        custom_llm_provider="github_copilot",
    )

    assert cost == pytest.approx(1000 * input_cost + 100 * output_cost)


def test_github_copilot_long_context_tier():
    cost = litellm.completion_cost(
        completion_response=_response("github_copilot/gpt-5.6-luna", 300_000, 1000),
        model="github_copilot/gpt-5.6-luna",
        custom_llm_provider="github_copilot",
    )

    assert cost == pytest.approx(300_000 * 4e-07 + 1000 * 1.8e-06)


def test_github_copilot_backup_matches_main():
    main_cost = _load_cost_map()
    backup_cost = _load_cost_map("litellm/model_prices_and_context_window_backup.json")
    copilot = [k for k in main_cost if k.startswith("github_copilot/")]

    assert {k: main_cost[k] for k in copilot} == {k: backup_cost.get(k) for k in copilot}
