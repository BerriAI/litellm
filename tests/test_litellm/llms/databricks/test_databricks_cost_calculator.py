from typing import Final

import pytest

import litellm
from litellm.llms.databricks.cost_calculator import cost_per_token
from litellm.types.utils import Usage


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force get_model_info to resolve against the in-repo cost map instead of the
    remote one fetched at import time, which still carries the pre-merge pricing."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def _model_info(model: str) -> dict:
    return litellm.get_model_info(model=model, custom_llm_provider="databricks")


def test_cached_tokens_are_billed_at_cache_rates(local_model_cost_map):
    """Cache reads and cache writes bill at the model's cache rates, not the input rate"""
    model: Final = "databricks/databricks-claude-opus-5"
    info: Final = _model_info(model)
    usage: Final = Usage(
        prompt_tokens=11000,
        completion_tokens=500,
        total_tokens=11500,
        cache_creation_input_tokens=2000,
        cache_read_input_tokens=8000,
    )

    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    assert prompt_cost == pytest.approx(
        1000 * info["input_cost_per_token"]
        + 2000 * info["cache_creation_input_token_cost"]
        + 8000 * info["cache_read_input_token_cost"]
    )
    assert completion_cost == pytest.approx(500 * info["output_cost_per_token"])
    assert prompt_cost < 11000 * info["input_cost_per_token"]


def test_uncached_request_bills_every_prompt_token_at_the_input_rate(local_model_cost_map):
    model: Final = "databricks/databricks-claude-sonnet-5"
    info: Final = _model_info(model)
    usage: Final = Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200)

    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    assert prompt_cost == pytest.approx(1000 * info["input_cost_per_token"])
    assert completion_cost == pytest.approx(200 * info["output_cost_per_token"])


def test_legacy_endpoint_names_still_resolve(local_model_cost_map):
    """Endpoint names that predate the `databricks-` prefixed registry keys keep their pricing"""
    info: Final = _model_info("databricks/databricks-mixtral-8x7b-instruct")
    usage: Final = Usage(prompt_tokens=100, completion_tokens=100, total_tokens=200)

    prompt_cost, completion_cost = cost_per_token(model="databricks/mixtral-8x7b-instruct-v0.1", usage=usage)

    assert prompt_cost == pytest.approx(100 * info["input_cost_per_token"])
    assert completion_cost == pytest.approx(100 * info["output_cost_per_token"])
