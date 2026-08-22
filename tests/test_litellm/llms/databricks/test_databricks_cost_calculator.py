from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm.llms.databricks.cost_calculator import cost_per_token
from litellm.types.utils import ModelInfo, Usage


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def _model_info(model: str) -> ModelInfo:
    return litellm.get_model_info(model=model, custom_llm_provider="databricks")


@pytest.mark.parametrize(
    "model",
    [
        "databricks/databricks-claude-opus-4-8",
        "databricks/databricks-claude-opus-5",
        "databricks/databricks-claude-sonnet-5",
    ],
)
def test_cached_tokens_bill_at_cache_rates(local_model_cost_map: None, model: str) -> None:
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


def test_uncached_request_bills_every_prompt_token_at_the_input_rate(local_model_cost_map: None) -> None:
    model: Final = "databricks/databricks-claude-sonnet-5"
    info: Final = _model_info(model)
    usage: Final = Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200)

    prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

    assert prompt_cost == pytest.approx(1000 * info["input_cost_per_token"])
    assert completion_cost == pytest.approx(200 * info["output_cost_per_token"])


def test_legacy_endpoint_names_still_resolve(local_model_cost_map: None) -> None:
    info: Final = _model_info("databricks/databricks-mixtral-8x7b-instruct")
    usage: Final = Usage(prompt_tokens=100, completion_tokens=100, total_tokens=200)

    prompt_cost, completion_cost = cost_per_token(model="databricks/mixtral-8x7b-instruct-v0.1", usage=usage)

    assert prompt_cost == pytest.approx(100 * info["input_cost_per_token"])
    assert completion_cost == pytest.approx(100 * info["output_cost_per_token"])


@pytest.mark.parametrize(
    "model",
    [
        "databricks/databricks-claude-opus-4-7",
        "databricks/databricks-claude-opus-4-8",
        "databricks/databricks-claude-opus-5",
        "databricks/databricks-claude-sonnet-5",
        "databricks/databricks-claude-fable-5",
    ],
)
def test_new_models_carry_cache_pricing(local_model_cost_map: None, model: str) -> None:
    info: Final = _model_info(model)

    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["cache_creation_input_token_cost"] == pytest.approx(1.25 * info["input_cost_per_token"], rel=1e-4)
    assert info["cache_read_input_token_cost"] == pytest.approx(0.1 * info["input_cost_per_token"], rel=1e-4)
    assert info["supports_prompt_caching"] is True


def test_sonnet_5_ships_standard_rates_not_introductory(local_model_cost_map: None) -> None:
    sonnet_5: Final = _model_info("databricks/databricks-claude-sonnet-5")
    sonnet_4_6: Final = _model_info("databricks/databricks-claude-sonnet-4-6")

    assert sonnet_5["input_cost_per_token"] == pytest.approx(sonnet_4_6["input_cost_per_token"])
    assert sonnet_5["output_cost_per_token"] == pytest.approx(sonnet_4_6["output_cost_per_token"])
    assert sonnet_5["cache_creation_input_token_cost"] == pytest.approx(sonnet_4_6["cache_creation_input_token_cost"])
    assert sonnet_5["cache_read_input_token_cost"] == pytest.approx(sonnet_4_6["cache_read_input_token_cost"])
