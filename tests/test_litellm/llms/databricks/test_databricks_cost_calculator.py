import json
from decimal import Decimal
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.llms.databricks.cost_calculator import cost_per_token
from litellm.types.utils import ModelInfo, Usage

REPO_ROOT: Final = Path(__file__).parents[4]
MAIN_PRICES: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PRICES: Final = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"
NEW_MODELS: Final = (
    "databricks/databricks-claude-opus-4-7",
    "databricks/databricks-claude-opus-4-8",
    "databricks/databricks-claude-opus-5",
    "databricks/databricks-claude-sonnet-5",
    "databricks/databricks-claude-fable-5",
)

DOLLARS_PER_DBU: Final = Decimal("0.070")
PRICE_FIELDS: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_creation_input_token_cost",
    "cache_read_input_token_cost",
)
PUBLISHED_DBU_PER_MILLION: Final = {
    "databricks/databricks-claude-opus-4-7": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/databricks-claude-opus-4-8": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/databricks-claude-opus-5": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/databricks-claude-sonnet-5": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/databricks-claude-fable-5": ("142.858", "714.286", "178.572", "14.286"),
}


def _model_info(model: str) -> ModelInfo:
    return litellm.get_model_info(model=model, custom_llm_provider="databricks")


def _dollars_per_token(dbu_per_million: str) -> float:
    return float(Decimal(dbu_per_million) * DOLLARS_PER_DBU / Decimal(10) ** 6)


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


@pytest.mark.parametrize("model", NEW_MODELS)
def test_new_models_price_at_published_dbu_rates(local_model_cost_map: None, model: str) -> None:
    info: Final = _model_info(model)

    for field, dbu_per_million in zip(PRICE_FIELDS, PUBLISHED_DBU_PER_MILLION[model]):
        assert info[field] == _dollars_per_token(dbu_per_million), field


@pytest.mark.parametrize("model", NEW_MODELS)
def test_new_models_carry_cache_pricing(local_model_cost_map: None, model: str) -> None:
    info: Final = _model_info(model)

    assert info["input_cost_per_token"] > 0
    assert info["output_cost_per_token"] > 0
    assert info["cache_creation_input_token_cost"] > info["input_cost_per_token"]
    assert info["cache_read_input_token_cost"] < info["input_cost_per_token"]
    assert info["supports_prompt_caching"] is True


def test_every_priced_databricks_model_declares_cache_rates(local_model_cost_map: None) -> None:
    undeclared: Final = [
        model
        for model, info in litellm.model_cost.items()
        if model.startswith("databricks/")
        and info.get("input_cost_per_token") is not None
        and info.get("cache_read_input_token_cost") is None
    ]

    assert undeclared == []


def test_models_without_a_cache_discount_bill_cache_tokens_at_the_input_rate(
    local_model_cost_map: None,
) -> None:
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    info: Final = _model_info(model)
    usage: Final = Usage(
        prompt_tokens=10000,
        completion_tokens=100,
        total_tokens=10100,
        cache_read_input_tokens=8000,
    )

    prompt_cost, _ = cost_per_token(model=model, usage=usage)

    assert info["cache_read_input_token_cost"] == info["input_cost_per_token"]
    assert info["cache_creation_input_token_cost"] == info["input_cost_per_token"]
    assert prompt_cost == pytest.approx(10000 * info["input_cost_per_token"])
    assert prompt_cost > 8000 * info["input_cost_per_token"]


@pytest.mark.parametrize("model", NEW_MODELS)
def test_backup_price_map_matches_main(model: str) -> None:
    main_cost: Final = json.loads(MAIN_PRICES.read_text())
    backup_cost: Final = json.loads(BACKUP_PRICES.read_text())

    assert model in main_cost
    assert model in backup_cost
    assert backup_cost[model] == main_cost[model]


def test_sonnet_5_ships_standard_rates_not_introductory(local_model_cost_map: None) -> None:
    sonnet_5: Final = _model_info("databricks/databricks-claude-sonnet-5")
    sonnet_4_6: Final = _model_info("databricks/databricks-claude-sonnet-4-6")

    for field in PRICE_FIELDS:
        assert sonnet_5[field] == pytest.approx(sonnet_4_6[field]), field


@pytest.mark.parametrize(
    "model",
    [
        "databricks/databricks-gemini-2-5-pro",
        "databricks/databricks-gemini-2-5-flash",
    ],
)
def test_entries_priced_at_an_older_vintage_keep_cache_rates_tied_to_their_own_input(
    local_model_cost_map: None,
    model: str,
) -> None:
    info: Final = _model_info(model)

    assert info["cache_creation_input_token_cost"] == pytest.approx(info["input_cost_per_token"])
    assert info["cache_read_input_token_cost"] == pytest.approx(0.1 * info["input_cost_per_token"])
