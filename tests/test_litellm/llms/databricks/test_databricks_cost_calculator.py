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
    "databricks/databricks-claude-fable-5-1",
    "databricks/databricks-gpt-5-6-sol",
    "databricks/databricks-gpt-5-6-terra",
    "databricks/databricks-gpt-5-6-luna",
)

DOLLARS_PER_DBU: Final = Decimal("0.070")
PRICE_FIELDS: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_creation_input_token_cost",
    "cache_read_input_token_cost",
)
PROMOTIONAL_DISCOUNT: Final = 0.80
PROMOTION_EXPIRES: Final = "2027-01-31"
ENTRIES_STORING_PROMOTIONAL_RATE: Final = (
    "databricks/databricks-gemini-2-5-pro",
    "databricks/databricks-gemini-2-5-flash",
)
ENTRIES_STORING_LIST_RATE_DESPITE_PROMOTION: Final = (
    "databricks/databricks-gemini-3-6-flash",
    "databricks/databricks-gemini-3-5-flash",
    "databricks/databricks-gemini-3-5-flash-lite",
    "databricks/databricks-grok-4-6",
    "databricks/databricks-gemini-3-1-pro",
    "databricks/databricks-gemini-3-pro",
    "databricks/databricks-gemini-3-flash",
    "databricks/databricks-gemini-3-1-flash-lite",
)
CACHE_FIELDS: Final = ("cache_creation_input_token_cost", "cache_read_input_token_cost")


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


def test_every_priced_databricks_model_declares_cache_rates(local_model_cost_map: None) -> None:
    undeclared: Final = [
        model
        for model, info in litellm.model_cost.items()
        if model.startswith("databricks/")
        and info.get("input_cost_per_token") is not None
        and any(info.get(field) is None for field in CACHE_FIELDS)
    ]

    assert undeclared == []


@pytest.mark.parametrize("model", NEW_MODELS)
def test_backup_price_map_matches_main(model: str) -> None:
    main_cost: Final = json.loads(MAIN_PRICES.read_text())
    backup_cost: Final = json.loads(BACKUP_PRICES.read_text())

    assert model in main_cost
    assert model in backup_cost
    assert backup_cost[model] == main_cost[model]
