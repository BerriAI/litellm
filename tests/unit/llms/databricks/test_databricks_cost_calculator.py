import json
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry
from litellm.llms.databricks.cost_calculator import cost_per_token
from litellm.types.utils import ModelInfo, Usage

REPO_ROOT: Final = Path(__file__).parents[4]
MAIN_PRICES: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PRICES: Final = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"
NEW_MODELS: Final = (
    "databricks/databricks-claude-opus-4-7",
    "databricks/databricks-claude-opus-4-8",
    "databricks/databricks-claude-opus-5",
    "databricks/databricks-claude-opus-5-5",
    "databricks/databricks-claude-sonnet-5",
    "databricks/databricks-claude-fable-5",
    "databricks/databricks-claude-fable-5-1",
    "databricks/databricks-gpt-5-6-sol",
    "databricks/databricks-gpt-5-6-terra",
    "databricks/databricks-gpt-5-6-luna",
)

CACHE_FIELDS: Final = ("cache_creation_input_token_cost", "cache_read_input_token_cost")


def _model_info(model: str) -> ModelInfo:
    return litellm.get_model_info(model=model, custom_llm_provider="databricks")


@pytest.mark.parametrize(
    "model",
    [
        "databricks/databricks-claude-opus-4-8",
        "databricks/databricks-claude-opus-5",
        "databricks/databricks-claude-opus-5-5",
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
        and any(info.get(field) is None for field in CACHE_FIELDS)
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

    assert prompt_cost == pytest.approx(10000 * info["input_cost_per_token"])
    assert prompt_cost > 8000 * info["input_cost_per_token"]


@pytest.mark.parametrize("uncached_tokens", (0, 1000))
def test_every_priced_model_bills_cache_writes_at_its_configured_rate(
    local_model_cost_map: None, uncached_tokens: int
) -> None:
    priced_models: Final = tuple(
        model
        for model, info in litellm.model_cost.items()
        if model.startswith("databricks/")
        and info.get("input_cost_per_token") is not None
        and not (isinstance(info, AliasedCostEntry) and info.names[0] != model)
    )
    assert priced_models
    usage: Final = Usage(
        prompt_tokens=uncached_tokens + 2000,
        completion_tokens=0,
        total_tokens=uncached_tokens + 2000,
        cache_creation_input_tokens=2000,
    )

    for model in priced_models:
        info: Final = _model_info(model)

        prompt_cost, completion_cost = cost_per_token(model=model, usage=usage)

        assert prompt_cost == pytest.approx(
            uncached_tokens * info["input_cost_per_token"] + 2000 * info["cache_creation_input_token_cost"]
        ), model
        assert completion_cost == 0, model


@pytest.mark.parametrize("model", NEW_MODELS)
def test_backup_price_map_matches_main(model: str) -> None:
    main_cost: Final = json.loads(MAIN_PRICES.read_text())
    backup_cost: Final = json.loads(BACKUP_PRICES.read_text())

    assert model in main_cost
    assert model in backup_cost
    assert backup_cost[model] == main_cost[model]


def test_cost_per_token_bills_the_served_priority_tier(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    rates: Final = {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
        "input_cost_per_token_priority": 0.01,
        "output_cost_per_token_priority": 0.02,
        "litellm_provider": "databricks",
        "mode": "chat",
    }
    monkeypatch.setitem(litellm.model_cost, "databricks/dbrx-tiered-test", rates)
    usage: Final = Usage(prompt_tokens=30, completion_tokens=40, total_tokens=70)

    prompt_cost, completion_cost = cost_per_token(
        model="databricks/dbrx-tiered-test", usage=usage, service_tier="priority"
    )
    assert prompt_cost == pytest.approx(30 * 0.01)
    assert completion_cost == pytest.approx(40 * 0.02)

    prompt_cost, completion_cost = cost_per_token(model="databricks/dbrx-tiered-test", usage=usage)
    assert prompt_cost == pytest.approx(30 * 0.001)
    assert completion_cost == pytest.approx(40 * 0.002)


RATES_WITHOUT_A_PRIORITY_FIELD: Final = frozenset({"input_cost_per_image_token"})


def _is_standard_rate_key(key: str) -> bool:
    return (
        ("_cost_per_" in key or key.endswith("_token_cost"))
        and "dbu" not in key
        and key not in RATES_WITHOUT_A_PRIORITY_FIELD
        and not key.endswith(("_priority", "_flex", "_batches"))
    )


def test_every_databricks_priority_row_prices_every_standard_rate_at_the_priority_tier() -> None:
    cost_map: Final = json.loads(MAIN_PRICES.read_text())
    priority_rows: Final = {
        key: row
        for key, row in cost_map.items()
        if key.startswith("databricks/") and "input_cost_per_token_priority" in row
    }

    assert priority_rows
    for key, row in priority_rows.items():
        missing = sorted(f for f in row if _is_standard_rate_key(f) and f"{f}_priority" not in row)
        assert not missing, f"{key} bills these at the standard rate on priority requests: {missing}"
