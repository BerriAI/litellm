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
    "databricks/databricks-claude-fable-5": ("142.858", "714.286", "178.572", "14.286"),
    "databricks/databricks-claude-opus-5": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/databricks-claude-opus-4-8": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/databricks-claude-opus-4-7": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/databricks-claude-opus-4-6": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/databricks-claude-opus-4-5": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/databricks-claude-opus-4-1": ("214.286", "1071.429", "267.857", "21.429"),
    "databricks/databricks-claude-opus-4": ("214.286", "1071.429", "267.857", "21.429"),
    "databricks/databricks-claude-sonnet-5": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/databricks-claude-sonnet-4-6": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/databricks-claude-sonnet-4-5": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/databricks-claude-sonnet-4-1": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/databricks-claude-sonnet-4": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/databricks-claude-3-7-sonnet": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/databricks-claude-haiku-4-5": ("14.286", "71.429", "17.857", "1.429"),
    "databricks/databricks-gpt-5": ("17.857", "142.857", "17.857", "1.786"),
    "databricks/databricks-gpt-5-1": ("17.857", "142.857", "17.857", "1.786"),
    "databricks/databricks-gpt-5-1-codex-max": ("17.857", "142.857", "17.857", "1.786"),
    "databricks/databricks-gpt-5-1-codex-mini": ("3.571", "28.571", "3.571", "0.357"),
    "databricks/databricks-gpt-5-mini": ("3.571", "28.571", "3.571", "0.357"),
    "databricks/databricks-gpt-5-nano": ("0.714", "5.714", "0.714", "0.071"),
    "databricks/databricks-gpt-5-2": ("25.000", "200.000", "25.000", "2.500"),
    "databricks/databricks-gpt-5-2-codex": ("25.000", "200.000", "25.000", "2.500"),
    "databricks/databricks-gpt-5-3-codex": ("25.000", "200.000", "25.000", "2.500"),
    "databricks/databricks-gpt-5-4": ("35.714", "214.286", "35.714", "3.571"),
    "databricks/databricks-gpt-5-4-mini": ("10.714", "64.286", "10.714", "1.071"),
    "databricks/databricks-gpt-5-4-nano": ("2.857", "17.857", "2.857", "0.286"),
    "databricks/databricks-gemini-3-1-pro": ("35.714", "214.286", "35.714", "3.571"),
    "databricks/databricks-gemini-3-pro": ("35.714", "214.286", "35.714", "3.571"),
    "databricks/databricks-gemini-3-flash": ("8.929", "53.571", "8.929", "0.893"),
    "databricks/databricks-gemini-3-1-flash-lite": ("4.464", "26.786", "4.464", "0.446"),
    "databricks/databricks-gemini-2-5-pro": ("22.321", "178.571", "22.321", "2.232"),
    "databricks/databricks-gemini-2-5-flash": ("5.357", "44.643", "5.357", "0.536"),
}
PROMOTIONAL_DISCOUNT: Final = 0.80
PROMOTION_EXPIRES: Final = "2027-01-31"
ENTRIES_STORING_PROMOTIONAL_RATE: Final = (
    "databricks/databricks-gemini-2-5-pro",
    "databricks/databricks-gemini-2-5-flash",
)
ENTRIES_STORING_LIST_RATE_DESPITE_PROMOTION: Final = (
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


@pytest.mark.parametrize("model", NEW_MODELS)
def test_new_models_price_at_published_dbu_rates(local_model_cost_map: None, model: str) -> None:
    info: Final = _model_info(model)

    for field, dbu_per_million in zip(PRICE_FIELDS, PUBLISHED_DBU_PER_MILLION[model]):
        assert info[field] == _dollars_per_token(dbu_per_million), field


@pytest.mark.parametrize("model", sorted(set(PUBLISHED_DBU_PER_MILLION) - set(ENTRIES_STORING_PROMOTIONAL_RATE)))
def test_cache_rates_derive_from_published_cache_dbu(local_model_cost_map: None, model: str) -> None:
    info: Final = _model_info(model)
    cache_dbu_per_million: Final = PUBLISHED_DBU_PER_MILLION[model][2:]

    for field, dbu_per_million in zip(CACHE_FIELDS, cache_dbu_per_million):
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


def test_every_model_without_published_cache_dbu_bills_cache_at_its_own_input_rate(
    local_model_cost_map: None,
) -> None:
    without_published_rates: Final = [
        model
        for model, info in litellm.model_cost.items()
        if model.startswith("databricks/")
        and info.get("input_cost_per_token")
        and model not in PUBLISHED_DBU_PER_MILLION
    ]

    assert len(without_published_rates) == 14
    for model in without_published_rates:
        info = _model_info(model)
        for field in CACHE_FIELDS:
            assert info[field] == pytest.approx(info["input_cost_per_token"]), (model, field)


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


@pytest.mark.parametrize("model", ENTRIES_STORING_PROMOTIONAL_RATE)
def test_entries_storing_the_promotional_rate_price_below_the_published_table(
    local_model_cost_map: None,
    model: str,
) -> None:
    info: Final = _model_info(model)
    input_dbu, output_dbu, _, _ = PUBLISHED_DBU_PER_MILLION[model]
    expiry_hint: Final = f"the gemini promotion expires {PROMOTION_EXPIRES}, after which the list rate applies"

    assert info["input_cost_per_token"] == pytest.approx(
        _dollars_per_token(input_dbu) * PROMOTIONAL_DISCOUNT, rel=2e-4
    ), expiry_hint
    assert info["output_cost_per_token"] == pytest.approx(
        _dollars_per_token(output_dbu) * PROMOTIONAL_DISCOUNT, rel=2e-4
    ), expiry_hint
    assert info["cache_creation_input_token_cost"] == pytest.approx(info["input_cost_per_token"])
    assert info["cache_read_input_token_cost"] == pytest.approx(0.1 * info["input_cost_per_token"])


@pytest.mark.parametrize("model", ENTRIES_STORING_LIST_RATE_DESPITE_PROMOTION)
def test_entries_storing_the_list_rate_bill_above_the_promotional_price(
    local_model_cost_map: None,
    model: str,
) -> None:
    info: Final = _model_info(model)
    input_dbu, _, _, _ = PUBLISHED_DBU_PER_MILLION[model]
    list_rate: Final = _dollars_per_token(input_dbu)

    assert info["input_cost_per_token"] == pytest.approx(list_rate, rel=2e-4), (
        f"{model} moved off the list rate; if it now stores the discount that runs to "
        f"{PROMOTION_EXPIRES}, move it into ENTRIES_STORING_PROMOTIONAL_RATE"
    )
    assert info["cache_creation_input_token_cost"] == pytest.approx(info["input_cost_per_token"])
