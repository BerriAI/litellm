import json
from decimal import Decimal
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.llms.databricks.cost_calculator import _DATABRICKS_LEGACY_TO_CANONICAL, cost_per_token
from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
from litellm.types.utils import ModelInfo, Usage

REPO_ROOT: Final = Path(__file__).parents[4]
MAIN_PRICES: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PRICES: Final = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"
NEW_MODELS: Final = (
    "databricks/system.ai.claude-opus-4-7",
    "databricks/system.ai.claude-opus-4-8",
    "databricks/system.ai.claude-opus-5",
    "databricks/system.ai.claude-sonnet-5",
    "databricks/system.ai.claude-fable-5",
    "databricks/system.ai.claude-fable-5-1",
    "databricks/system.ai.gpt-5-6-sol",
    "databricks/databricks-gpt-5-6-terra",
    "databricks/system.ai.gpt-5-6-luna",
)

DOLLARS_PER_DBU: Final = Decimal("0.070")
PRICE_FIELDS: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_creation_input_token_cost",
    "cache_read_input_token_cost",
)
PUBLISHED_DBU_PER_MILLION: Final = {
    "databricks/system.ai.claude-fable-5-1": ("142.858", "714.286", "178.572", "3.572"),
    "databricks/system.ai.claude-fable-5": ("142.858", "714.286", "178.572", "14.286"),
    "databricks/system.ai.claude-opus-5": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/system.ai.claude-opus-4-8": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/system.ai.claude-opus-4-7": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/system.ai.claude-opus-4-6": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/databricks-claude-opus-4-5": ("71.429", "357.143", "89.286", "7.143"),
    "databricks/databricks-claude-opus-4-1": ("214.286", "1071.429", "267.857", "21.429"),
    "databricks/databricks-claude-opus-4": ("214.286", "1071.429", "267.857", "21.429"),
    "databricks/system.ai.claude-sonnet-5": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/system.ai.claude-sonnet-4-6": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/system.ai.claude-sonnet-4-5": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/databricks-claude-sonnet-4-1": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/system.ai.claude-sonnet-4": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/databricks-claude-3-7-sonnet": ("42.857", "214.286", "53.571", "4.286"),
    "databricks/system.ai.claude-haiku-4-5": ("14.286", "71.429", "17.857", "1.429"),
    "databricks/system.ai.gpt-5": ("17.857", "142.857", "17.857", "1.786"),
    "databricks/system.ai.gpt-5-1": ("17.857", "142.857", "17.857", "1.786"),
    "databricks/databricks-gpt-5-1-codex-max": ("17.857", "142.857", "17.857", "1.786"),
    "databricks/databricks-gpt-5-1-codex-mini": ("3.571", "28.571", "3.571", "0.357"),
    "databricks/system.ai.gpt-5-mini": ("3.571", "28.571", "3.571", "0.357"),
    "databricks/system.ai.gpt-5-nano": ("0.714", "5.714", "0.714", "0.071"),
    "databricks/system.ai.gpt-5-2": ("25.000", "200.000", "25.000", "2.500"),
    "databricks/databricks-gpt-5-2-codex": ("25.000", "200.000", "25.000", "2.500"),
    "databricks/databricks-gpt-5-3-codex": ("25.000", "200.000", "25.000", "2.500"),
    "databricks/system.ai.gpt-5-6-sol": ("57.143", "285.714", "71.429", "5.714"),
    "databricks/databricks-gpt-5-6-terra": ("35.714", "214.286", "44.643", "3.571"),
    "databricks/system.ai.gpt-5-6-luna": ("14.286", "85.714", "17.857", "1.429"),
    "databricks/system.ai.gpt-5-5": ("71.429", "428.571", "71.429", "7.143"),
    "databricks/databricks-gpt-5-5-pro": ("428.571", "2571.429", "428.571", "428.571"),
    "databricks/system.ai.gpt-5-4": ("35.714", "214.286", "35.714", "3.571"),
    "databricks/system.ai.gpt-5-4-mini": ("10.714", "64.286", "10.714", "1.071"),
    "databricks/system.ai.gpt-5-4-nano": ("2.857", "17.857", "2.857", "0.286"),
    "databricks/system.ai.gemini-3-6-flash": ("26.786", "133.929", "26.786", "2.679"),
    "databricks/system.ai.gemini-3-5-flash": ("26.786", "160.714", "26.786", "2.679"),
    "databricks/databricks-gemini-3-5-flash-lite": ("5.357", "44.643", "5.357", "0.536"),
    "databricks/system.ai.gemini-3-1-pro": ("35.714", "214.286", "35.714", "3.571"),
    "databricks/databricks-gemini-3-pro": ("35.714", "214.286", "35.714", "3.571"),
    "databricks/system.ai.gemini-3-flash": ("8.929", "53.571", "8.929", "0.893"),
    "databricks/system.ai.gemini-3-1-flash-lite": ("4.464", "26.786", "4.464", "0.446"),
    "databricks/system.ai.gemini-2-5-pro": ("22.321", "178.571", "22.321", "2.232"),
    "databricks/system.ai.gemini-2-5-flash": ("5.357", "44.643", "5.357", "0.536"),
    "databricks/system.ai.kimi-k3": ("42.857", "214.286", "42.857", "4.286"),
    "databricks/system.ai.deepseek-v4-flash-0731": ("2.000", "4.000", "2.000", "0.400"),
    "databricks/system.ai.deepseek-v4-pro-0813": ("18.857", "56.571", "18.857", "1.886"),
    "databricks/system.ai.glm-5-2": ("20.000", "62.857", "20.000", "3.714"),
    "databricks/system.ai.glm-5-3": ("20.000", "62.857", "20.000", "3.714"),
    "databricks/system.ai.glm-5-3-flash": ("2.143", "7.143", "2.143", "0.429"),
    "databricks/system.ai.inkling": ("14.286", "57.857", "14.286", "2.429"),
    "databricks/system.ai.grok-4-6": ("35.714", "107.143", "35.714", "8.929"),
    "databricks/system.ai.qwen35-122b-a10b": ("3.143", "31.429", "3.143", "3.143"),
    "databricks/system.ai.qwen3-next-80b-a3b-instruct": ("2.143", "17.143", "2.143", "2.143"),
    "databricks/system.ai.qwen3-embedding-0-6b": ("0.286", "0", "0.286", "0.286"),
}
PROMOTIONAL_DISCOUNT: Final = 0.80
PROMOTION_EXPIRES: Final = "2027-01-31"
ENTRIES_STORING_PROMOTIONAL_RATE: Final = (
    "databricks/system.ai.gemini-2-5-pro",
    "databricks/system.ai.gemini-2-5-flash",
)
ENTRIES_STORING_LIST_RATE_DESPITE_PROMOTION: Final = (
    "databricks/system.ai.gemini-3-6-flash",
    "databricks/system.ai.gemini-3-5-flash",
    "databricks/databricks-gemini-3-5-flash-lite",
    "databricks/system.ai.grok-4-6",
    "databricks/system.ai.gemini-3-1-pro",
    "databricks/databricks-gemini-3-pro",
    "databricks/system.ai.gemini-3-flash",
    "databricks/system.ai.gemini-3-1-flash-lite",
)
CACHE_FIELDS: Final = ("cache_creation_input_token_cost", "cache_read_input_token_cost")


def _model_info(model: str) -> ModelInfo:
    return litellm.get_model_info(model=model, custom_llm_provider="databricks")


def _dollars_per_token(dbu_per_million: str) -> float:
    return float(Decimal(dbu_per_million) * DOLLARS_PER_DBU / Decimal(10) ** 6)


@pytest.mark.parametrize(
    "model",
    [
        "databricks/system.ai.claude-opus-4-8",
        "databricks/system.ai.claude-opus-5",
        "databricks/system.ai.claude-sonnet-5",
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
    model: Final = "databricks/system.ai.claude-sonnet-5"
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
    model: Final = "databricks/system.ai.meta-llama-3-3-70b-instruct"
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
    sonnet_5: Final = _model_info("databricks/system.ai.claude-sonnet-5")
    sonnet_4_6: Final = _model_info("databricks/system.ai.claude-sonnet-4-6")

    for field in PRICE_FIELDS:
        assert sonnet_5[field] == pytest.approx(sonnet_4_6[field]), field


def test_sonnet_5_system_ai_and_legacy_cost_lookup(local_model_cost_map: None) -> None:
    system_ai_model: Final = "databricks/system.ai.claude-sonnet-5"
    legacy_model: Final = "databricks/databricks-claude-sonnet-5"
    info: Final = _model_info(system_ai_model)
    input_cost: Final = info["input_cost_per_token"]
    output_cost: Final = info["output_cost_per_token"]
    assert input_cost is not None and input_cost > 0
    assert output_cost is not None and output_cost > 0

    usage: Final = Usage(
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=400,
    )
    system_cost: Final = cost_per_token(model=system_ai_model, usage=usage)
    legacy_cost: Final = cost_per_token(model=legacy_model, usage=usage)
    assert system_cost[0] > 0
    assert system_cost[1] > 0
    assert legacy_cost == pytest.approx(system_cost)


def test_databricks_provider_models_advertises_unity_catalog_sonnet_5(
    local_model_cost_map: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_models: Final[set[str]] = set()
    monkeypatch.setattr(litellm, "databricks_models", fresh_models)
    monkeypatch.setitem(litellm.models_by_provider, "databricks", fresh_models)
    litellm._populate_provider_model_sets(litellm.model_cost)

    models: Final = get_known_models_from_wildcard("databricks/*")
    assert "databricks/system.ai.claude-sonnet-5" in models
    assert "databricks/databricks-claude-sonnet-5" not in models


def test_sonnet_5_legacy_metadata_and_max_tokens_lookup(local_model_cost_map: None) -> None:
    legacy_model: Final = "databricks/databricks-claude-sonnet-5"
    system_ai_model: Final = "databricks/system.ai.claude-sonnet-5"

    info: Final = litellm.get_model_info(legacy_model, custom_llm_provider="databricks")
    assert info["key"] == system_ai_model
    assert info["max_tokens"] == 128000
    input_cost: Final = info["input_cost_per_token"]
    assert input_cost is not None and input_cost > 0

    assert litellm.get_max_tokens(legacy_model) == 128000
    assert litellm.get_max_tokens(system_ai_model) == 128000


@pytest.mark.parametrize("legacy_name,canonical_name", _DATABRICKS_LEGACY_TO_CANONICAL.items())
def test_migrated_models_wildcard_and_map_parity(
    legacy_name: str,
    canonical_name: str,
) -> None:
    main_cost: Final = json.loads(MAIN_PRICES.read_text())
    backup_cost: Final = json.loads(BACKUP_PRICES.read_text())

    legacy_key: Final = f"databricks/{legacy_name}"
    canonical_key: Final = f"databricks/{canonical_name}"

    assert canonical_key in main_cost
    assert canonical_key in backup_cost
    assert main_cost[canonical_key] == backup_cost[canonical_key]

    assert legacy_key not in main_cost
    assert legacy_key not in backup_cost


@pytest.mark.parametrize("legacy_name,canonical_name", _DATABRICKS_LEGACY_TO_CANONICAL.items())
def test_migrated_models_metadata_and_max_tokens_lookup(
    local_model_cost_map: None,
    legacy_name: str,
    canonical_name: str,
) -> None:
    legacy_key: Final = f"databricks/{legacy_name}"
    canonical_key: Final = f"databricks/{canonical_name}"

    legacy_info: Final = litellm.get_model_info(legacy_key, custom_llm_provider="databricks")
    canonical_info: Final = litellm.get_model_info(canonical_key, custom_llm_provider="databricks")

    assert legacy_info["key"] == canonical_key
    assert legacy_info["max_tokens"] == canonical_info["max_tokens"]
    assert legacy_info["input_cost_per_token"] == canonical_info["input_cost_per_token"]
    assert legacy_info["output_cost_per_token"] == canonical_info["output_cost_per_token"]
    assert legacy_info["mode"] == canonical_info["mode"]

    if canonical_info["max_tokens"] is not None:
        assert litellm.get_max_tokens(legacy_key) == litellm.get_max_tokens(canonical_key)


@pytest.mark.parametrize("legacy_name,canonical_name", _DATABRICKS_LEGACY_TO_CANONICAL.items())
def test_migrated_models_cost_lookup(
    local_model_cost_map: None,
    legacy_name: str,
    canonical_name: str,
) -> None:
    legacy_key: Final = f"databricks/{legacy_name}"
    canonical_key: Final = f"databricks/{canonical_name}"

    usage: Final = Usage(
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
    )
    legacy_cost: Final = cost_per_token(model=legacy_key, usage=usage)
    canonical_cost: Final = cost_per_token(model=canonical_key, usage=usage)
    assert legacy_cost == pytest.approx(canonical_cost)


def test_untouched_models_collision_guard(local_model_cost_map: None) -> None:
    untouched_pro: Final = "databricks/databricks-gemini-3-pro"
    migrated_3_1: Final = "databricks/system.ai.gemini-3-1-pro"

    info_pro: Final = litellm.get_model_info(untouched_pro, custom_llm_provider="databricks")
    assert info_pro["key"] == untouched_pro

    info_3_1: Final = litellm.get_model_info(migrated_3_1, custom_llm_provider="databricks")
    assert info_3_1["key"] == migrated_3_1


def test_wildcard_discovery_advertises_only_canonical(
    local_model_cost_map: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_models: Final[set[str]] = set()
    monkeypatch.setattr(litellm, "databricks_models", fresh_models)
    monkeypatch.setitem(litellm.models_by_provider, "databricks", fresh_models)
    litellm._populate_provider_model_sets(litellm.model_cost)

    advertised_models: Final = get_known_models_from_wildcard("databricks/*")
    for legacy_name, canonical_name in _DATABRICKS_LEGACY_TO_CANONICAL.items():
        assert f"databricks/{canonical_name}" in advertised_models
        assert f"databricks/{legacy_name}" not in advertised_models


def test_non_databricks_providers_regression(local_model_cost_map: None) -> None:
    openai_info: Final = litellm.get_model_info("gpt-4o", custom_llm_provider="openai")
    assert openai_info["litellm_provider"] == "openai"


def test_databricks_get_valid_models_security_guard(requests_mock) -> None:
    sentinel: Final = "DATABRICKS_SECRET_SENTINEL"
    requests_mock.get("https://api.openai.com/v1/models", text="forbidden")

    models: Final = litellm.get_valid_models(
        check_provider_endpoint=True,
        custom_llm_provider="databricks",
        api_key=sentinel,
    )
    assert len(models) > 0
    assert not any(req.url.startswith("https://api.openai.com") for req in requests_mock.request_history)


def test_databricks_model_info_provider_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.databricks.common_utils import DatabricksModelInfo
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    provider_info: Final = ProviderConfigManager.get_provider_model_info(
        model=None,
        provider=LlmProviders.DATABRICKS,
    )
    assert isinstance(provider_info, DatabricksModelInfo)
    assert provider_info.get_model_cost_key("databricks-claude-sonnet-5") == "databricks/system.ai.claude-sonnet-5"
    assert provider_info.get_model_cost_key("unknown-model") is None
    assert provider_info.get_base_model("system.ai.claude-sonnet-5") == "system.ai.claude-sonnet-5"
    assert provider_info.validate_environment({"h": "1"}, "m", [], {}, {}) == {"h": "1"}

    monkeypatch.setenv("DATABRICKS_API_KEY", "env_key")
    monkeypatch.setenv("DATABRICKS_API_BASE", "https://env.example.com")
    assert provider_info.get_api_key(None) == "env_key"
    assert provider_info.get_api_key("explicit_key") == "explicit_key"
    assert provider_info.get_api_base(None) == "https://env.example.com"
    assert provider_info.get_api_base("https://explicit.example.com") == "https://explicit.example.com"
