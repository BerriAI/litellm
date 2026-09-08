from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import cost_per_token, get_model_info
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

REPO_ROOT: Final = Path(__file__).parents[4]
COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
AZURE_OPENAI_PRICING: Final = "https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/"
FOUNDRY_AOAI_PRICING: Final = "https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/aoai/"
FOUNDRY_COHERE_PRICING: Final = "https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/cohere/"
FOUNDRY_GROK_PRICING: Final = "https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/grok/"


@dataclass(frozen=True, slots=True)
class TokenPricedCatalogModel:
    catalog_name: str
    mode: str
    source: str
    input_cost_per_token: float
    output_cost_per_token: float
    max_input_tokens: int
    max_output_tokens: int
    cache_read_input_token_cost: float | None
    deprecation_date: str | None
    supported_flags: tuple[str, ...]


TOKEN_PRICED_MODELS: Final = (
    TokenPricedCatalogModel(
        catalog_name="gpt-chat-latest",
        mode="chat",
        source=AZURE_OPENAI_PRICING,
        input_cost_per_token=5e-06,
        output_cost_per_token=3e-05,
        max_input_tokens=272000,
        max_output_tokens=128000,
        cache_read_input_token_cost=5e-07,
        deprecation_date="2026-12-02",
        supported_flags=(
            "supports_function_calling",
            "supports_prompt_caching",
            "supports_reasoning",
            "supports_response_schema",
            "supports_tool_choice",
            "supports_vision",
            "supports_web_search",
        ),
    ),
    TokenPricedCatalogModel(
        catalog_name="codex-mini",
        mode="responses",
        source=AZURE_OPENAI_PRICING,
        input_cost_per_token=1.5e-06,
        output_cost_per_token=6e-06,
        max_input_tokens=200000,
        max_output_tokens=100000,
        cache_read_input_token_cost=3.75e-07,
        deprecation_date="2026-11-15",
        supported_flags=(
            "supports_function_calling",
            "supports_prompt_caching",
            "supports_reasoning",
            "supports_vision",
        ),
    ),
    TokenPricedCatalogModel(
        catalog_name="model-router",
        mode="chat",
        source=FOUNDRY_AOAI_PRICING,
        input_cost_per_token=1.4e-07,
        output_cost_per_token=0.0,
        max_input_tokens=200000,
        max_output_tokens=32768,
        cache_read_input_token_cost=None,
        deprecation_date="2027-05-20",
        supported_flags=(),
    ),
    TokenPricedCatalogModel(
        catalog_name="cohere-command-a",
        mode="chat",
        source=FOUNDRY_COHERE_PRICING,
        input_cost_per_token=2.5e-06,
        output_cost_per_token=1e-05,
        max_input_tokens=131072,
        max_output_tokens=8182,
        cache_read_input_token_cost=None,
        deprecation_date=None,
        supported_flags=("supports_function_calling", "supports_tool_choice"),
    ),
    TokenPricedCatalogModel(
        catalog_name="grok-4-20-reasoning",
        mode="chat",
        source=FOUNDRY_GROK_PRICING,
        input_cost_per_token=1.25e-06,
        output_cost_per_token=2.5e-06,
        max_input_tokens=262000,
        max_output_tokens=8192,
        cache_read_input_token_cost=None,
        deprecation_date="2027-04-06",
        supported_flags=(
            "supports_function_calling",
            "supports_reasoning",
            "supports_response_schema",
            "supports_tool_choice",
            "supports_vision",
            "supports_web_search",
        ),
    ),
    TokenPricedCatalogModel(
        catalog_name="grok-4-20-non-reasoning",
        mode="chat",
        source=FOUNDRY_GROK_PRICING,
        input_cost_per_token=1.25e-06,
        output_cost_per_token=2.5e-06,
        max_input_tokens=262000,
        max_output_tokens=8192,
        cache_read_input_token_cost=None,
        deprecation_date="2027-04-06",
        supported_flags=(
            "supports_function_calling",
            "supports_response_schema",
            "supports_tool_choice",
            "supports_vision",
            "supports_web_search",
        ),
    ),
)
CATALOG_NAMES: Final = tuple(spec.catalog_name for spec in TOKEN_PRICED_MODELS) + ("whisper",)


def _cost_map_entry(path: Path, catalog_name: str) -> dict[str, object]:
    return COST_MAP_ADAPTER.validate_json(path.read_bytes())[f"azure_ai/{catalog_name}"]


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("spec", TOKEN_PRICED_MODELS, ids=lambda spec: spec.catalog_name)
def test_azure_ai_catalog_name_is_priced_and_routed(spec: TokenPricedCatalogModel) -> None:
    routed_model, provider, _, _ = get_llm_provider(model=f"azure_ai/{spec.catalog_name}")
    assert (routed_model, provider) == (spec.catalog_name, "azure_ai")

    info = get_model_info(model=routed_model, custom_llm_provider=provider)
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == spec.mode
    assert info["input_cost_per_token"] == spec.input_cost_per_token
    assert info["output_cost_per_token"] == spec.output_cost_per_token
    assert info["cache_read_input_token_cost"] == spec.cache_read_input_token_cost
    assert info["max_input_tokens"] == spec.max_input_tokens
    assert info["max_output_tokens"] == spec.max_output_tokens
    assert info["max_tokens"] == spec.max_output_tokens
    for flag in spec.supported_flags:
        assert info[flag] is True, flag


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(
    "spec",
    [spec for spec in TOKEN_PRICED_MODELS if spec.catalog_name != "model-router"],
    ids=lambda spec: spec.catalog_name,
)
def test_azure_ai_catalog_name_costs_a_million_tokens_at_list_price(spec: TokenPricedCatalogModel) -> None:
    prompt_cost, completion_cost = cost_per_token(
        model=f"azure_ai/{spec.catalog_name}", prompt_tokens=1_000_000, completion_tokens=1_000_000
    )
    assert prompt_cost == pytest.approx(spec.input_cost_per_token * 1_000_000)
    assert completion_cost == pytest.approx(spec.output_cost_per_token * 1_000_000)


@pytest.mark.usefixtures("local_model_cost_map")
def test_azure_ai_whisper_catalog_name_is_priced_per_second() -> None:
    routed_model, provider, _, _ = get_llm_provider(model="azure_ai/whisper")
    assert (routed_model, provider) == ("whisper", "azure_ai")

    info = get_model_info(model=routed_model, custom_llm_provider=provider)
    assert info["mode"] == "audio_transcription"
    assert info["input_cost_per_second"] == 0.0001
    assert info["output_cost_per_second"] == 0.0001


@pytest.mark.parametrize("catalog_name", CATALOG_NAMES)
def test_azure_ai_catalog_entry_source_and_backup_match(catalog_name: str) -> None:
    main_entry = _cost_map_entry(REPO_ROOT / "model_prices_and_context_window.json", catalog_name)
    backup_entry = _cost_map_entry(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json", catalog_name)

    assert str(main_entry["source"]).startswith("https://azure.microsoft.com/en-us/pricing/details/")
    assert backup_entry == main_entry


@pytest.mark.parametrize("spec", TOKEN_PRICED_MODELS, ids=lambda spec: spec.catalog_name)
def test_azure_ai_catalog_entry_carries_its_retirement_date(spec: TokenPricedCatalogModel) -> None:
    entry = _cost_map_entry(REPO_ROOT / "model_prices_and_context_window.json", spec.catalog_name)
    assert entry.get("deprecation_date") == spec.deprecation_date
