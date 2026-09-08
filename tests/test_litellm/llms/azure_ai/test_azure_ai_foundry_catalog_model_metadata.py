from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import completion_cost, cost_per_token
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import TranscriptionResponse

REPO_ROOT: Final = Path(__file__).parents[4]
MAIN_COST_MAP: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_COST_MAP: Final = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"
COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
AZURE_PRICING_PREFIX: Final = "https://azure.microsoft.com/en-us/pricing/details/"
A_MILLION: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class TokenPricedCatalogModel:
    catalog_name: str
    dollars_per_million_input: float
    dollars_per_million_output: float


TOKEN_PRICED_MODELS: Final = (
    TokenPricedCatalogModel("gpt-chat-latest", 5.0, 30.0),
    TokenPricedCatalogModel("codex-mini", 1.5, 6.0),
    TokenPricedCatalogModel("model-router", 0.14, 0.0),
    TokenPricedCatalogModel("cohere-command-a", 2.5, 10.0),
    TokenPricedCatalogModel("grok-4-20-reasoning", 1.25, 2.5),
    TokenPricedCatalogModel("grok-4-20-non-reasoning", 1.25, 2.5),
)
GROK_4_20_NAMES: Final = ("grok-4-20-reasoning", "grok-4-20-non-reasoning")
CATALOG_NAMES: Final = tuple(spec.catalog_name for spec in TOKEN_PRICED_MODELS) + ("whisper",)


def _cost_map_entry(path: Path, catalog_name: str) -> dict[str, object]:
    return COST_MAP_ADAPTER.validate_json(path.read_bytes())[f"azure_ai/{catalog_name}"]


@pytest.mark.parametrize("catalog_name", CATALOG_NAMES)
def test_azure_ai_catalog_name_routes_to_azure_ai(catalog_name: str) -> None:
    routed_model, provider, _, _ = get_llm_provider(model=f"azure_ai/{catalog_name}")
    assert (routed_model, provider) == (catalog_name, "azure_ai")


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("spec", TOKEN_PRICED_MODELS, ids=lambda spec: spec.catalog_name)
def test_azure_ai_catalog_name_costs_a_million_tokens_at_list_price(spec: TokenPricedCatalogModel) -> None:
    prompt_cost, completion_cost_usd = cost_per_token(
        model=f"azure_ai/{spec.catalog_name}", prompt_tokens=A_MILLION, completion_tokens=A_MILLION
    )
    assert prompt_cost == pytest.approx(spec.dollars_per_million_input)
    assert completion_cost_usd == pytest.approx(spec.dollars_per_million_output)


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("spec", TOKEN_PRICED_MODELS, ids=lambda spec: spec.catalog_name)
def test_azure_ai_catalog_name_prices_the_same_in_any_casing(spec: TokenPricedCatalogModel) -> None:
    lowercase_cost = cost_per_token(model=f"azure_ai/{spec.catalog_name}", prompt_tokens=A_MILLION, completion_tokens=0)
    upper_cost = cost_per_token(model=f"azure_ai/{spec.catalog_name.upper()}", prompt_tokens=A_MILLION, completion_tokens=0)
    assert upper_cost == lowercase_cost


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("catalog_name", GROK_4_20_NAMES)
def test_azure_ai_grok_4_20_bills_cached_prompt_tokens_at_the_input_price(catalog_name: str) -> None:
    uncached_prompt_cost, _ = cost_per_token(model=f"azure_ai/{catalog_name}", prompt_tokens=A_MILLION, completion_tokens=0)
    cached_prompt_cost, _ = cost_per_token(
        model=f"azure_ai/{catalog_name}",
        prompt_tokens=A_MILLION,
        completion_tokens=0,
        cache_read_input_tokens=A_MILLION,
    )
    assert uncached_prompt_cost > 0
    assert cached_prompt_cost == pytest.approx(uncached_prompt_cost)


@pytest.mark.usefixtures("local_model_cost_map")
def test_azure_ai_whisper_catalog_name_is_priced_per_second() -> None:
    transcription: Final = TranscriptionResponse(text="hello")
    transcription._hidden_params = {  # pyright: ignore[reportPrivateUsage]  # TranscriptionResponse exposes no public hidden-params setter
        "custom_llm_provider": "azure_ai",
        "model": "azure_ai/whisper",
        "audio_transcription_duration": 3600,
    }
    cost = completion_cost(
        completion_response=transcription,
        model="azure_ai/whisper",
        custom_llm_provider="azure_ai",
        call_type="atranscription",
    )
    assert cost == pytest.approx(0.36)


@pytest.mark.parametrize("catalog_name", CATALOG_NAMES)
def test_azure_ai_catalog_entry_source_and_backup_match(catalog_name: str) -> None:
    main_entry = _cost_map_entry(MAIN_COST_MAP, catalog_name)
    backup_entry = _cost_map_entry(BACKUP_COST_MAP, catalog_name)

    assert str(main_entry["source"]).startswith(AZURE_PRICING_PREFIX)
    assert backup_entry == main_entry


def test_azure_ai_model_router_spellings_share_one_entry() -> None:
    underscore_entry = _cost_map_entry(MAIN_COST_MAP, "model_router")
    hyphen_entry = _cost_map_entry(MAIN_COST_MAP, "model-router")

    assert {k: v for k, v in underscore_entry.items() if k != "comment"} == {
        k: v for k, v in hyphen_entry.items() if k != "comment"
    }
