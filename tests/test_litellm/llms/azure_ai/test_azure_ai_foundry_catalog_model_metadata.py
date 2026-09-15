from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import completion_cost
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import TranscriptionResponse

REPO_ROOT: Final = Path(__file__).parents[4]
MAIN_COST_MAP: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_COST_MAP: Final = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"
COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
A_MILLION: Final = 1_000_000
AN_HOUR_IN_SECONDS: Final = 3600

TOKEN_PRICED_NAMES: Final = (
    "gpt-chat-latest",
    "codex-mini",
    "model-router",
    "cohere-command-a",
    "grok-4-20-reasoning",
    "grok-4-20-non-reasoning",
)
GROK_4_20_NAMES: Final = ("grok-4-20-reasoning", "grok-4-20-non-reasoning")
CATALOG_NAMES: Final = TOKEN_PRICED_NAMES + ("whisper",)


def _cost_map_entry(path: Path, catalog_name: str) -> dict[str, object]:
    return COST_MAP_ADAPTER.validate_json(path.read_bytes())[f"azure_ai/{catalog_name}"]


def _whisper_transcription_cost(duration_seconds: int) -> float:
    transcription: Final = TranscriptionResponse(text="hello")
    transcription._hidden_params = {  # pyright: ignore[reportPrivateUsage]  # TranscriptionResponse exposes no public hidden-params setter
        "custom_llm_provider": "azure_ai",
        "model": "azure_ai/whisper",
        "audio_transcription_duration": duration_seconds,
    }
    return completion_cost(
        completion_response=transcription,
        model="azure_ai/whisper",
        custom_llm_provider="azure_ai",
        call_type="atranscription",
    )


@pytest.mark.parametrize("catalog_name", CATALOG_NAMES)
def test_azure_ai_catalog_name_routes_to_azure_ai(catalog_name: str) -> None:
    routed_model, provider, _, _ = get_llm_provider(model=f"azure_ai/{catalog_name}")
    assert (routed_model, provider) == (catalog_name, "azure_ai")


def test_azure_ai_model_router_spellings_share_one_entry() -> None:
    underscore_entry = _cost_map_entry(MAIN_COST_MAP, "model_router")
    hyphen_entry = _cost_map_entry(MAIN_COST_MAP, "model-router")

    assert {k: v for k, v in underscore_entry.items() if k != "comment"} == {
        k: v for k, v in hyphen_entry.items() if k != "comment"
    }
