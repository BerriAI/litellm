import json
from pathlib import Path

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"
MODEL = "gpt-6-astra"


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_gpt_6_astra_backup_matches_main():
    main_entry = _load(MAIN_PATH).get(MODEL)
    assert main_entry is not None, f"{MODEL} missing from model_prices_and_context_window.json"
    assert _load(BACKUP_PATH).get(MODEL) == main_entry


def test_gpt_6_astra_routes_to_openai_on_the_gpt_5_reasoning_path():
    routed_model, provider, _, _ = get_llm_provider(model=f"openai/{MODEL}")
    assert (routed_model, provider) == (MODEL, "openai")

    config = ProviderConfigManager.get_provider_chat_config(model=MODEL, provider=LlmProviders.OPENAI)
    assert isinstance(config, OpenAIGPT5Config)


def test_gpt_6_astra_maps_max_tokens_and_drops_temperature_like_gpt_5():
    mapped = litellm.get_optional_params(
        model=MODEL,
        custom_llm_provider="openai",
        max_tokens=100,
        temperature=0.2,
        reasoning_effort="high",
        drop_params=True,
    )

    assert mapped["max_completion_tokens"] == 100
    assert "max_tokens" not in mapped
    assert "temperature" not in mapped
    assert mapped["reasoning_effort"] == "high"
