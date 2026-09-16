import json
from pathlib import Path

import pytest

import litellm
from litellm import get_model_info

AZURE_AI_GROK_4_3_MODEL = "azure_ai/grok-4.3"
AZURE_AI_GROK_4_3_SOURCE = "https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-grok-4-3-on-microsoft-foundry-latest-generation-agentic-capabilities/4517096"


def _load_model_cost(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def reload_model_costs():
    original_model_cost = litellm.model_cost
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    litellm.model_cost = _load_model_cost(json_path)
    get_model_info.cache_clear()
    yield
    litellm.model_cost = original_model_cost
    get_model_info.cache_clear()


def test_azure_ai_grok_4_3_backup_matches_main():
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    main_cost = _load_model_cost(main_path)
    backup_cost = _load_model_cost(backup_path)

    assert backup_cost.get(AZURE_AI_GROK_4_3_MODEL) == main_cost.get(
        AZURE_AI_GROK_4_3_MODEL
    )
