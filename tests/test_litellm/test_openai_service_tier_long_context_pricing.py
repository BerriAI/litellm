import json
from functools import lru_cache
from pathlib import Path

import pytest

import litellm

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

FLEX_LONG_CONTEXT = {
    "gpt-5.4": {
        "input_cost_per_token_above_272k_tokens_flex": 2.5e-06,
        "output_cost_per_token_above_272k_tokens_flex": 1.125e-05,
        "cache_read_input_token_cost_above_272k_tokens_flex": 2.5e-07,
    },
    "gpt-5.4-pro": {
        "input_cost_per_token_above_272k_tokens_flex": 3e-05,
        "output_cost_per_token_above_272k_tokens_flex": 0.000135,
    },
    "gpt-5.5": {
        "input_cost_per_token_above_272k_tokens_flex": 5e-06,
        "output_cost_per_token_above_272k_tokens_flex": 2.25e-05,
        "cache_read_input_token_cost_above_272k_tokens_flex": 5e-07,
    },
}

PRIORITY_LONG_CONTEXT = {
    "gpt-5.6": {
        "input_cost_per_token_above_272k_tokens_priority": 1.6e-05,
        "output_cost_per_token_above_272k_tokens_priority": 6e-05,
        "cache_read_input_token_cost_above_272k_tokens_priority": 1.6e-06,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 2e-05,
    },
    "gpt-5.6-sol": {
        "input_cost_per_token_above_272k_tokens_priority": 1.6e-05,
        "output_cost_per_token_above_272k_tokens_priority": 6e-05,
        "cache_read_input_token_cost_above_272k_tokens_priority": 1.6e-06,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 2e-05,
    },
    "gpt-5.6-terra": {
        "input_cost_per_token_above_272k_tokens_priority": 8e-06,
        "output_cost_per_token_above_272k_tokens_priority": 3.6e-05,
        "cache_read_input_token_cost_above_272k_tokens_priority": 8e-07,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 1e-05,
    },
    "gpt-5.6-luna": {
        "input_cost_per_token_above_272k_tokens_priority": 8e-07,
        "output_cost_per_token_above_272k_tokens_priority": 3.6e-06,
        "cache_read_input_token_cost_above_272k_tokens_priority": 8e-08,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 1e-06,
    },
    "gpt-6-astra": {
        "input_cost_per_token_above_272k_tokens_priority": 4e-05,
        "output_cost_per_token_above_272k_tokens_priority": 0.00015,
        "cache_read_input_token_cost_above_272k_tokens_priority": 4e-06,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 5e-05,
    },
}

EXPECTED = {**FLEX_LONG_CONTEXT, **PRIORITY_LONG_CONTEXT}

NO_PUBLISHED_PRIORITY_LONG_CONTEXT = ("gpt-5.4", "gpt-5.5")


@pytest.fixture(autouse=True)
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.add_known_models()


@lru_cache(maxsize=2)
def _load(path: Path) -> dict[str, dict[str, object]]:
    with open(path) as f:
        return json.load(f)


LONG_CONTEXT_PROMPT_TOKENS = 300_000
COMPLETION_TOKENS = 1_000

TIERED_COST_CASES = [
    ("gpt-5.4", "flex", 2.5e-06, 1.125e-05),
    ("gpt-5.4-pro", "flex", 3e-05, 0.000135),
    ("gpt-5.5", "flex", 5e-06, 2.25e-05),
    ("gpt-5.6", "priority", 1.6e-05, 6e-05),
    ("gpt-5.6-sol", "priority", 1.6e-05, 6e-05),
    ("gpt-5.6-terra", "priority", 8e-06, 3.6e-05),
    ("gpt-5.6-luna", "priority", 8e-07, 3.6e-06),
    ("gpt-6-astra", "priority", 4e-05, 0.00015),
]
