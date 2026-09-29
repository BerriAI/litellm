import json
from functools import lru_cache
from pathlib import Path
from typing import Final, cast

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

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
    "gpt-6-sol": {
        "input_cost_per_token_above_272k_tokens_priority": 8e-06,
        "output_cost_per_token_above_272k_tokens_priority": 3e-05,
        "cache_read_input_token_cost_above_272k_tokens_priority": 8e-07,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 1e-05,
    },
    "gpt-6-luna": {
        "input_cost_per_token_above_272k_tokens_priority": 4e-07,
        "output_cost_per_token_above_272k_tokens_priority": 1.5e-06,
        "cache_read_input_token_cost_above_272k_tokens_priority": 4e-08,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 5e-07,
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
    ("gpt-6-sol", "priority", 8e-06, 3e-05),
    ("gpt-6-luna", "priority", 4e-07, 1.5e-06),
]

ULTRAFAST_LONG_CONTEXT_MODELS = sorted(
    key
    for key, row in _load(MAIN_PATH).items()
    if "input_cost_per_token_above_272k_tokens_ultrafast" in row
)

assert ULTRAFAST_LONG_CONTEXT_MODELS


@pytest.mark.parametrize("model", ULTRAFAST_LONG_CONTEXT_MODELS)
def test_ultrafast_long_context_prompt_bills_the_ultrafast_long_context_rate(model: str) -> None:
    row: Final = _load(MAIN_PATH)[model]
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model,
        prompt_tokens=LONG_CONTEXT_PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        service_tier="ultrafast",
    )
    assert prompt_cost == pytest.approx(
        LONG_CONTEXT_PROMPT_TOKENS * cast(float, row["input_cost_per_token_above_272k_tokens_ultrafast"])
    )
    assert completion_cost == pytest.approx(
        COMPLETION_TOKENS * cast(float, row["output_cost_per_token_above_272k_tokens_ultrafast"])
    )


def test_ultrafast_long_context_cached_tokens_bill_the_ultrafast_cache_read_rate() -> None:
    row: Final = _load(MAIN_PATH)["gpt-6-astra"]
    usage: Final = Usage(
        prompt_tokens=LONG_CONTEXT_PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=100_000),
    )
    prompt_cost, _ = generic_cost_per_token(
        model="gpt-6-astra",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="ultrafast",
    )
    assert prompt_cost == pytest.approx(
        200_000 * cast(float, row["input_cost_per_token_above_272k_tokens_ultrafast"])
        + 100_000 * cast(float, row["cache_read_input_token_cost_above_272k_tokens_ultrafast"])
    )
