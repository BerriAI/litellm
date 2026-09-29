import json
from functools import lru_cache
from pathlib import Path
from typing import Final

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

# Source: https://developers.openai.com/api/docs/pricing (2026-09-29)
ULTRAFAST_LONG_CONTEXT = {
    "gpt-6-astra": {
        "input_cost_per_token_above_272k_tokens_ultrafast": 0.00012,
        "output_cost_per_token_above_272k_tokens_ultrafast": 0.00045,
        "cache_read_input_token_cost_above_272k_tokens_ultrafast": 1.2e-05,
        "cache_creation_input_token_cost_above_272k_tokens_ultrafast": 0.00015,
    }
}

EXPECTED: Final = {
    model: {
        **FLEX_LONG_CONTEXT.get(model, {}),
        **PRIORITY_LONG_CONTEXT.get(model, {}),
        **ULTRAFAST_LONG_CONTEXT.get(model, {}),
    }
    for model in {**FLEX_LONG_CONTEXT, **PRIORITY_LONG_CONTEXT, **ULTRAFAST_LONG_CONTEXT}
}

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
    ("gpt-6-astra", "ultrafast", 0.00012, 0.00045),
    ("gpt-6-sol", "priority", 8e-06, 3e-05),
    ("gpt-6-luna", "priority", 4e-07, 1.5e-06),
]


@pytest.mark.parametrize("path", (MAIN_PATH, BACKUP_PATH), ids=("main", "backup"))
def test_catalogs_contain_expected_tiered_long_context_rates(path: Path) -> None:
    catalog: Final = _load(path)

    assert {model: {key: catalog[model][key] for key in rates} for model, rates in EXPECTED.items()} == EXPECTED


def test_get_model_info_preserves_expected_tiered_long_context_rates() -> None:
    assert {
        model: {key: litellm.get_model_info(model)[key] for key in rates} for model, rates in EXPECTED.items()
    } == EXPECTED


@pytest.mark.parametrize(("model", "service_tier", "input_rate", "output_rate"), TIERED_COST_CASES)
def test_tiered_long_context_cost_uses_catalog_rates(
    model: str, service_tier: str, input_rate: float, output_rate: float
) -> None:
    usage: Final = Usage(
        prompt_tokens=LONG_CONTEXT_PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        total_tokens=LONG_CONTEXT_PROMPT_TOKENS + COMPLETION_TOKENS,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
        service_tier=service_tier,
    )

    assert prompt_cost == pytest.approx(LONG_CONTEXT_PROMPT_TOKENS * input_rate)
    assert completion_cost == pytest.approx(COMPLETION_TOKENS * output_rate)


def test_gpt_6_astra_ultrafast_long_context_costs_and_controls() -> None:
    ultrafast_usage: Final = Usage(
        prompt_tokens=300_000,
        completion_tokens=1_000,
        total_tokens=301_000,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=100, cache_creation_tokens=200),
    )
    ultrafast_prompt_cost, ultrafast_completion_cost = generic_cost_per_token(
        model="gpt-6-astra",
        usage=ultrafast_usage,
        custom_llm_provider="openai",
        service_tier="ultrafast",
    )
    standard_prompt_cost, standard_completion_cost = generic_cost_per_token(
        model="gpt-6-astra",
        usage=ultrafast_usage,
        custom_llm_provider="openai",
    )
    below_threshold_usage: Final = Usage(
        prompt_tokens=271_000,
        completion_tokens=1_000,
        total_tokens=272_000,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=100, cache_creation_tokens=200),
    )
    below_threshold_prompt_cost, below_threshold_completion_cost = generic_cost_per_token(
        model="gpt-6-astra",
        usage=below_threshold_usage,
        custom_llm_provider="openai",
        service_tier="ultrafast",
    )

    assert (ultrafast_prompt_cost, ultrafast_completion_cost) == pytest.approx(
        (299_700 * 0.00012 + 100 * 1.2e-05 + 200 * 0.00015, 1_000 * 0.00045)
    )
    assert ultrafast_prompt_cost + ultrafast_completion_cost == pytest.approx(36.4452)
    assert (standard_prompt_cost, standard_completion_cost) == pytest.approx(
        (299_700 * 0.00002 + 100 * 2e-06 + 200 * 2.5e-05, 1_000 * 7.5e-05)
    )
    assert (below_threshold_prompt_cost, below_threshold_completion_cost) == pytest.approx(
        (270_700 * 6e-05 + 100 * 6e-06 + 200 * 7.5e-05, 1_000 * 0.0003)
    )
