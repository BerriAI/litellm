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
}

EXPECTED = {**FLEX_LONG_CONTEXT, **PRIORITY_LONG_CONTEXT}

NO_PUBLISHED_PRIORITY_LONG_CONTEXT = ("gpt-5.4", "gpt-5.5")


@pytest.fixture(autouse=True)
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


@lru_cache(maxsize=2)
def _load(path: Path) -> dict[str, dict[str, object]]:
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("path", [MAIN_PATH, BACKUP_PATH], ids=["main", "backup"])
@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_service_tier_long_context_rates_are_published(model: str, path: Path) -> None:
    """Each tier must carry its own above-272K rates, in both price files."""
    info = _load(path).get(model)
    assert info is not None, f"{model} not found in {path.name}"
    for key, expected in EXPECTED[model].items():
        assert info.get(key) == pytest.approx(expected), f"{model}.{key} is {info.get(key)!r}, expected {expected!r}"


@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_tier_long_context_rate_is_half_or_double_the_standard(model: str) -> None:
    """Flex is half the standard long-context rate; priority is double it."""
    info = _load(MAIN_PATH)[model]
    tier = "flex" if model in FLEX_LONG_CONTEXT else "priority"
    ratio = 0.5 if tier == "flex" else 2.0
    for base in ("input_cost_per_token", "output_cost_per_token"):
        standard = info[f"{base}_above_272k_tokens"]
        tiered = info[f"{base}_above_272k_tokens_{tier}"]
        assert tiered == pytest.approx(standard * ratio), (
            f"{model}.{base}_above_272k_tokens_{tier} is {tiered!r}, "
            f"expected {ratio}x the standard long-context rate {standard!r}"
        )


@pytest.mark.parametrize("model", NO_PUBLISHED_PRIORITY_LONG_CONTEXT)
def test_no_priority_long_context_rates_where_openai_publishes_none(model: str) -> None:
    """Guard against back-filling a rate OpenAI does not publish."""
    info = _load(MAIN_PATH)[model]
    assert "input_cost_per_token_above_272k_tokens_priority" not in info


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
]


@pytest.mark.parametrize("model,tier,input_rate,output_rate", TIERED_COST_CASES)
def test_cost_per_token_bills_long_context_at_the_tier_rate(
    model: str, tier: str, input_rate: float, output_rate: float
) -> None:
    """A prompt over 272K on flex or priority must bill at that tier's long-context rate."""
    input_cost, output_cost = litellm.cost_per_token(
        model=model,
        prompt_tokens=LONG_CONTEXT_PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        service_tier=tier,
    )
    assert input_cost == pytest.approx(LONG_CONTEXT_PROMPT_TOKENS * input_rate)
    assert output_cost == pytest.approx(COMPLETION_TOKENS * output_rate)


@pytest.mark.parametrize("model,tier,input_rate,output_rate", TIERED_COST_CASES)
def test_cost_per_token_tier_differs_from_the_standard_long_context_cost(
    model: str, tier: str, input_rate: float, output_rate: float
) -> None:
    """Flex halves the standard long-context bill and priority doubles it."""
    ratio = 0.5 if tier == "flex" else 2.0
    standard = sum(
        litellm.cost_per_token(
            model=model,
            prompt_tokens=LONG_CONTEXT_PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
        )
    )
    tiered = sum(
        litellm.cost_per_token(
            model=model,
            prompt_tokens=LONG_CONTEXT_PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
            service_tier=tier,
        )
    )
    assert tiered == pytest.approx(standard * ratio)
