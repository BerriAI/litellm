import json
from pathlib import Path

import pytest

import litellm

ISSUE_MODELS = (
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-opus-4-1",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-3-7-sonnet-20250219",
)


def _load(name):
    path = Path(__file__).parents[2] / name
    with open(path) as f:
        return json.load(f)


def _main():
    return _load("model_prices_and_context_window.json")


def _backup():
    return _load("litellm/model_prices_and_context_window_backup.json")


@pytest.mark.parametrize("model", ISSUE_MODELS)
def test_anthropic_web_search_flag_present(model):
    info = _main().get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"
    assert info["litellm_provider"] == "anthropic"
    assert info["supports_web_search"] is True


@pytest.mark.parametrize("model", ISSUE_MODELS)
def test_supports_web_search_resolves_true(monkeypatch, model):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    assert litellm.supports_web_search(model=model) is True


def test_anthropic_entries_with_search_billing_advertise_web_search():
    """Any Anthropic-direct model priced for the web search tool must set the flag.

    ``search_context_cost_per_query`` is the billing metadata for Anthropic's web
    search tool, so an entry that carries it while omitting ``supports_web_search``
    would be silently excluded by ``supports_web_search`` / router web-search filtering.
    """
    missing = tuple(
        model
        for model, info in _main().items()
        if isinstance(info, dict)
        and info.get("litellm_provider") == "anthropic"
        and "search_context_cost_per_query" in info
        and info.get("supports_web_search") is not True
    )
    assert missing == (), f"Anthropic entries priced for web search but missing the flag: {missing}"


@pytest.mark.parametrize("model", ISSUE_MODELS)
def test_backup_matches_main(model):
    assert _backup().get(model) == _main().get(model), (
        f"{model} differs between main and backup model cost maps"
    )
