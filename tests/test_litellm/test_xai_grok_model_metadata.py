import json
from pathlib import Path

# https://github.com/BerriAI/litellm/issues/38179
# grok-2-era slugs: xAI deprecated them effective 2026-02-28 (same batch date as
# the pre-existing xai/grok-2-vision-1212 annotation) and they now hard-fail with
# "Model not found — retired by xAI".
RETIRED_GROK2_MODELS = [
    "xai/grok-2",
    "xai/grok-2-1212",
    "xai/grok-2-latest",
    "xai/grok-2-vision",
    "xai/grok-2-vision-latest",
    "xai/grok-beta",
    "xai/grok-vision-beta",
]
GROK2_DEPRECATION_DATE = "2026-02-28"

# Per https://docs.x.ai/developers/model-capabilities/text/multi-agent (Limitations):
# "The multi-agent model does not work with the OpenAI Chat Completions API."
RESPONSES_ONLY_MODELS = [
    "xai/grok-4.20-multi-agent-0309",
    "xai/grok-4.20-multi-agent-beta-0309",
]

# Slugs still served by xAI (https://docs.x.ai/developers/models) that must not
# be marked deprecated.
ACTIVE_GROK_MODELS = [
    "xai/grok-4.5",
    "xai/grok-4.6",
    "xai/grok-4.20-0309-reasoning",
]


def _load_model_cost(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def test_retired_grok2_models_are_annotated():
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    model_cost = _load_model_cost(json_path)

    for model in RETIRED_GROK2_MODELS:
        info = model_cost.get(model)
        assert info is not None, f"{model} not found in model_prices_and_context_window.json"
        assert (
            info.get("deprecation_date") == GROK2_DEPRECATION_DATE
        ), f"{model} should carry deprecation_date {GROK2_DEPRECATION_DATE}"


def test_grok_4_20_multi_agent_is_responses_only():
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    model_cost = _load_model_cost(json_path)

    for model in RESPONSES_ONLY_MODELS:
        info = model_cost.get(model)
        assert info is not None, f"{model} not found in model_prices_and_context_window.json"
        assert (
            info.get("mode") == "responses"
        ), f"{model} only works with the xAI Responses API, not Chat Completions"


def test_active_grok_models_are_not_marked_deprecated():
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    model_cost = _load_model_cost(json_path)

    for model in ACTIVE_GROK_MODELS:
        info = model_cost.get(model)
        assert info is not None, f"{model} not found in model_prices_and_context_window.json"
        assert (
            "deprecation_date" not in info
        ), f"{model} is still served by xAI and must not be marked deprecated"


def test_xai_grok_metadata_backup_matches_main():
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    main_cost = _load_model_cost(main_path)
    backup_cost = _load_model_cost(backup_path)

    for model in RETIRED_GROK2_MODELS + RESPONSES_ONLY_MODELS:
        assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup"
