import json
import typing
from pathlib import Path

import pytest

import litellm
from litellm.types.utils import ModelInfoBase

REALTIME_ONLY_GPT_MODELS = (
    "azure/gpt-realtime-2025-08-28",
    "azure/gpt-realtime-1.5-2026-02-23",
    "azure/gpt-realtime-mini-2025-10-06",
    "gpt-realtime",
    "gpt-realtime-1.5",
    "gpt-realtime-2",
    "gpt-realtime-2.1",
    "gpt-realtime-2.1-mini",
    "gpt-realtime-mini",
    "gpt-realtime-2025-08-28",
    "gpt-realtime-mini-2025-10-06",
    "gpt-realtime-mini-2025-12-15",
)


def _load_cost_map() -> dict:
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        return json.load(f)


def test_realtime_is_a_valid_mode_literal():
    hints = typing.get_type_hints(ModelInfoBase, include_extras=False)
    assert "realtime" in typing.get_args(hints["mode"])


@pytest.mark.parametrize("model", REALTIME_ONLY_GPT_MODELS)
def test_realtime_only_gpt_models_are_mode_realtime(model):
    """These models only serve /v1/realtime and are rejected by /v1/chat/completions
    ("This is not a chat model ..."), so they must not be tagged mode=chat."""
    info = _load_cost_map()[model]
    assert info["supported_endpoints"] == ["/v1/realtime"]
    assert info["mode"] == "realtime"


def test_get_model_info_reports_realtime_mode():
    assert litellm.get_model_info("gpt-realtime-mini")["mode"] == "realtime"


def test_backup_matches_main_for_realtime_models():
    repo_root = Path(__file__).parents[2]
    with open(repo_root / "model_prices_and_context_window.json") as f:
        main_cost = json.load(f)
    with open(repo_root / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup_cost = json.load(f)
    for model in REALTIME_ONLY_GPT_MODELS:
        assert backup_cost.get(model) == main_cost.get(model)
