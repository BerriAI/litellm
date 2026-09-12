import json
from pathlib import Path

from typing_extensions import get_args, get_type_hints

from litellm.types.utils import ModelInfoBase

REALTIME_ONLY_GPT_MODELS = (
    "azure/gpt-realtime-2025-08-28",
    "azure/gpt-realtime-1.5-2026-02-23",
    "azure/gpt-realtime-mini",
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

REALTIME_ONLY_GPT_MODELS_WITHOUT_ENDPOINTS = (
    "azure/eu/gpt-4o-mini-realtime-preview-2024-12-17",
    "azure/eu/gpt-4o-realtime-preview-2024-10-01",
    "azure/eu/gpt-4o-realtime-preview-2024-12-17",
    "azure/gpt-4o-mini-realtime-preview-2024-12-17",
    "azure/gpt-4o-realtime-preview-2024-10-01",
    "azure/gpt-4o-realtime-preview-2024-12-17",
    "azure/us/gpt-4o-mini-realtime-preview-2024-12-17",
    "azure/us/gpt-4o-realtime-preview-2024-10-01",
    "azure/us/gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-mini-realtime-preview-2024-12-17",
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-realtime-preview-2025-06-03",
)

ALL_REALTIME_ONLY_GPT_MODELS = REALTIME_ONLY_GPT_MODELS + REALTIME_ONLY_GPT_MODELS_WITHOUT_ENDPOINTS


def test_realtime_is_a_valid_mode_literal():
    hints = get_type_hints(ModelInfoBase, include_extras=False)
    assert "realtime" in get_args(hints["mode"])


def test_backup_matches_main_for_realtime_models():
    repo_root = Path(__file__).parents[2]
    with open(repo_root / "model_prices_and_context_window.json") as f:
        main_cost = json.load(f)
    with open(repo_root / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup_cost = json.load(f)
    for model in ALL_REALTIME_ONLY_GPT_MODELS:
        assert backup_cost.get(model) == main_cost.get(model)
