import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def model_cost() -> dict[str, Any]:
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def test_replicate_models_have_valid_key_prefix(model_cost: dict[str, Any]) -> None:
    replicate_models = {k for k, v in model_cost.items() if v.get("litellm_provider") == "replicate"}
    malformed = [k for k in replicate_models if not k.startswith("replicate/")]
    assert not malformed, (
        f"Replicate models must use 'replicate/owner/model' key format, found malformed keys: {malformed}"
    )


def test_replicate_openai_gpt_oss_20b_key_exists(model_cost: dict[str, Any]) -> None:
    assert "replicate/openai/gpt-oss-20b" in model_cost
    info = model_cost["replicate/openai/gpt-oss-20b"]
    assert info["litellm_provider"] == "replicate"
    assert info["mode"] == "chat"
    assert info["supports_function_calling"] is True


def test_replicate_backup_matches_main() -> None:
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path, encoding="utf-8") as f:
        main_cost: dict[str, Any] = json.load(f)
    with open(backup_path, encoding="utf-8") as f:
        backup_cost: dict[str, Any] = json.load(f)

    for key in main_cost:
        if main_cost[key].get("litellm_provider") == "replicate":
            assert backup_cost.get(key) == main_cost.get(key), f"{key} differs between main and backup model cost maps"
