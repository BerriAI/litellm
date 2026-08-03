import json
from pathlib import Path

import pytest


def test_kimi_k3_model_info():
    json_path = Path(__file__).parents[4] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    model = "moonshot/kimi-k3"
    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "moonshot"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 3e-06
    assert info["output_cost_per_token"] == 1.5e-05
    assert info["cache_read_input_token_cost"] == 3e-07

    assert info["max_input_tokens"] == 1048576
    assert info["max_output_tokens"] == 1048576
    assert info["max_tokens"] == 1048576

    assert info["supports_function_calling"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_vision"] is True
    assert info["supports_video_input"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_response_schema"] is True


def test_kimi_k3_backup_matches_main():
    repo_root = Path(__file__).parents[4]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    model = "moonshot/kimi-k3"
    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"
