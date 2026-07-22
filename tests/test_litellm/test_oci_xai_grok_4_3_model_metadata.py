import json
from pathlib import Path


def test_oci_xai_grok_4_3_model_info():
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    model = "oci/xai.grok-4.3"
    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "oci"
    assert info["mode"] == "chat"

    assert info["input_cost_per_token"] == 1.25e-06
    assert info["output_cost_per_token"] == 2.5e-06
    assert info["cache_read_input_token_cost"] == 2e-07

    assert info["input_cost_per_token_above_200k_tokens"] == 2.5e-06
    assert info["output_cost_per_token_above_200k_tokens"] == 5e-06
    assert info["cache_read_input_token_cost_above_200k_tokens"] == 4e-07

    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 1000000
    assert info["max_tokens"] == 1000000

    assert info["supports_function_calling"] is True
    assert info["supports_vision"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is False


def test_oci_xai_grok_4_3_backup_matches_main():
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    model = "oci/xai.grok-4.3"
    assert backup_cost.get(model) == main_cost.get(
        model
    ), f"{model} differs between main and backup model cost maps"
