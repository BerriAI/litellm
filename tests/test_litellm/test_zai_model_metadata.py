import json
from pathlib import Path


CURRENT_ZAI_MODELS = {
    "zai/glm-5.3",
    "zai/glm-5.2",
    "zai/glm-5.1",
    "zai/glm-5",
    "zai/glm-5-turbo",
    "zai/glm-5v-turbo",
    "zai/glm-4.7",
    "zai/glm-4.7-flash",
    "zai/glm-4.7-flashx",
    "zai/glm-4.6",
    "zai/glm-4.6v",
    "zai/glm-4.5",
    "zai/glm-4.5v",
    "zai/glm-4.5-air",
    "zai/glm-4.5-flash",
}

ZAI_NEW_MODEL_METADATA = {
    "zai/glm-5.3": {
        "input_cost_per_token": 1.4e-06,
        "output_cost_per_token": 4.4e-06,
        "max_input_tokens": 1048576,
        "max_output_tokens": 131072,
    },
    "zai/glm-5.2": {
        "input_cost_per_token": 1.4e-06,
        "output_cost_per_token": 4.4e-06,
        "max_input_tokens": 1048576,
        "max_output_tokens": 131072,
    },
    "zai/glm-5-turbo": {
        "input_cost_per_token": 1.2e-06,
        "output_cost_per_token": 4e-06,
        "max_input_tokens": 200000,
        "max_output_tokens": 131072,
    },
    "zai/glm-5v-turbo": {
        "input_cost_per_token": 1.2e-06,
        "output_cost_per_token": 4e-06,
        "max_input_tokens": 200000,
        "max_output_tokens": 131072,
        "supports_vision": True,
    },
    "zai/glm-4.7-flashx": {
        "input_cost_per_token": 7e-08,
        "output_cost_per_token": 4e-07,
        "max_input_tokens": 200000,
        "max_output_tokens": 131072,
    },
    "zai/glm-4.6v": {
        "input_cost_per_token": 3e-07,
        "output_cost_per_token": 9e-07,
        "max_input_tokens": 128000,
        "max_output_tokens": 32768,
        "supports_vision": True,
    },
}


def test_zai_model_metadata():
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    model_cost = json.loads(json_path.read_text())

    assert CURRENT_ZAI_MODELS.issubset(model_cost)

    for model, expected_metadata in ZAI_NEW_MODEL_METADATA.items():
        assert model in model_cost, f"{model} missing from model_prices_and_context_window.json"
        metadata = model_cost[model]
        assert metadata["litellm_provider"] == "zai"
        assert metadata["mode"] == "chat"
        assert metadata["supports_function_calling"] is True
        assert metadata["supports_reasoning"] is True
        assert metadata["supports_tool_choice"] is True
        assert metadata["supports_prompt_caching"] is True

        for key, value in expected_metadata.items():
            assert metadata[key] == value


def test_zai_model_metadata_backup_matches_main():
    repo_root = Path(__file__).parents[2]
    model_cost = json.loads((repo_root / "model_prices_and_context_window.json").read_text())
    backup_model_cost = json.loads((repo_root / "litellm" / "model_prices_and_context_window_backup.json").read_text())

    for model in ZAI_NEW_MODEL_METADATA:
        assert backup_model_cost[model] == model_cost[model]
