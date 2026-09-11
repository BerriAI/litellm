import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
REGISTRY_PATHS = (
    REPO_ROOT / "model_prices_and_context_window.json",
    REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json",
)

# Public Cerebras catalog: https://api.cerebras.ai/public/v1/models/qwen-3.8-27b
EXPECTED = {
    "input_cost_per_token": 9.9e-07,
    "litellm_provider": "cerebras",
    "max_input_tokens": 65536,
    "max_output_tokens": 32768,
    "max_tokens": 32768,
    "mode": "chat",
    "output_cost_per_token": 1.49e-06,
    "supports_function_calling": True,
    "supports_parallel_function_calling": True,
    "supports_reasoning": True,
    "supports_response_schema": True,
    "supports_tool_choice": True,
    "supports_vision": True,
}


def test_qwen_38_27b_matches_public_cerebras_catalog() -> None:
    for path in REGISTRY_PATHS:
        payload = json.loads(path.read_text())
        entry = payload["cerebras/qwen-3.8-27b"]
        for key, value in EXPECTED.items():
            assert entry[key] == value, f"{path.name} {key}: {entry.get(key)!r} != {value!r}"
        assert entry["source"] == "https://api.cerebras.ai/public/v1/models/qwen-3.8-27b"
