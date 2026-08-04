import json
from pathlib import Path
from unittest.mock import patch

import litellm
import pytest

from litellm.cost_calculator import cost_per_token
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import Usage


@pytest.mark.parametrize("model", ["xai/grok-4.3", "xai/grok-4.3-latest"])
def test_xai_grok_4_3_model_info(model):
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "xai"
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
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supports_web_search"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == model.split("/", 1)[1]
    assert provider == "xai"


def test_xai_grok_4_3_backup_matches_main():
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    for model in ("xai/grok-4.3", "xai/grok-4.3-latest"):
        assert backup_cost.get(model) == main_cost.get(
            model
        ), f"{model} differs between main and backup model cost maps"


@pytest.mark.parametrize(
    "model_cost_map_path",
    [
        Path(__file__).parents[2] / "model_prices_and_context_window.json",
        Path(__file__).parents[2] / "litellm" / "model_prices_and_context_window_backup.json",
    ],
)
def test_vertex_ai_grok_4_3_model_info(model_cost_map_path):
    with open(model_cost_map_path) as f:
        model_cost = json.load(f)

    model = "vertex_ai/xai/grok-4.3"
    info = model_cost[model]

    assert info["litellm_provider"] == "vertex_ai"
    assert info["mode"] == "chat"
    assert info["input_cost_per_token"] == 1.25e-06
    assert info["output_cost_per_token"] == 2.5e-06
    assert info["cache_read_input_token_cost"] == 2e-07
    assert info["input_cost_per_token_above_200k_tokens"] == 2.5e-06
    assert info["output_cost_per_token_above_200k_tokens"] == 5e-06
    assert info["cache_read_input_token_cost_above_200k_tokens"] == 4e-07
    assert info["max_input_tokens"] == 200000
    assert info["max_output_tokens"] == 200000
    assert info["max_tokens"] == 200000
    assert info["supported_regions"] == ["global"]
    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True

    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert routed_model == "xai/grok-4.3"
    assert provider == "vertex_ai"


@pytest.mark.parametrize(
    ("prompt_tokens", "input_rate", "output_rate"),
    [
        (200000, 1.25e-06, 2.5e-06),
        (200001, 2.5e-06, 5e-06),
    ],
)
def test_vertex_ai_grok_4_3_tiered_pricing(prompt_tokens, input_rate, output_rate):
    model_cost_map_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(model_cost_map_path) as f:
        model_info = json.load(f)["vertex_ai/xai/grok-4.3"]

    completion_tokens = 10
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    with patch.dict(
        litellm.model_cost,
        {"vertex_ai/xai/grok-4.3": model_info},
    ):
        prompt_cost, completion_cost = cost_per_token(
            model="vertex_ai/xai/grok-4.3",
            custom_llm_provider="vertex_ai",
            prompt_characters=prompt_tokens * 4,
            completion_characters=completion_tokens * 4,
            usage_object=usage,
        )

    assert prompt_cost == pytest.approx(prompt_tokens * input_rate)
    assert completion_cost == pytest.approx(completion_tokens * output_rate)
