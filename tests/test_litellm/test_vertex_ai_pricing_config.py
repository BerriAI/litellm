import json
import os

import pytest


def _load_model_data():
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path) as f:
        return json.load(f)


@pytest.mark.parametrize(
    "model_name, expected",
    [
        (
            "vertex_ai/claude-3-5-haiku",
            {
                "input_cost_per_token": 8e-07,
                "output_cost_per_token": 4e-06,
                "cache_creation_input_token_cost": 1e-06,
                "cache_read_input_token_cost": 8e-08,
                "input_cost_per_token_batches": 4e-07,
                "output_cost_per_token_batches": 2e-06,
            },
        ),
        (
            "vertex_ai/claude-3-5-haiku@20241022",
            {
                "input_cost_per_token": 8e-07,
                "output_cost_per_token": 4e-06,
                "cache_creation_input_token_cost": 1e-06,
                "cache_read_input_token_cost": 8e-08,
                "input_cost_per_token_batches": 4e-07,
                "output_cost_per_token_batches": 2e-06,
            },
        ),
        (
            "vertex_ai/claude-3-haiku",
            {
                "cache_creation_input_token_cost": 3e-07,
                "cache_read_input_token_cost": 3e-08,
            },
        ),
        (
            "vertex_ai/claude-3-haiku@20240307",
            {
                "cache_creation_input_token_cost": 3e-07,
                "cache_read_input_token_cost": 3e-08,
            },
        ),
        (
            "vertex_ai/deepseek-ai/deepseek-v3.1-maas",
            {
                "input_cost_per_token": 6e-07,
                "output_cost_per_token": 1.7e-06,
                "cache_read_input_token_cost": 6e-08,
                "input_cost_per_token_batches": 3e-07,
                "output_cost_per_token_batches": 8.5e-07,
            },
        ),
        (
            "vertex_ai/deepseek-ai/deepseek-v3.2-maas",
            {
                "cache_read_input_token_cost": 5.6e-08,
            },
        ),
        (
            "vertex_ai/minimaxai/minimax-m2-maas",
            {
                "cache_read_input_token_cost": 3e-08,
            },
        ),
        (
            "vertex_ai/moonshotai/kimi-k2-thinking-maas",
            {
                "cache_read_input_token_cost": 6e-08,
            },
        ),
        (
            "vertex_ai/mistral-small-2503",
            {
                "input_cost_per_token": 1e-07,
                "output_cost_per_token": 3e-07,
            },
        ),
        (
            "vertex_ai/mistral-small-2503@001",
            {
                "input_cost_per_token": 1e-07,
                "output_cost_per_token": 3e-07,
            },
        ),
        (
            "vertex_ai/openai/gpt-oss-120b-maas",
            {
                "input_cost_per_token": 9e-08,
                "output_cost_per_token": 3.6e-07,
                "input_cost_per_token_batches": 4.5e-08,
                "output_cost_per_token_batches": 1.8e-07,
            },
        ),
        (
            "vertex_ai/openai/gpt-oss-20b-maas",
            {
                "output_cost_per_token": 2.5e-07,
                "cache_read_input_token_cost": 7e-09,
                "input_cost_per_token_batches": 3.5e-08,
                "output_cost_per_token_batches": 1.25e-07,
            },
        ),
        (
            "vertex_ai/qwen/qwen3-235b-a22b-instruct-2507-maas",
            {
                "input_cost_per_token": 2.2e-07,
                "output_cost_per_token": 8.8e-07,
                "input_cost_per_token_batches": 1.1e-07,
                "output_cost_per_token_batches": 4.4e-07,
            },
        ),
        (
            "vertex_ai/qwen/qwen3-coder-480b-a35b-instruct-maas",
            {
                "input_cost_per_token": 2.2e-07,
                "output_cost_per_token": 1.8e-06,
                "cache_read_input_token_cost": 2.2e-08,
                "input_cost_per_token_batches": 1.1e-07,
                "output_cost_per_token_batches": 9e-07,
            },
        ),
    ],
)
def test_vertex_ai_pricing_config_matches_vertex_ai_pricing_page(model_name, expected):
    model_data = _load_model_data()

    assert model_name in model_data, f"Missing model entry: {model_name}"
    for key, value in expected.items():
        assert model_data[model_name][key] == value
