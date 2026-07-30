"""
Regression test for #35250: "Gemini Robotics ER 2 Preview" and "Gemini
Robotics ER 1.6 Preview" were missing from the price maps entirely, so
litellm had no cost data for either model on any provider route.

Pins the published costs (https://ai.google.dev/gemini-api/docs/pricing)
in both the primary price map and the ``litellm/`` backup, for both the
bare (vertex_ai) and ``gemini/``-prefixed routes, and verifies
``get_model_info`` surfaces them.
"""

import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm

MODELS = {
    "gemini-robotics-er-1.6-preview": {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 5e-06,
        "litellm_provider": "vertex_ai-language-models",
    },
    "gemini/gemini-robotics-er-1.6-preview": {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 5e-06,
        "litellm_provider": "gemini",
    },
    "gemini-robotics-er-2-preview": {
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 1e-05,
        "litellm_provider": "vertex_ai-language-models",
    },
    "gemini/gemini-robotics-er-2-preview": {
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 1e-05,
        "litellm_provider": "gemini",
    },
}


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _backup_path() -> str:
    return os.path.join(
        os.path.dirname(litellm.__file__),
        "model_prices_and_context_window_backup.json",
    )


def _main_path() -> str:
    # This test lives at ``tests/test_litellm/``; the primary price map sits at
    # the repo root, two directories up.
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "model_prices_and_context_window.json",
    )


class TestGeminiRoboticsErPricingData:
    """Both price maps must carry all four routes with Google's published
    costs, and output must be more expensive than input on each."""

    def test_backup_has_all_routes(self):
        data = _load_json(_backup_path())
        for model, expected in MODELS.items():
            assert model in data, f"{model} missing from backup price map"
            entry = data[model]
            assert entry["input_cost_per_token"] == expected["input_cost_per_token"]
            assert entry["output_cost_per_token"] == expected["output_cost_per_token"]
            assert entry["litellm_provider"] == expected["litellm_provider"]
            assert entry["output_cost_per_token"] > entry["input_cost_per_token"]

    def test_main_has_all_routes(self):
        data = _load_json(_main_path())
        for model, expected in MODELS.items():
            assert model in data, f"{model} missing from main price map"
            entry = data[model]
            assert entry["input_cost_per_token"] == expected["input_cost_per_token"]
            assert entry["output_cost_per_token"] == expected["output_cost_per_token"]
            assert entry["litellm_provider"] == expected["litellm_provider"]
            assert entry["output_cost_per_token"] > entry["input_cost_per_token"]


class TestGeminiRoboticsErModelInfo:
    """``get_model_info`` must report costs for each route."""

    def test_get_model_info_costs(self):
        original = litellm.model_cost
        try:
            litellm.model_cost = _load_json(_backup_path())
            for model, expected in MODELS.items():
                info = litellm.get_model_info(model)
                assert info["input_cost_per_token"] == expected["input_cost_per_token"]
                assert info["output_cost_per_token"] == expected["output_cost_per_token"]
        finally:
            litellm.model_cost = original
