"""
Regression test: ``command-r7b-12-2024`` had its input/output per-token
costs transposed in the model-cost maps (input=1.5e-07 / output=3.75e-08),
even though Cohere publishes $0.0375/1M input and $0.15/1M output, i.e.
output is ~4x input like every other ``command-r`` entry.

These tests pin the corrected values in both the primary price map and the
``litellm/`` backup, and verify ``get_model_info`` surfaces them, so the
swap cannot silently regress.
"""

import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm

MODEL = "command-r7b-12-2024"
EXPECTED_INPUT_COST = 3.75e-08
EXPECTED_OUTPUT_COST = 1.5e-07


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
    # the repo root, two directories up. Resolve it relative to this file so the
    # test works regardless of where ``litellm`` itself is installed (e.g. a pip
    # install into site-packages).
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "model_prices_and_context_window.json",
    )


class TestCommandR7bPricingData:
    """The JSON price maps must carry Cohere's published costs, with output
    more expensive than input."""

    def test_backup_costs_not_swapped(self):
        entry = _load_json(_backup_path())[MODEL]
        assert entry["input_cost_per_token"] == EXPECTED_INPUT_COST
        assert entry["output_cost_per_token"] == EXPECTED_OUTPUT_COST
        assert entry["output_cost_per_token"] > entry["input_cost_per_token"]

    def test_main_costs_not_swapped(self):
        entry = _load_json(_main_path())[MODEL]
        assert entry["input_cost_per_token"] == EXPECTED_INPUT_COST
        assert entry["output_cost_per_token"] == EXPECTED_OUTPUT_COST
        assert entry["output_cost_per_token"] > entry["input_cost_per_token"]


class TestCommandR7bPricingModelInfo:
    """``get_model_info`` must report the corrected, un-swapped costs."""

    def test_get_model_info_costs(self):
        # Patch litellm.model_cost with the local backup so the test is not
        # dependent on the remote fetch hitting a not-yet-merged main branch.
        original = litellm.model_cost
        try:
            litellm.model_cost = _load_json(_backup_path())
            info = litellm.get_model_info(MODEL)
            assert info["input_cost_per_token"] == EXPECTED_INPUT_COST
            assert info["output_cost_per_token"] == EXPECTED_OUTPUT_COST
            assert info["output_cost_per_token"] > info["input_cost_per_token"]
        finally:
            litellm.model_cost = original
