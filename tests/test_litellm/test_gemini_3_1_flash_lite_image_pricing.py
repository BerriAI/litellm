"""Pricing entry for ``gemini-3.1-flash-lite-image`` (Google's Nano Banana 2 Lite).

Google publishes: $0.25/1M input, $1.50/1M text output, and $30/1M image-output
tokens for the Lite image model (https://cloud.google.com/vertex-ai/generative-ai/pricing).
A 1K image is ~1120 output image tokens => ~$0.0336 / image.

Without this entry, ``completion_cost`` raises "model isn't mapped yet" and Vertex
generateContent pass-through cost tracking silently logs $0. These tests pin the
values in both the primary price map and the ``litellm/`` backup, and verify
``get_model_info`` / ``completion_cost`` surface them.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path

import litellm
from litellm import completion_cost
from litellm.types.utils import CompletionTokensDetailsWrapper, ModelResponse, Usage

VARIANTS = [
    "gemini-3.1-flash-lite-image",
    "gemini/gemini-3.1-flash-lite-image",
    "vertex_ai/gemini-3.1-flash-lite-image",
]

EXPECTED = {
    "input_cost_per_token": 2.5e-07,
    "output_cost_per_token": 1.5e-06,
    "output_cost_per_image_token": 3e-05,
    "mode": "image_generation",
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
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "model_prices_and_context_window.json"
    )


class TestGeminiFlashLiteImagePricingData:
    """Both price maps must carry Google's published Nano Banana 2 Lite costs."""

    def test_present_in_both_maps(self):
        main = _load_json(_main_path())
        backup = _load_json(_backup_path())
        for key in VARIANTS:
            for label, data in (("main", main), ("backup", backup)):
                assert key in data, f"{key} missing from {label} JSON"
                entry = data[key]
                for field, value in EXPECTED.items():
                    assert entry[field] == value, f"{key} {field} in {label}: {entry.get(field)} != {value}"

    def test_image_output_pricing_consistent(self):
        """1120 image-output tokens * output_cost_per_image_token == output_cost_per_image."""
        backup = _load_json(_backup_path())
        entry = backup["gemini-3.1-flash-lite-image"]
        assert round(1120 * entry["output_cost_per_image_token"], 6) == entry["output_cost_per_image"]


class TestGeminiFlashLiteImageModelInfo:
    """``get_model_info`` and ``completion_cost`` must report the new costs."""

    def test_get_model_info_and_cost(self):
        original = litellm.model_cost
        try:
            litellm.model_cost = _load_json(_backup_path())
            info = litellm.get_model_info("gemini-3.1-flash-lite-image")
            assert info["input_cost_per_token"] == EXPECTED["input_cost_per_token"]
            assert info["output_cost_per_token"] == EXPECTED["output_cost_per_token"]

            # A 1K image => 1120 output image tokens => ~$0.0336 (billed at
            # output_cost_per_image_token, not the text output rate).
            resp = ModelResponse()
            resp.model = "gemini-3.1-flash-lite-image"
            resp.usage = Usage(
                prompt_tokens=7,
                completion_tokens=1120,
                total_tokens=1127,
                completion_tokens_details=CompletionTokensDetailsWrapper(
                    image_tokens=1120, text_tokens=0
                ),
            )
            cost = completion_cost(
                completion_response=resp,
                model="gemini-3.1-flash-lite-image",
                custom_llm_provider="vertex_ai",
            )
            # 1120 * 3e-5 (image) + 7 * 2.5e-7 (input) == 0.03360175
            assert abs(cost - 0.03360175) < 1e-6, f"unexpected cost {cost}"
        finally:
            litellm.model_cost = original
