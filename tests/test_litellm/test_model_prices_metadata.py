"""
Tests to validate model_prices_and_context_window.json metadata fields.

Ensures all model entries have required metadata fields:
- display_name: Human-readable name
- model_vendor: Vendor/provider identifier (e.g., "openai", "anthropic", "meta")
"""

import json
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def model_prices_data():
    """Load the model prices JSON file."""
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "model_prices_and_context_window.json",
        Path("model_prices_and_context_window.json"),
    ]

    for path in possible_paths:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)

    pytest.fail("Could not find model_prices_and_context_window.json")


class TestModelMetadataPresence:
    """Test that required metadata fields are present."""

    def test_all_models_have_display_name(self, model_prices_data):
        """Every model entry should have a display_name."""
        missing = []
        for model_key, model_data in model_prices_data.items():
            if "display_name" not in model_data:
                missing.append(model_key)

        if missing:
            pytest.fail(
                f"Missing display_name in {len(missing)} models. "
                f"First 10: {missing[:10]}"
            )

    def test_all_models_have_model_vendor(self, model_prices_data):
        """Every model entry should have a model_vendor."""
        missing = []
        for model_key, model_data in model_prices_data.items():
            if "model_vendor" not in model_data:
                missing.append(model_key)

        if missing:
            pytest.fail(
                f"Missing model_vendor in {len(missing)} models. "
                f"First 10: {missing[:10]}"
            )
