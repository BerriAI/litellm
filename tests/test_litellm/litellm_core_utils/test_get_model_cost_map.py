"""
Tests for litellm/litellm_core_utils/get_model_cost_map.py

Covers the incident where a bad commit to model_prices_and_context_window.json
on GitHub caused live deployments to fail. Two failure modes:

1. Trailing comma made JSON unparseable (caught by existing except block)
2. Valid JSON but structurally broken - missing model entries (caught by new validation)
"""

import json
import os
from unittest.mock import MagicMock, patch

import litellm.litellm_core_utils.get_model_cost_map as mod
from litellm.litellm_core_utils.get_model_cost_map import (
    validate_model_cost_map,
)


class TestValidateModelCostMap:
    """Unit tests for validate_model_cost_map()"""

    def test_rejects_empty_dict(self):
        assert validate_model_cost_map({}) is False

    def test_rejects_non_dict(self):
        assert validate_model_cost_map("not a dict") is False  # type: ignore

    def test_rejects_too_few_entries(self):
        small_map = {"sample_spec": {}, "model-1": {}, "model-2": {}}
        assert validate_model_cost_map(small_map) is False

    def test_accepts_valid_map(self):
        valid_map = {"sample_spec": {}}
        valid_map.update({f"model-{i}": {} for i in range(200)})
        assert validate_model_cost_map(valid_map) is True


class TestGetModelCostMapFallback:
    """
    Tests that get_model_cost_map() falls back to the local backup
    when the remote response is bad.

    Note: litellm.__init__ sets LITELLM_LOCAL_MODEL_COST_MAP=True at import
    time, so we must clear it and use patch.object on the module's httpx
    reference to properly mock the remote fetch path.
    """

    def _mock_get_model_cost_map(self, mock_response):
        """Helper: call get_model_cost_map with a mocked httpx.get."""
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(mod, "httpx") as mock_httpx:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
            mock_httpx.get.return_value = mock_response
            return mod.get_model_cost_map(url="https://example.com/fake.json")

    def test_trailing_comma_invalid_json(self):
        """
        Exact reproduction of incident commit 5cec5c3bef:
        A trailing comma made the entire JSON unparseable.
        httpx.Response.json() raises JSONDecodeError -> fallback should work.
        """
        bad_response = MagicMock()
        bad_response.raise_for_status.return_value = None
        bad_response.json.side_effect = json.JSONDecodeError(
            "Expecting property name enclosed in double quotes",
            '{"supports_native_streaming": true,}',
            35,
        )

        result = self._mock_get_model_cost_map(bad_response)

        # Fallback to local backup — should have real models
        assert "gpt-4" in result
        assert len(result) > 100

    def test_valid_json_but_missing_models(self):
        """
        Upstream JSON is valid but structurally broken
        (missing model entries). response.json() succeeds
        but validation catches it.
        """
        bad_response = MagicMock()
        bad_response.raise_for_status.return_value = None
        bad_response.json.return_value = {"only_one_bad_entry": {"broken": True}}

        result = self._mock_get_model_cost_map(bad_response)

        # Validation rejects it -> fallback to local backup
        assert "gpt-4" in result
        assert len(result) > 100

    def test_valid_json_wrapped_in_extra_dict(self):
        """
        Upstream returns models wrapped in an unexpected extra dict layer.
        e.g., {"data": {"gpt-4": {...}, ...}} instead of {"gpt-4": {...}, ...}
        """
        bad_response = MagicMock()
        bad_response.raise_for_status.return_value = None
        bad_response.json.return_value = {
            "data": {f"model-{i}": {} for i in range(200)}
        }

        result = self._mock_get_model_cost_map(bad_response)

        # Missing sample_spec + only 1 top-level key -> validation rejects
        assert "gpt-4" in result
        assert len(result) > 100
