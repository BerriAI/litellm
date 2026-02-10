"""
Tests for model cost map resilience.

Simulates:
- A bad (invalid JSON) model cost map upstream
- A bad (empty/missing) backup model cost map
- Verifies litellm.completion() still works even with a broken cost map
- Verifies litellm.get_model_info() raises the expected error for unmapped models
- Verifies the integrity validation helper catches corrupted maps
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

import litellm
from litellm.litellm_core_utils.get_model_cost_map import (
    GetModelCostMap,
    get_model_cost_map,
)


class TestValidateModelCostMap:
    """Unit tests for the validate_model_cost_map helper."""

    def test_should_reject_non_dict(self):
        """Non-dict fetched map should fail validation."""
        assert GetModelCostMap.validate_model_cost_map(fetched_map="not a dict", backup_map={}) is False

    def test_should_reject_empty_map(self):
        """Empty fetched map should fail validation."""
        assert GetModelCostMap.validate_model_cost_map(fetched_map={}, backup_map={}) is False

    def test_should_reject_too_few_models(self):
        """Fetched map with fewer models than min_model_count should fail."""
        small_map = {f"model-{i}": {} for i in range(5)}
        assert (
            GetModelCostMap.validate_model_cost_map(
                fetched_map=small_map, backup_map={}, min_model_count=10
            )
            is False
        )

    def test_should_reject_significant_shrinkage(self):
        """Fetched map that shrunk >50% vs backup should fail."""
        backup = {f"model-{i}": {} for i in range(100)}
        fetched = {f"model-{i}": {} for i in range(40)}  # 40% of backup
        assert (
            GetModelCostMap.validate_model_cost_map(
                fetched_map=fetched, backup_map=backup, min_model_count=10
            )
            is False
        )

    def test_should_accept_valid_map(self):
        """A fetched map with enough models that hasn't shrunk should pass."""
        backup = {f"model-{i}": {} for i in range(100)}
        fetched = {f"model-{i}": {} for i in range(120)}
        assert (
            GetModelCostMap.validate_model_cost_map(
                fetched_map=fetched, backup_map=backup, min_model_count=10
            )
            is True
        )

    def test_should_accept_equal_size_map(self):
        """A fetched map equal in size to backup should pass."""
        backup = {f"model-{i}": {} for i in range(100)}
        fetched = {f"model-{i}": {} for i in range(100)}
        assert (
            GetModelCostMap.validate_model_cost_map(
                fetched_map=fetched, backup_map=backup, min_model_count=10
            )
            is True
        )


class TestGetModelCostMapFallback:
    """Tests for get_model_cost_map fallback behavior with bad upstream."""

    def test_should_fallback_to_backup_on_invalid_json(self):
        """When upstream returns invalid JSON, should fall back to local backup."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("bad json", "", 0)

        with patch("httpx.get", return_value=mock_response):
            result = get_model_cost_map("https://fake-url.com/model_prices.json")

        # Should have fallen back to backup — backup always has models
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_should_fallback_to_backup_on_network_error(self):
        """When upstream is unreachable, should fall back to local backup."""
        with patch("httpx.get", side_effect=Exception("Connection refused")):
            result = get_model_cost_map("https://fake-url.com/model_prices.json")

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_should_fallback_when_fetched_map_is_empty(self):
        """When upstream returns valid JSON but empty dict, should fall back."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {}  # empty map

        with patch("httpx.get", return_value=mock_response):
            result = get_model_cost_map("https://fake-url.com/model_prices.json")

        # Should have fallen back to backup since empty map fails validation
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_should_fallback_when_fetched_map_shrinks_dramatically(self):
        """When upstream returns far fewer models than backup, should fall back."""
        tiny_map = {f"model-{i}": {"litellm_provider": "test"} for i in range(11)}
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = tiny_map

        with patch("httpx.get", return_value=mock_response):
            result = get_model_cost_map("https://fake-url.com/model_prices.json")

        # Backup has thousands of models; 11 is a massive shrinkage → fallback
        assert len(result) > 11

    def test_should_use_local_map_when_env_var_set(self):
        """LITELLM_LOCAL_MODEL_COST_MAP=True should skip remote fetch entirely."""
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            with patch("httpx.get") as mock_get:
                result = get_model_cost_map(
                    "https://fake-url.com/model_prices.json"
                )
                mock_get.assert_not_called()

        assert isinstance(result, dict)
        assert len(result) > 0


class TestCompletionWithBadModelCostMap:
    """
    Simulates a bad model cost map and verifies litellm.completion()
    still works (it catches cost map errors silently) and
    litellm.get_model_info() raises the expected error.
    """

    def test_should_raise_model_not_mapped_with_empty_cost_map(self):
        """
        With an empty model cost map, get_model_info should raise
        'This model isn't mapped yet' for any model.
        """
        original = litellm.model_cost
        litellm.model_cost = {}
        try:
            with pytest.raises(Exception, match="This model isn't mapped yet"):
                litellm.get_model_info("azure/gpt-5.2")
        finally:
            litellm.model_cost = original

    def test_should_raise_model_not_mapped_for_missing_model(self):
        """
        With a cost map that has some models but not the requested one,
        get_model_info should raise 'This model isn't mapped yet'.
        """
        original = litellm.model_cost
        litellm.model_cost = {
            "gpt-4": {
                "litellm_provider": "openai",
                "max_tokens": 8192,
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006,
                "mode": "chat",
            }
        }
        try:
            # gpt-4 should work
            info = litellm.get_model_info("gpt-4")
            assert info is not None

            # azure/gpt-5.2 is not in the map — should fail
            with pytest.raises(Exception, match="This model isn't mapped yet"):
                litellm.get_model_info("azure/gpt-5.2")
        finally:
            litellm.model_cost = original

    @pytest.mark.asyncio
    async def test_should_complete_even_with_bad_cost_map(self):
        """
        litellm.completion() should NOT fail even when the model cost map
        is empty. It catches get_model_info errors internally and proceeds
        to the API call. The cost tracking will be wrong, but the LLM
        request itself should succeed.

        This is the key behavior: completion() is resilient to cost map failures.
        """
        original = litellm.model_cost
        litellm.model_cost = {}
        try:
            # This should still work — completion catches model_info errors
            response = litellm.completion(
                model="azure/gpt-4o-mini",
                messages=[{"role": "user", "content": "say hi"}],
                stream=True,
            )
            chunks = []
            for chunk in response:
                chunks.append(chunk)
            assert len(chunks) > 0, "Expected streaming chunks from completion()"
        finally:
            litellm.model_cost = original
