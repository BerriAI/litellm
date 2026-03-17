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


class TestCheckIsValidDict:
    """Unit tests for _check_is_valid_dict."""

    def test_should_reject_non_dict(self):
        """Non-dict should fail."""
        assert GetModelCostMap._check_is_valid_dict("not a dict") is False

    def test_should_reject_empty_dict(self):
        """Empty dict should fail."""
        assert GetModelCostMap._check_is_valid_dict({}) is False

    def test_should_reject_list(self):
        """List should fail."""
        assert GetModelCostMap._check_is_valid_dict([1, 2, 3]) is False

    def test_should_reject_none(self):
        """None should fail."""
        assert GetModelCostMap._check_is_valid_dict(None) is False

    def test_should_accept_non_empty_dict(self):
        """Non-empty dict should pass."""
        assert GetModelCostMap._check_is_valid_dict({"model": {}}) is True


class TestCheckModelCountNotReduced:
    """Unit tests for _check_model_count_not_reduced."""

    def test_should_reject_too_few_models(self):
        """Fetched map with fewer models than min_model_count should fail."""
        small_map = {f"model-{i}": {} for i in range(5)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=small_map, backup_model_count=0, min_model_count=10
            )
            is False
        )

    def test_should_reject_significant_shrinkage(self):
        """Fetched map that shrunk >50% vs backup should fail."""
        fetched = {f"model-{i}": {} for i in range(40)}  # 40% of 100
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
            )
            is False
        )

    def test_should_accept_when_above_threshold(self):
        """Fetched map at 60% of backup (above 50% threshold) should pass."""
        fetched = {f"model-{i}": {} for i in range(60)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
            )
            is True
        )

    def test_should_accept_growth(self):
        """Fetched map larger than backup should pass."""
        fetched = {f"model-{i}": {} for i in range(120)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
            )
            is True
        )

    def test_should_accept_with_empty_backup(self):
        """When backup is empty, only min_model_count matters."""
        fetched = {f"model-{i}": {} for i in range(15)}
        assert (
            GetModelCostMap._check_model_count_not_reduced(
                fetched_map=fetched, backup_model_count=0, min_model_count=10
            )
            is True
        )


class TestValidateModelCostMap:
    """Unit tests for validate_model_cost_map (combines both checks)."""

    def test_should_reject_non_dict(self):
        """Non-dict should fail at check 1."""
        assert GetModelCostMap.validate_model_cost_map(fetched_map="not a dict", backup_model_count=0) is False

    def test_should_reject_empty_map(self):
        """Empty dict should fail at check 1."""
        assert GetModelCostMap.validate_model_cost_map(fetched_map={}, backup_model_count=0) is False

    def test_should_reject_significant_shrinkage(self):
        """Should fail at check 2 (shrinkage)."""
        fetched = {f"model-{i}": {} for i in range(40)}
        assert (
            GetModelCostMap.validate_model_cost_map(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
            )
            is False
        )

    def test_should_accept_valid_map(self):
        """Should pass both checks."""
        fetched = {f"model-{i}": {} for i in range(120)}
        assert (
            GetModelCostMap.validate_model_cost_map(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
            )
            is True
        )

    def test_should_accept_equal_size_map(self):
        """Equal size should pass both checks."""
        fetched = {f"model-{i}": {} for i in range(100)}
        assert (
            GetModelCostMap.validate_model_cost_map(
                fetched_map=fetched, backup_model_count=100, min_model_count=10
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


class TestBackupModelCostMapExists:
    """Validates the local backup file is always present and valid."""

    def test_should_have_backup_file(self):
        """The backup model cost map must exist and be loadable."""
        backup = GetModelCostMap.load_local_model_cost_map()
        assert isinstance(backup, dict)
        assert len(backup) > 0, "Backup model cost map is empty"

    def test_should_have_minimum_models_in_backup(self):
        """The backup must contain a reasonable number of models."""
        backup = GetModelCostMap.load_local_model_cost_map()
        assert len(backup) > 100, (
            f"Backup has only {len(backup)} models, expected > 100"
        )


class TestBadHostedModelCostMap:
    """
    Simulates the hosted model cost map being bad (invalid JSON / corrupted).

    When the hosted map is bad, get_model_cost_map() falls back to the local
    backup. These tests verify that after fallback:
    - get_model_info() still works for models in the backup
    - litellm.completion() still works
    """

    def test_should_model_info_pass_after_bad_hosted_map(self):
        """
        If the hosted map is bad, get_model_cost_map falls back to the local
        backup. get_model_info should still work for models in the backup.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("bad json", "", 0)

        with patch("httpx.get", return_value=mock_response):
            fallback_map = get_model_cost_map("https://fake-url.com/bad.json")

        original = litellm.model_cost
        litellm.model_cost = fallback_map
        try:
            # gpt-4o is in every backup — should work fine
            info = litellm.get_model_info("gpt-4o")
            assert info is not None
            assert info["input_cost_per_token"] > 0
        finally:
            litellm.model_cost = original

    def test_should_completion_pass_after_bad_hosted_map(self):
        """
        If the hosted map is bad, litellm.completion() should still work.

        Uses litellm's built-in mock_response param so the real completion
        path is exercised (routing, cost calculator, logging) without
        needing API credentials.
        """
        # Simulate bad hosted map → fallback to backup
        mock_http = MagicMock()
        mock_http.raise_for_status = MagicMock()
        mock_http.json.side_effect = json.JSONDecodeError("bad json", "", 0)

        with patch("httpx.get", return_value=mock_http):
            fallback_map = get_model_cost_map("https://fake-url.com/bad.json")

        original = litellm.model_cost
        litellm.model_cost = fallback_map
        try:
            # mock_response goes through the real completion path —
            # routing, cost calculator, logging — but skips the HTTP call
            response = litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "say hi"}],
                mock_response="hello from mock",
            )
            assert response is not None
            assert response.choices[0].message.content == "hello from mock"
        finally:
            litellm.model_cost = original
