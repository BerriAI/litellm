"""
Unit tests for get_model_cost_map.py validation logic.

These tests ensure that malformed model cost maps are rejected before
they can break LLM calls.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

# Set local mode for tests by default to avoid network calls
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from litellm.litellm_core_utils.get_model_cost_map import (
    validate_model_cost_map,
    get_model_cost_map,
    _load_local_model_cost_map,
    MIN_MODEL_ENTRIES,
    VALID_MODES,
)


class TestValidateModelCostMap:
    """Tests for the validate_model_cost_map function."""

    def test_valid_model_cost_map(self):
        """Test that a valid model cost map passes validation."""
        # Create a minimal valid map with enough entries
        valid_map = {
            "sample_spec": {"litellm_provider": "example"},
        }
        # Add enough model entries to pass minimum check
        for i in range(MIN_MODEL_ENTRIES + 10):
            valid_map[f"model-{i}"] = {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00002,
            }
        
        is_valid, error = validate_model_cost_map(valid_map)
        assert is_valid is True
        assert error is None

    def test_rejects_non_dict(self):
        """Test that non-dict input is rejected."""
        is_valid, error = validate_model_cost_map([])
        assert is_valid is False
        assert "Expected dict" in error

        is_valid, error = validate_model_cost_map("string")
        assert is_valid is False
        assert "Expected dict" in error

    def test_rejects_too_few_entries(self):
        """Test that maps with too few entries are rejected."""
        small_map = {
            "model-1": {"litellm_provider": "openai"},
            "model-2": {"litellm_provider": "openai"},
        }
        
        is_valid, error = validate_model_cost_map(small_map)
        assert is_valid is False
        assert "Too few model entries" in error

    def test_rejects_missing_litellm_provider(self):
        """Test that entries without litellm_provider are flagged."""
        invalid_map = {"sample_spec": {}}
        for i in range(MIN_MODEL_ENTRIES + 10):
            invalid_map[f"model-{i}"] = {
                "mode": "chat",  # Missing litellm_provider
            }
        
        is_valid, error = validate_model_cost_map(invalid_map)
        assert is_valid is False
        assert "litellm_provider" in error

    def test_rejects_invalid_mode(self):
        """Test that invalid mode values are flagged."""
        invalid_map = {"sample_spec": {}}
        for i in range(MIN_MODEL_ENTRIES + 10):
            invalid_map[f"model-{i}"] = {
                "litellm_provider": "openai",
                "mode": "invalid_mode_xyz",  # Invalid mode
            }
        
        is_valid, error = validate_model_cost_map(invalid_map)
        assert is_valid is False
        assert "invalid mode" in error.lower()

    def test_rejects_non_numeric_cost(self):
        """Test that non-numeric cost values are flagged."""
        invalid_map = {"sample_spec": {}}
        for i in range(MIN_MODEL_ENTRIES + 10):
            invalid_map[f"model-{i}"] = {
                "litellm_provider": "openai",
                "input_cost_per_token": "not_a_number",  # Should be numeric
            }
        
        is_valid, error = validate_model_cost_map(invalid_map)
        assert is_valid is False
        assert "non-numeric" in error.lower()

    def test_accepts_all_valid_modes(self):
        """Test that all valid modes are accepted."""
        for mode in VALID_MODES:
            valid_map = {"sample_spec": {}}
            for i in range(MIN_MODEL_ENTRIES + 10):
                valid_map[f"model-{i}"] = {
                    "litellm_provider": "openai",
                    "mode": mode,
                }
            
            is_valid, error = validate_model_cost_map(valid_map)
            assert is_valid is True, f"Mode '{mode}' should be valid but got error: {error}"

    def test_sample_spec_is_ignored(self):
        """Test that sample_spec entry doesn't count toward minimum."""
        # Only sample_spec - should fail due to too few entries
        only_sample = {"sample_spec": {"litellm_provider": "example"}}
        
        is_valid, error = validate_model_cost_map(only_sample)
        assert is_valid is False
        assert "Too few" in error


class TestGetModelCostMap:
    """Tests for the get_model_cost_map function."""

    def test_local_mode_uses_backup(self):
        """Test that local mode uses the backup file."""
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            result = get_model_cost_map("https://fake-url.com/model.json")
            
            # Should return a dict with many entries
            assert isinstance(result, dict)
            assert len(result) > MIN_MODEL_ENTRIES

    def test_local_mode_variations(self):
        """Test that various truthy values enable local mode."""
        for value in ["True", "true", "TRUE", "1", "yes", "YES"]:
            with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": value}):
                # Should not make network request
                with patch("httpx.get") as mock_get:
                    result = get_model_cost_map("https://fake-url.com/model.json")
                    mock_get.assert_not_called()
                    assert isinstance(result, dict)

    @patch("httpx.get")
    def test_fallback_on_network_error(self, mock_get):
        """Test fallback to local on network errors."""
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": ""}):
            mock_get.side_effect = Exception("Network error")
            
            result = get_model_cost_map("https://fake-url.com/model.json")
            
            # Should fall back to local backup
            assert isinstance(result, dict)
            assert len(result) > MIN_MODEL_ENTRIES

    @patch("httpx.get")
    def test_fallback_on_invalid_response(self, mock_get):
        """Test fallback when remote returns invalid data."""
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": ""}):
            # Mock a response with too few entries
            mock_response = MagicMock()
            mock_response.json.return_value = {"only": "one entry"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = get_model_cost_map("https://fake-url.com/model.json")
            
            # Should fall back to local backup due to validation failure
            assert isinstance(result, dict)
            assert len(result) > MIN_MODEL_ENTRIES

    @patch("httpx.get")
    def test_fallback_on_malformed_entry(self, mock_get):
        """Test fallback when remote has malformed entries."""
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": ""}):
            # Create invalid map (missing litellm_provider)
            invalid_map = {}
            for i in range(MIN_MODEL_ENTRIES + 10):
                invalid_map[f"model-{i}"] = {"mode": "chat"}  # Missing litellm_provider
            
            mock_response = MagicMock()
            mock_response.json.return_value = invalid_map
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = get_model_cost_map("https://fake-url.com/model.json")
            
            # Should fall back to local backup due to validation failure
            assert isinstance(result, dict)
            # Local backup should have litellm_provider in entries
            for key, value in list(result.items())[:5]:
                if key != "sample_spec":
                    assert "litellm_provider" in value


class TestLocalBackup:
    """Tests for the local backup file."""

    def test_local_backup_is_valid(self):
        """Test that the bundled local backup passes validation."""
        backup = _load_local_model_cost_map()
        
        is_valid, error = validate_model_cost_map(backup)
        assert is_valid is True, f"Local backup failed validation: {error}"

    def test_local_backup_has_required_models(self):
        """Test that local backup has common models."""
        backup = _load_local_model_cost_map()
        
        # Check for some common models that should always be present
        common_models = [
            "gpt-4",
            "gpt-3.5-turbo", 
            "claude-3-opus-20240229",
        ]
        
        for model in common_models:
            assert model in backup, f"Common model '{model}' missing from backup"

    def test_local_backup_entries_have_provider(self):
        """Test that all entries in local backup have litellm_provider."""
        backup = _load_local_model_cost_map()
        
        for key, value in backup.items():
            if key == "sample_spec":
                continue
            assert "litellm_provider" in value, f"Entry '{key}' missing litellm_provider"


class TestRealWorldScenarios:
    """Tests simulating real-world failure scenarios."""

    @patch("httpx.get")
    def test_scenario_empty_json_response(self, mock_get):
        """Simulate GitHub returning empty JSON object."""
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": ""}):
            mock_response = MagicMock()
            mock_response.json.return_value = {}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = get_model_cost_map("https://fake-url.com/model.json")
            
            # Should fall back, not return empty dict
            assert len(result) > MIN_MODEL_ENTRIES

    @patch("httpx.get")
    def test_scenario_truncated_response(self, mock_get):
        """Simulate GitHub returning truncated response."""
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": ""}):
            # Only 5 entries - way below minimum
            truncated = {f"model-{i}": {"litellm_provider": "openai"} for i in range(5)}
            
            mock_response = MagicMock()
            mock_response.json.return_value = truncated
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = get_model_cost_map("https://fake-url.com/model.json")
            
            # Should fall back due to too few entries
            assert len(result) > MIN_MODEL_ENTRIES

    @patch("httpx.get")
    def test_scenario_corrupted_entry(self, mock_get):
        """Simulate one corrupted entry in otherwise valid response."""
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": ""}):
            # Most entries valid, but some corrupted
            corrupted_map = {}
            for i in range(MIN_MODEL_ENTRIES + 10):
                if i == 5:
                    # Corrupted entry - not a dict
                    corrupted_map[f"model-{i}"] = "corrupted"
                else:
                    corrupted_map[f"model-{i}"] = {"litellm_provider": "openai"}
            
            mock_response = MagicMock()
            mock_response.json.return_value = corrupted_map
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = get_model_cost_map("https://fake-url.com/model.json")
            
            # Should fall back due to corrupted entry
            assert len(result) > MIN_MODEL_ENTRIES
            # All entries in result should be dicts
            for key, value in result.items():
                assert isinstance(value, dict), f"Entry '{key}' is not a dict"
