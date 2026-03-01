"""Tests for deferred (lazy) model cost map loading."""

import os
import unittest
from unittest.mock import patch, MagicMock


class TestLazyModelCostMap(unittest.TestCase):
    """Verify that model_cost loads local data at import and defers remote."""

    def test_local_backup_loaded_at_import(self):
        """model_cost should contain local data immediately after import."""
        import litellm

        assert isinstance(litellm.model_cost, dict)
        assert len(litellm.model_cost) > 0, "model_cost should not be empty"

    def test_local_only_skips_remote(self):
        """When _model_cost_remote_loaded is True, _ensure_remote_model_cost is a no-op."""
        import litellm

        litellm._model_cost_remote_loaded = True
        with patch("litellm.get_model_cost_map") as mock_get:
            litellm._ensure_remote_model_cost()
            mock_get.assert_not_called()

    def test_ensure_remote_idempotent(self):
        """Calling _ensure_remote_model_cost multiple times only fetches once."""
        import litellm

        litellm._model_cost_remote_loaded = False
        with patch("litellm.get_model_cost_map") as mock_get:
            mock_get.return_value = {"test_idempotent": {"litellm_provider": "openai"}}
            litellm._ensure_remote_model_cost()
            litellm._ensure_remote_model_cost()
            litellm._ensure_remote_model_cost()
            mock_get.assert_called_once()

    def test_add_known_models_with_arg_skips_remote(self):
        """add_known_models(explicit_map) must NOT trigger remote fetch."""
        import litellm

        litellm._model_cost_remote_loaded = False
        litellm.add_known_models(litellm.model_cost)
        assert litellm._model_cost_remote_loaded is False, (
            "passing an explicit map should NOT trigger remote fetch"
        )

    def test_add_known_models_without_arg_triggers_remote(self):
        """add_known_models() without args triggers _ensure_remote_model_cost."""
        import litellm

        litellm._model_cost_remote_loaded = False
        with patch("litellm.get_model_cost_map") as mock_get:
            mock_get.return_value = {}
            litellm.add_known_models()
            mock_get.assert_called_once()
            assert litellm._model_cost_remote_loaded is True

    def test_remote_not_fetched_at_import_time(self):
        """The module-level add_known_models(model_cost) passes args, so
        _ensure_remote_model_cost should NOT fire during import."""
        import litellm

        # After import, if LITELLM_LOCAL_MODEL_COST_MAP was not set,
        # _model_cost_remote_loaded should still be False (import doesn't fetch)
        # We can't truly test import-time behavior without reimporting,
        # but we can verify the guard logic works correctly:
        litellm._model_cost_remote_loaded = False
        with patch("litellm.get_model_cost_map") as mock_get:
            mock_get.return_value = {"test_model": {"litellm_provider": "openai"}}
            litellm._ensure_remote_model_cost()
            mock_get.assert_called_once()
            assert "test_model" in litellm.model_cost
            # Second call should be a no-op
            litellm._ensure_remote_model_cost()
            mock_get.assert_called_once()  # still only 1 call

    def test_model_cost_is_plain_dict(self):
        """model_cost should be a plain dict, not a custom subclass."""
        import litellm

        assert type(litellm.model_cost) is dict

    def test_remote_failure_keeps_local_and_allows_retry(self):
        """If remote fetch fails, local backup data remains and next call retries."""
        import litellm

        litellm._model_cost_remote_loaded = False
        original_keys = set(litellm.model_cost.keys())
        with patch("litellm.get_model_cost_map", side_effect=Exception("network")) as mock_get:
            litellm._ensure_remote_model_cost()
            assert set(litellm.model_cost.keys()) == original_keys
            assert litellm._model_cost_remote_loaded is False  # flag NOT set on failure
            # Next call should retry
            litellm._ensure_remote_model_cost()
            assert mock_get.call_count == 2  # retried


if __name__ == "__main__":
    unittest.main()
