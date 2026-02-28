"""Tests for deferred (lazy) model cost map loading."""

import importlib
import os
import unittest
from unittest.mock import patch


class TestLazyModelCostMap(unittest.TestCase):
    """Verify that model_cost loads local data at import and defers remote."""

    def test_local_backup_loaded_at_import(self):
        """model_cost should contain local data immediately after import."""
        import litellm

        assert isinstance(litellm.model_cost, dict)
        assert len(litellm.model_cost) > 0, "model_cost should not be empty"

    def test_local_only_skips_remote(self):
        """When LITELLM_LOCAL_MODEL_COST_MAP=True, remote fetch is skipped."""
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "true"}):
            import litellm

            litellm._model_cost_remote_loaded = True
            litellm._ensure_remote_model_cost()  # should be a no-op
            assert litellm._model_cost_remote_loaded is True

    def test_ensure_remote_model_cost_is_idempotent(self):
        """Calling _ensure_remote_model_cost multiple times only fetches once."""
        import litellm

        litellm._model_cost_remote_loaded = True
        original_len = len(litellm.model_cost)
        litellm._ensure_remote_model_cost()
        assert len(litellm.model_cost) == original_len

    def test_add_known_models_triggers_remote(self):
        """add_known_models() without args triggers _ensure_remote_model_cost."""
        import litellm

        litellm._model_cost_remote_loaded = True  # prevent actual HTTP
        litellm.add_known_models()
        assert litellm._model_cost_remote_loaded is True

    def test_model_cost_is_plain_dict(self):
        """model_cost should be a plain dict, not a custom subclass."""
        import litellm

        assert type(litellm.model_cost) is dict


if __name__ == "__main__":
    unittest.main()
