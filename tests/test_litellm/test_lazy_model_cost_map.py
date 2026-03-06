"""Tests for deferred (lazy) model cost map loading."""

import threading
import pytest
from unittest.mock import patch

from litellm.litellm_core_utils.get_model_cost_map import _FetchResult


@pytest.fixture
def isolate_model_cost_state():
    """Save and restore _model_cost_remote_loaded state and provider model sets across tests."""
    import litellm
    from litellm.litellm_core_utils.get_model_cost_map import _cost_map_source_info

    original_state = litellm._model_cost_remote_loaded
    original_last_failure = litellm._model_cost_last_failure_monotonic
    original_cost_dict = dict(litellm.model_cost)  # shallow copy
    original_source = _cost_map_source_info.source
    original_url = _cost_map_source_info.url
    original_is_env_forced = _cost_map_source_info.is_env_forced
    original_fallback_reason = _cost_map_source_info.fallback_reason
    
    # Save all provider model sets that add_known_models might modify
    saved_model_sets = {}
    for attr_name in dir(litellm):
        attr = getattr(litellm, attr_name, None)
        if isinstance(attr, set) and attr_name.endswith("_models"):
            # Save a copy of each provider model set
            saved_model_sets[attr_name] = set(attr)
    
    litellm._model_cost_last_failure_monotonic = 0.0
    yield
    # Restore state after test
    litellm._model_cost_remote_loaded = original_state
    litellm._model_cost_last_failure_monotonic = original_last_failure
    litellm.model_cost.clear()
    litellm.model_cost.update(original_cost_dict)
    _cost_map_source_info.source = original_source
    _cost_map_source_info.url = original_url
    _cost_map_source_info.is_env_forced = original_is_env_forced
    _cost_map_source_info.fallback_reason = original_fallback_reason
    
    # Restore all provider model sets
    for attr_name, original_set in saved_model_sets.items():
        getattr(litellm, attr_name).clear()
        getattr(litellm, attr_name).update(original_set)


class TestLazyModelCostMap:
    """Verify that model_cost loads local data at import and defers remote."""

    def test_local_backup_loaded_at_import(self):
        """model_cost should contain local data immediately after import."""
        import litellm

        assert isinstance(litellm.model_cost, dict)
        assert len(litellm.model_cost) > 0, "model_cost should not be empty"

    def test_local_only_skips_remote(self, isolate_model_cost_state):
        """When _model_cost_remote_loaded is True, _ensure_remote_model_cost is a no-op."""
        import litellm

        litellm._model_cost_remote_loaded = True
        with patch("litellm._get_model_cost_map_with_source") as mock_get:
            litellm._ensure_remote_model_cost()
            mock_get.assert_not_called()

    def test_ensure_remote_idempotent(self, isolate_model_cost_state):
        """Calling _ensure_remote_model_cost multiple times only fetches once."""
        import litellm

        litellm._model_cost_remote_loaded = False

        def fake_remote_get(url):
            return _FetchResult(
                data={"test_idempotent": {"litellm_provider": "openai"}},
                source="remote",
            )

        with patch("litellm._get_model_cost_map_with_source", side_effect=fake_remote_get) as mock_get:
            litellm._ensure_remote_model_cost()
            litellm._ensure_remote_model_cost()
            litellm._ensure_remote_model_cost()
            mock_get.assert_called_once()

    def test_add_known_models_with_arg_skips_remote(self, isolate_model_cost_state):
        """add_known_models(explicit_map) must NOT trigger remote fetch."""
        import litellm

        litellm._model_cost_remote_loaded = False
        litellm.add_known_models(litellm.model_cost)
        assert litellm._model_cost_remote_loaded is False, (
            "passing an explicit map should NOT trigger remote fetch"
        )

    def test_add_known_models_without_arg_uses_current_data(self, isolate_model_cost_state):
        """add_known_models() without args uses current model_cost — no remote fetch."""
        import litellm

        litellm._model_cost_remote_loaded = False
        # Inject a sentinel model to verify add_known_models reads from model_cost
        litellm.model_cost["_test_sentinel_model"] = {"litellm_provider": "openai"}
        litellm.add_known_models()
        assert "_test_sentinel_model" in litellm.open_ai_chat_completion_models, (
            "add_known_models() should populate provider sets from current model_cost"
        )
        assert litellm._model_cost_remote_loaded is False
        # Cleanup
        litellm.open_ai_chat_completion_models.discard("_test_sentinel_model")
        litellm.model_cost.pop("_test_sentinel_model", None)

    def test_silent_local_fallback_does_not_set_flag(self, isolate_model_cost_state):
        """If _get_model_cost_map_with_source returns local source,
        _model_cost_remote_loaded must stay False so retries remain active."""
        import litellm

        litellm._model_cost_remote_loaded = False
        original_keys = set(litellm.model_cost.keys())

        def fake_get(url):
            return _FetchResult(
                data=dict(litellm.model_cost),
                source="local",
                fallback_reason="Remote fetch failed: timeout",
            )

        with patch("litellm._get_model_cost_map_with_source", side_effect=fake_get) as mock_get:
            litellm._ensure_remote_model_cost()
            mock_get.assert_called_once()
            # Flag must NOT be set — only local data was loaded
            assert litellm._model_cost_remote_loaded is False
            # Failure timestamp should be recorded for cooldown
            assert litellm._model_cost_last_failure_monotonic > 0
            # Local data should be unchanged
            assert set(litellm.model_cost.keys()) == original_keys

    def test_model_cost_is_plain_dict(self):
        """model_cost should be a plain dict, not a custom subclass."""
        import litellm

        assert type(litellm.model_cost) is dict

    def test_remote_failure_keeps_local_and_uses_cooldown(self, isolate_model_cost_state):
        """If remote fetch fails, local backup remains and retries respect cooldown."""
        import litellm

        litellm._model_cost_remote_loaded = False
        original_keys = set(litellm.model_cost.keys())
        with patch("litellm._get_model_cost_map_with_source", side_effect=Exception("network")) as mock_get:
            litellm._ensure_remote_model_cost()
            assert set(litellm.model_cost.keys()) == original_keys
            assert litellm._model_cost_remote_loaded is False  # flag NOT set on failure
            # Immediate next call should skip due cooldown
            litellm._ensure_remote_model_cost()
            assert mock_get.call_count == 1
            # Force cooldown expiry, then retry should happen
            litellm._model_cost_last_failure_monotonic = 0.0
            litellm._ensure_remote_model_cost()
            assert mock_get.call_count == 2

    def test_cost_per_token_triggers_remote_fetch(self, isolate_model_cost_state):
        """cost_per_token() should trigger _ensure_remote_model_cost on first use."""
        import litellm
        from litellm.cost_calculator import cost_per_token

        litellm._model_cost_remote_loaded = False
        with patch("litellm._ensure_remote_model_cost") as mock_ensure:
            try:
                cost_per_token(model="gpt-4o", prompt_tokens=10, completion_tokens=5)
            except Exception:
                pass  # model lookup may fail in test env
            mock_ensure.assert_called_once()

    def test_completion_cost_triggers_remote_fetch(self, isolate_model_cost_state):
        """completion_cost() should trigger _ensure_remote_model_cost on first use."""
        import litellm

        litellm._model_cost_remote_loaded = False
        with patch("litellm._ensure_remote_model_cost") as mock_ensure:
            try:
                litellm.completion_cost(model="gpt-4o", prompt="test", completion="test")
            except Exception:
                pass
            mock_ensure.assert_called()  # called at least once (may be >1 via internal cost helpers)

    def test_concurrent_ensure_fetches_only_once(self, isolate_model_cost_state):
        """Only one thread should perform the remote fetch even under contention."""
        import litellm

        litellm._model_cost_remote_loaded = False
        barrier = threading.Barrier(4)

        def fake_remote_get(url):
            return _FetchResult(
                data={"concurrent_test": {"litellm_provider": "openai"}},
                source="remote",
            )

        with patch("litellm._get_model_cost_map_with_source", side_effect=fake_remote_get) as mock_get:

            def worker():
                barrier.wait()
                litellm._ensure_remote_model_cost()

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            for t in threads:
                assert not t.is_alive(), "Thread did not complete within timeout"
            mock_get.assert_called_once()
