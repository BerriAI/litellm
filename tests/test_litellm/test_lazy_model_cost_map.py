"""Tests for LazyModelCostMap — deferred remote HTTP fetch on import."""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from litellm.litellm_core_utils.get_model_cost_map import (
    GetModelCostMap,
    LazyModelCostMap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def local_data():
    """A small local-style cost map for testing."""
    return {
        "gpt-4": {"litellm_provider": "openai", "max_tokens": 8192},
        "claude-3": {"litellm_provider": "anthropic", "max_tokens": 200000},
    }


@pytest.fixture
def remote_data():
    """A small remote-style cost map (superset of local)."""
    return {
        "gpt-4": {"litellm_provider": "openai", "max_tokens": 8192},
        "claude-3": {"litellm_provider": "anthropic", "max_tokens": 200000},
        "gpt-5": {"litellm_provider": "openai", "max_tokens": 128000},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLazyModelCostMapImportPhase:
    """During import phase, no remote HTTP call should be made."""

    def test_should_load_local_data_immediately(self, local_data):
        with patch.object(
            GetModelCostMap, "load_local_model_cost_map", return_value=local_data
        ):
            m = LazyModelCostMap(url="http://example.com/cost.json")
            # Local data is available via dict internals (no remote fetch)
            assert dict.__len__(m) == 2
            assert dict.__contains__(m, "gpt-4")

    def test_should_not_fetch_remote_during_import_phase(self, local_data):
        mock_fetch = MagicMock()
        with (
            patch.object(
                GetModelCostMap, "load_local_model_cost_map", return_value=local_data
            ),
            patch.object(
                GetModelCostMap, "fetch_remote_model_cost_map", mock_fetch
            ),
        ):
            m = LazyModelCostMap(url="http://example.com/cost.json")
            # Access during import phase — should NOT trigger remote fetch
            assert "gpt-4" in m
            assert len(m) == 2
            mock_fetch.assert_not_called()

    def test_should_allow_iteration_during_import_phase(self, local_data):
        mock_fetch = MagicMock()
        with (
            patch.object(
                GetModelCostMap, "load_local_model_cost_map", return_value=local_data
            ),
            patch.object(
                GetModelCostMap, "fetch_remote_model_cost_map", mock_fetch
            ),
        ):
            m = LazyModelCostMap(url="http://example.com/cost.json")
            keys = list(m.keys())
            assert "gpt-4" in keys
            assert "claude-3" in keys
            mock_fetch.assert_not_called()


class TestLazyModelCostMapRemoteFetch:
    """After seal_import_phase(), first access triggers remote fetch."""

    def test_should_fetch_remote_after_seal(self, local_data, remote_data):
        with (
            patch.object(
                GetModelCostMap, "load_local_model_cost_map", return_value=local_data
            ),
            patch.object(
                GetModelCostMap,
                "fetch_remote_model_cost_map",
                return_value=remote_data,
            ),
            patch.object(
                GetModelCostMap,
                "validate_model_cost_map",
                return_value=True,
            ),
        ):
            m = LazyModelCostMap(url="http://example.com/cost.json")
            m.seal_import_phase()

            # First access triggers remote fetch
            assert "gpt-5" in m
            assert len(m) == 3

    def test_should_fetch_remote_only_once(self, local_data, remote_data):
        mock_fetch = MagicMock(return_value=remote_data)
        with (
            patch.object(
                GetModelCostMap, "load_local_model_cost_map", return_value=local_data
            ),
            patch.object(
                GetModelCostMap, "fetch_remote_model_cost_map", mock_fetch
            ),
            patch.object(
                GetModelCostMap, "validate_model_cost_map", return_value=True
            ),
        ):
            m = LazyModelCostMap(url="http://example.com/cost.json")
            m.seal_import_phase()

            # Multiple accesses
            _ = m.get("gpt-4")
            _ = m.get("gpt-5")
            _ = len(m)
            list(m.items())

            # Remote fetch should have been called exactly once
            mock_fetch.assert_called_once()

    def test_should_fallback_to_local_on_network_error(self, local_data):
        with (
            patch.object(
                GetModelCostMap, "load_local_model_cost_map", return_value=local_data
            ),
            patch.object(
                GetModelCostMap,
                "fetch_remote_model_cost_map",
                side_effect=ConnectionError("no network"),
            ),
        ):
            m = LazyModelCostMap(url="http://example.com/cost.json")
            m.seal_import_phase()

            # Should still work with local data
            assert "gpt-4" in m
            assert len(m) == 2

    def test_should_fallback_on_validation_failure(self, local_data, remote_data):
        with (
            patch.object(
                GetModelCostMap, "load_local_model_cost_map", return_value=local_data
            ),
            patch.object(
                GetModelCostMap,
                "fetch_remote_model_cost_map",
                return_value=remote_data,
            ),
            patch.object(
                GetModelCostMap, "validate_model_cost_map", return_value=False
            ),
        ):
            m = LazyModelCostMap(url="http://example.com/cost.json")
            m.seal_import_phase()

            # Validation failed — local data preserved
            assert "gpt-4" in m
            assert "gpt-5" not in m
            assert len(m) == 2


class TestLazyModelCostMapLocalOnly:
    """When LITELLM_LOCAL_MODEL_COST_MAP=True, no remote fetch ever."""

    def test_should_skip_remote_when_env_set(self, local_data):
        mock_fetch = MagicMock()
        with (
            patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}),
            patch.object(
                GetModelCostMap, "load_local_model_cost_map", return_value=local_data
            ),
            patch.object(
                GetModelCostMap, "fetch_remote_model_cost_map", mock_fetch
            ),
        ):
            m = LazyModelCostMap(url="http://example.com/cost.json")
            m.seal_import_phase()

            # Access after seal — should NOT trigger remote
            assert "gpt-4" in m
            assert len(m) == 2
            mock_fetch.assert_not_called()


class TestLazyModelCostMapThreadSafety:
    """Remote fetch should be thread-safe (only one thread fetches)."""

    def test_should_fetch_remote_only_once_under_contention(
        self, local_data, remote_data
    ):
        call_count = {"n": 0}

        def counting_fetch(url, timeout=5):
            call_count["n"] += 1
            return remote_data

        with (
            patch.object(
                GetModelCostMap, "load_local_model_cost_map", return_value=local_data
            ),
            patch.object(
                GetModelCostMap,
                "fetch_remote_model_cost_map",
                side_effect=counting_fetch,
            ),
            patch.object(
                GetModelCostMap, "validate_model_cost_map", return_value=True
            ),
        ):
            m = LazyModelCostMap(url="http://example.com/cost.json")
            m.seal_import_phase()

            barrier = threading.Barrier(10)
            results = []

            def access():
                barrier.wait()
                results.append("gpt-4" in m)

            threads = [threading.Thread(target=access) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert all(results)
            assert call_count["n"] == 1


class TestLazyModelCostMapDictMethods:
    """All dict methods should work correctly after seal."""

    def test_dict_methods(self, local_data, remote_data):
        with (
            patch.object(
                GetModelCostMap, "load_local_model_cost_map", return_value=local_data
            ),
            patch.object(
                GetModelCostMap,
                "fetch_remote_model_cost_map",
                return_value=remote_data,
            ),
            patch.object(
                GetModelCostMap, "validate_model_cost_map", return_value=True
            ),
        ):
            m = LazyModelCostMap(url="http://example.com/cost.json")
            m.seal_import_phase()

            assert m["gpt-4"]["max_tokens"] == 8192
            assert m.get("nonexistent") is None
            assert m.get("nonexistent", "default") == "default"
            assert set(m.keys()) == {"gpt-4", "claude-3", "gpt-5"}
            assert len(list(m.values())) == 3
            assert len(list(m.items())) == 3
            assert bool(m) is True
            c = m.copy()
            assert isinstance(c, dict)
            assert len(c) == 3
