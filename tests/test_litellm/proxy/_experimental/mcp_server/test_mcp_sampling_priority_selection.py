"""
Tests for MCP sampling handler priority-based model selection.

Verifies that _resolve_model_from_preferences honours costPriority,
speedPriority, and intelligencePriority when hints don't match,
per the MCP spec.
"""

from types import SimpleNamespace
from unittest.mock import patch

from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _has_priorities,
    _resolve_model_from_preferences,
    _select_model_by_priority,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prefs(*, hints=None, cost=None, speed=None, intelligence=None):
    """Build a minimal ModelPreferences-like object."""
    return SimpleNamespace(
        hints=hints or [],
        costPriority=cost,
        speedPriority=speed,
        intelligencePriority=intelligence,
    )


# Model info stubs keyed by model name
_MODEL_INFO = {
    "gpt-3.5-turbo": {
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
        "max_output_tokens": 4096,
        "max_tokens": 4096,
        "output_tokens_per_second": 50.0,
    },
    "gpt-4o": {
        "input_cost_per_token": 0.0000025,
        "output_cost_per_token": 0.0000100,
        "max_output_tokens": 16384,
        "max_tokens": 128000,
        "output_tokens_per_second": 60.0,
    },
    "claude-3-opus": {
        "input_cost_per_token": 0.0000150,
        "output_cost_per_token": 0.0000750,
        "max_output_tokens": 4096,
        "max_tokens": 200000,
        "output_tokens_per_second": 20.0,
    },
    "gpt-4o-mini": {
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.0000006,
        "max_output_tokens": 16384,
        "max_tokens": 128000,
        "output_tokens_per_second": 100.0,
    },
}


def _mock_get_model_info(model, **kwargs):
    """Mock litellm.get_model_info using our test data."""
    if model in _MODEL_INFO:
        return _MODEL_INFO[model]
    raise Exception(f"Unknown model: {model}")


# ---------------------------------------------------------------------------
# _has_priorities
# ---------------------------------------------------------------------------


class TestHasPriorities:
    def test_should_return_false_when_no_priorities_set(self):
        prefs = _prefs()
        assert _has_priorities(prefs) is False

    def test_should_return_false_when_all_zero(self):
        prefs = _prefs(cost=0, speed=0, intelligence=0)
        assert _has_priorities(prefs) is False

    def test_should_return_true_when_cost_set(self):
        prefs = _prefs(cost=0.8)
        assert _has_priorities(prefs) is True

    def test_should_return_true_when_intelligence_set(self):
        prefs = _prefs(intelligence=0.5)
        assert _has_priorities(prefs) is True


# ---------------------------------------------------------------------------
# _select_model_by_priority
# ---------------------------------------------------------------------------


class TestSelectModelByPriority:
    """Tests for the priority-based scoring logic."""

    @patch("litellm.get_model_info", side_effect=_mock_get_model_info)
    def test_should_prefer_cheapest_when_cost_priority_high(self, _mock):
        """High costPriority should select the cheapest model."""
        prefs = _prefs(cost=1.0, speed=0, intelligence=0)
        models = ["gpt-3.5-turbo", "gpt-4o", "claude-3-opus", "gpt-4o-mini"]
        result = _select_model_by_priority(models, prefs)
        # gpt-4o-mini has the lowest combined cost
        assert result == "gpt-4o-mini"

    @patch("litellm.get_model_info", side_effect=_mock_get_model_info)
    def test_should_prefer_smartest_when_intelligence_priority_high(self, _mock):
        """High intelligencePriority should select the model with highest max_output_tokens."""
        prefs = _prefs(cost=0, speed=0, intelligence=1.0)
        models = ["gpt-3.5-turbo", "gpt-4o", "claude-3-opus", "gpt-4o-mini"]
        result = _select_model_by_priority(models, prefs)
        # gpt-4o and gpt-4o-mini both have 16384 max_output_tokens (tied)
        # Either is acceptable
        assert result in ("gpt-4o", "gpt-4o-mini")

    @patch("litellm.get_model_info", side_effect=_mock_get_model_info)
    def test_should_balance_cost_and_intelligence(self, _mock):
        """Balanced priorities should pick a middle-ground model."""
        prefs = _prefs(cost=0.5, speed=0, intelligence=0.5)
        models = ["gpt-3.5-turbo", "gpt-4o", "claude-3-opus", "gpt-4o-mini"]
        result = _select_model_by_priority(models, prefs)
        # gpt-4o-mini is cheap AND has high max_output_tokens → best balance
        assert result == "gpt-4o-mini"

    @patch("litellm.get_model_info", side_effect=_mock_get_model_info)
    def test_should_prefer_fastest_when_speed_priority_high(self, _mock):
        """High speedPriority should prefer cheaper (faster proxy) models."""
        prefs = _prefs(cost=0, speed=1.0, intelligence=0)
        models = ["gpt-3.5-turbo", "gpt-4o", "claude-3-opus", "gpt-4o-mini"]
        result = _select_model_by_priority(models, prefs)
        # gpt-4o-mini has lowest cost → fastest proxy
        assert result == "gpt-4o-mini"

    @patch(
        "litellm.get_model_info",
        side_effect=lambda m, **kw: (_ for _ in ()).throw(Exception("no info")),
    )
    def test_should_return_none_when_no_model_info(self, _mock):
        """If get_model_info fails for all models, return None."""
        prefs = _prefs(cost=1.0)
        models = ["unknown-model-1", "unknown-model-2"]
        result = _select_model_by_priority(models, prefs)
        assert result is None

    @patch("litellm.get_model_info", side_effect=_mock_get_model_info)
    def test_should_handle_single_model(self, _mock):
        """Single model should always be returned regardless of priorities."""
        prefs = _prefs(cost=1.0, intelligence=1.0)
        result = _select_model_by_priority(["gpt-4o"], prefs)
        assert result == "gpt-4o"

    def test_speed_priority_is_neutral_when_no_tps_data(self):
        """When no candidate exposes output_tokens_per_second, speedPriority
        must not fall back to context-window size as a latency proxy: that
        biased selection toward the smallest-context model regardless of real
        speed. With a neutral score the tie resolves to the first candidate,
        so the larger-context model listed first is kept."""
        no_tps_info = {
            "big-ctx": {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "max_output_tokens": 100000,
                "max_tokens": 100000,
            },
            "small-ctx": {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "max_output_tokens": 1000,
                "max_tokens": 1000,
            },
        }

        def info(model, **kwargs):
            return no_tps_info[model]

        with patch("litellm.get_model_info", side_effect=info):
            prefs = _prefs(speed=1.0)
            # The inverse-max_output proxy would pick "small-ctx" here; a
            # neutral score keeps the first candidate.
            assert _select_model_by_priority(["big-ctx", "small-ctx"], prefs) == (
                "big-ctx"
            )


# ---------------------------------------------------------------------------
# _resolve_model_from_preferences — priority integration
# ---------------------------------------------------------------------------


class TestResolveModelPriorityIntegration:
    """End-to-end tests for priority selection within _resolve_model_from_preferences."""

    @patch("litellm.get_model_info", side_effect=_mock_get_model_info)
    @patch("litellm.proxy.proxy_server.llm_router", None)
    @patch(
        "litellm.model_list",
        [
            {"model_name": "gpt-3.5-turbo"},
            {"model_name": "gpt-4o"},
            {"model_name": "gpt-4o-mini"},
        ],
    )
    def test_should_use_priority_when_hints_empty(self, _mock_info):
        """With no hints but priorities set, should use priority-based selection."""
        prefs = _prefs(cost=1.0, speed=0, intelligence=0)
        result = _resolve_model_from_preferences(prefs, default_model="gpt-4o")
        # Should pick cheapest, NOT fall through to default_model
        assert result == "gpt-4o-mini"

    @patch("litellm.get_model_info", side_effect=_mock_get_model_info)
    @patch("litellm.proxy.proxy_server.llm_router", None)
    @patch(
        "litellm.model_list",
        [
            {"model_name": "gpt-3.5-turbo"},
            {"model_name": "gpt-4o"},
            {"model_name": "gpt-4o-mini"},
        ],
    )
    def test_should_skip_priority_when_no_priorities_set(self, _mock_info):
        """With no priorities set, should fall through to default_model."""
        prefs = _prefs()  # no priorities
        result = _resolve_model_from_preferences(prefs, default_model="gpt-4o")
        assert result == "gpt-4o"

    @patch("litellm.get_model_info", side_effect=_mock_get_model_info)
    @patch("litellm.proxy.proxy_server.llm_router", None)
    @patch(
        "litellm.model_list",
        [
            {"model_name": "gpt-3.5-turbo"},
            {"model_name": "gpt-4o"},
            {"model_name": "gpt-4o-mini"},
        ],
    )
    def test_should_prefer_hint_over_priority(self, _mock_info):
        """Hints should take precedence over priority-based selection."""
        hints = [SimpleNamespace(name="gpt-4o")]
        prefs = _prefs(hints=hints, cost=1.0)  # cost says cheap, but hint says gpt-4o
        result = _resolve_model_from_preferences(prefs, default_model="gpt-3.5-turbo")
        assert result == "gpt-4o"
