"""Tests for the provider-agnostic responses surface-fallback hooks on
``BaseResponsesAPIConfig``.

The Databricks connector opts into a multi-surface fallback chain by overriding
``get_responses_surface_fallbacks`` / ``should_fallback_on_responses_error``.
These tests lock in the *default* no-op behavior so the hooks stay invisible to
every provider that does not override them (i.e. ``responses()`` keeps using the
single config with no fallback, exactly as before the hooks were added).
"""

import pytest

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig


class TestResponsesSurfaceFallbackDefaults:
    """A provider that does not override the hooks inherits the no-op default."""

    def setup_method(self):
        self.config = OpenAIResponsesAPIConfig()

    @pytest.mark.parametrize(
        "model",
        ["gpt-4o", "gpt-5", "o1-mini", "anything-at-all"],
    )
    def test_no_surface_fallbacks_by_default(self, model):
        # Empty chain == opt-out: responses() uses this single config, no fallback.
        assert self.config.get_responses_surface_fallbacks(model) == []

    @pytest.mark.parametrize(
        "exc",
        [
            Exception("boom"),
            ValueError("bad request"),
            RuntimeError("500 internal error"),
        ],
    )
    def test_never_falls_back_on_error_by_default(self, exc):
        assert self.config.should_fallback_on_responses_error(exc) is False


class TestDatabricksOverridesDefault:
    """Contrast: the Databricks configs DO override the default, so the opt-in is
    what activates the chain (guards against a future edit silently reverting the
    override to the base no-op)."""

    def test_databricks_supervisor_opts_into_a_chain(self):
        from litellm.llms.databricks.responses.transformation import (
            DatabricksSupervisorResponsesAPIConfig,
        )

        chain = DatabricksSupervisorResponsesAPIConfig().get_responses_surface_fallbacks("databricks-claude-sonnet-4-5")
        assert isinstance(chain, list) and len(chain) >= 1
