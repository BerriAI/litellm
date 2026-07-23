import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import pytest

from litellm.llms.azure_ai.anthropic.output_params_utils import (
    sanitize_azure_anthropic_output_params,
)


class TestSanitizeAzureAnthropicOutputParams:
    def test_drops_effort_for_model_without_supports_output_config(self):
        """Regression: Haiku 4.5 has no supports_output_config in the model map; effort must be stripped.

        See: https://github.com/BerriAI/litellm/issues/27168
        """
        data = {"output_config": {"effort": "medium"}, "max_tokens": 100}
        sanitize_azure_anthropic_output_params(data, "claude-haiku-4-5")
        assert "output_config" not in data

    def test_preserves_effort_for_model_with_supports_output_config(self):
        """Models with supports_output_config (e.g. Sonnet 4.6) must forward effort unchanged."""
        data = {"output_config": {"effort": "medium"}, "max_tokens": 100}
        sanitize_azure_anthropic_output_params(data, "claude-sonnet-4-6")
        assert data["output_config"] == {"effort": "medium"}

    def test_preserves_other_output_config_keys_when_effort_dropped(self):
        """Only effort is removed; other output_config keys are forwarded."""
        data = {"output_config": {"effort": "medium", "other_key": "value"}}
        sanitize_azure_anthropic_output_params(data, "claude-haiku-4-5")
        assert data["output_config"] == {"other_key": "value"}

    def test_no_output_config_is_noop(self):
        data: dict = {"max_tokens": 100}
        sanitize_azure_anthropic_output_params(data, "claude-haiku-4-5")
        assert "output_config" not in data
        assert data["max_tokens"] == 100

    def test_non_dict_output_config_is_dropped(self):
        data = {"output_config": "bad_value"}
        sanitize_azure_anthropic_output_params(data, "claude-haiku-4-5")
        assert "output_config" not in data

    def test_empty_output_config_after_effort_drop_is_removed(self):
        data = {"output_config": {"effort": "high"}}
        sanitize_azure_anthropic_output_params(data, "claude-haiku-4-5")
        assert "output_config" not in data

    def test_output_config_without_effort_is_unchanged(self):
        data = {"output_config": {"other_key": "value"}}
        sanitize_azure_anthropic_output_params(data, "claude-haiku-4-5")
        assert data["output_config"] == {"other_key": "value"}
