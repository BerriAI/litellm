"""Tests for AzureOpenAIGPT5Config._supports_reasoning_effort_level.

Covers the canonical-prefix fallback and the explicit-disable guard
added in the fix for issue #31243 (custom Azure deployment names).
"""

import pytest
from litellm.llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config


@pytest.mark.parametrize(
    "model",
    [
        "azure/gpt-5.5-pro",
        "azure/gpt-5.4-nano",
        "azure/gpt-5.4-mini-2026-03-17",
    ],
)
def test_supports_reasoning_effort_level_explicitly_disabled_returns_false(model):
    """Registry entries with explicit False short-circuit before canonical fallback fires."""
    assert AzureOpenAIGPT5Config._supports_reasoning_effort_level(model, "none") is False


@pytest.mark.parametrize(
    "model",
    [
        "azure/gpt-5.1_2025-11-13_global",
        "azure/gpt-5.2_custom_deployment",
        "azure/gpt-5.4_preview-eu",
    ],
)
def test_supports_reasoning_effort_level_custom_deployment_falls_back_to_canonical(model):
    """Custom Azure deployment names not in the registry resolve via canonical prefix."""
    assert AzureOpenAIGPT5Config._supports_reasoning_effort_level(model, "none") is True


def test_supports_reasoning_effort_level_gpt5_no_minor_version_canonical():
    """Deployment with gpt-5 but no minor version uses canonical azure/gpt-5."""
    # rest = "_enterprise_deploy", rest[0] != "." → else branch → canonical = azure/gpt-5
    # azure/gpt-5 has no supports_none_reasoning_effort → returns False
    result = AzureOpenAIGPT5Config._supports_reasoning_effort_level(
        "azure/gpt-5_enterprise_deploy", "none"
    )
    assert result is False


def test_supports_reasoning_effort_level_non_gpt5_deployment_returns_false():
    """Non-gpt-5 custom deployment names reach the final return False."""
    # deployment_name "gpt-4_custom_2025" does not start with "gpt-5"
    result = AzureOpenAIGPT5Config._supports_reasoning_effort_level(
        "azure/gpt-4_custom_2025", "none"
    )
    assert result is False
