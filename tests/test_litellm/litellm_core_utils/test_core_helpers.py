"""Tests for litellm_core_utils.core_helpers module."""

import pytest

from litellm.litellm_core_utils.core_helpers import map_finish_reason, reconstruct_model_name


def test_reconstruct_model_name_prefers_deployment_value():
    """Ensure deployment metadata wins when reconstructing the model name."""

    metadata = {"deployment": "vertex_ai/gemini-1.5-flash"}

    result = reconstruct_model_name(
        model_name="gemini-1.5-flash",
        custom_llm_provider="vertex_ai",
        metadata=metadata,
    )

    assert result == "vertex_ai/gemini-1.5-flash"


def test_reconstruct_model_name_adds_bedrock_prefix_when_missing():
    """Bedrock model names without prefixes should gain the provider prefix."""

    metadata = {}

    result = reconstruct_model_name(
        model_name="us.anthropic.claude-3-sonnet",
        custom_llm_provider="bedrock",
        metadata=metadata,
    )

    assert result == "bedrock/us.anthropic.claude-3-sonnet"


def test_reconstruct_model_name_returns_original_for_other_providers():
    """Non-Bedrock providers should not prepend anything."""

    metadata = {}

    result = reconstruct_model_name(
        model_name="claude-3-sonnet",
        custom_llm_provider="anthropic",
        metadata=metadata,
    )

    assert result == "claude-3-sonnet"


@pytest.mark.parametrize(
    "finish_reason, expected",
    [
        ("stop_sequence", "stop"),
        ("content_filtered", "content_filter"),
        ("ERROR_TOXIC", "content_filter"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
    ],
)
def test_map_finish_reason_maps_provider_values(finish_reason: str, expected: str) -> None:
    """Ensure provider-specific finish reasons are normalized to OpenAI-compatible values."""

    assert map_finish_reason(finish_reason) == expected
