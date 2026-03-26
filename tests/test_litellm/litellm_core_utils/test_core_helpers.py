"""Tests for litellm_core_utils.core_helpers module."""

import pytest

from litellm.litellm_core_utils.core_helpers import map_finish_reason, reconstruct_model_name


# ---------------------------------------------------------------------------
# map_finish_reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Anthropic-specific mappings
        ("stop_sequence", "stop"),
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
        ("refusal", "content_filter"),  # Anthropic refuses to respond
        ("compaction", "length"),
        # Cohere
        ("COMPLETE", "stop"),
        ("MAX_TOKENS", "length"),
        ("ERROR_TOXIC", "content_filter"),
        ("ERROR", "stop"),
        # Vertex AI
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("STOP", "stop"),
        ("MALFORMED_FUNCTION_CALL", "malformed_function_call"),
        # HuggingFace
        ("eos_token", "stop"),
        # Pass-through for unknown reasons
        ("unknown_reason", "unknown_reason"),
    ],
)
def test_map_finish_reason(raw: str, expected: str):
    """map_finish_reason should normalise provider-specific finish reasons."""
    assert map_finish_reason(raw) == expected


def test_map_finish_reason_anthropic_refusal_returns_content_filter():
    """Anthropic 'refusal' finish reason must map to OpenAI 'content_filter'."""
    result = map_finish_reason("refusal")
    assert result == "content_filter", (
        f"Expected 'content_filter' for Anthropic 'refusal', got {result!r}"
    )


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
