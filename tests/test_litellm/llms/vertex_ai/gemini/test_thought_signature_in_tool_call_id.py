"""
Tests for embedding thought signatures in tool call IDs for OpenAI client compatibility.

When using OpenAI clients (instead of LiteLLM SDK), provider_specific_fields are not preserved.
This test suite validates that thought signatures can be embedded in tool call IDs and extracted
when converting back to Gemini format.
"""

import pytest
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    THOUGHT_SIGNATURE_SEPARATOR,
    convert_to_gemini_tool_call_invoke,
    _encode_tool_call_id_with_signature,
    _get_thought_signature_from_tool,
)
from litellm.types.llms.vertex_ai import HttpxPartType


def test_encode_decode_tool_call_id_with_signature():
    """Test that thought signatures can be encoded in and decoded from tool call IDs"""
    base_id = "call_abc123"
    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    # Test encoding
    encoded_id = _encode_tool_call_id_with_signature(base_id, test_signature)
    assert THOUGHT_SIGNATURE_SEPARATOR in encoded_id
    assert encoded_id.startswith(base_id)

    # Test decoding using factory function with realistic tool call structure
    tool = {
        "id": encoded_id,
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "arguments": '{"location": "Paris"}',
        },
    }

    extracted_signature = _get_thought_signature_from_tool(tool)
    assert extracted_signature == test_signature

    # Verify base ID is preserved
    decoded_base_id = encoded_id.split(THOUGHT_SIGNATURE_SEPARATOR)[0]
    assert decoded_base_id == base_id


def test_encode_tool_call_id_without_signature():
    """Test that IDs without signatures are returned unchanged"""
    base_id = "call_abc123def456"

    # Encode without signature
    encoded_id = _encode_tool_call_id_with_signature(base_id, None)
    assert encoded_id == base_id
    assert THOUGHT_SIGNATURE_SEPARATOR not in encoded_id

    # Decode ID without signature using factory function
    tool_obj = {"id": base_id, "type": "function"}
    decoded_signature = _get_thought_signature_from_tool(tool_obj)
    assert decoded_signature is None


def test_tool_call_id_includes_signature_in_response():
    """Test that tool call IDs in responses include embedded thought signatures"""
    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    parts_with_signature = [
        HttpxPartType(
            functionCall={
                "name": "get_current_temperature",
                "args": {"location": "Paris"},
            },
            thoughtSignature=test_signature,
        )
    ]

    function, tools, _ = VertexGeminiConfig._transform_parts(
        parts=parts_with_signature,
        cumulative_tool_call_idx=0,
        is_function_call=False,
    )

    # Verify tool call ID includes thought signature
    assert tools is not None
    assert len(tools) == 1
    tool_call_id = tools[0]["id"]
    assert THOUGHT_SIGNATURE_SEPARATOR in tool_call_id

    # Verify we can decode it using the factory function
    tool_obj = {"id": tool_call_id, "type": "function"}
    decoded_sig = _get_thought_signature_from_tool(tool_obj)
    assert decoded_sig == test_signature


def test_get_thought_signature_backward_compatibility():
    """Test that provider_specific_fields still works (backward compatibility)"""
    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    # Test with provider_specific_fields (LiteLLM SDK scenario)
    tool = {
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "arguments": '{"location": "Paris"}',
        },
        "provider_specific_fields": {"thought_signature": test_signature},
    }

    extracted_signature = _get_thought_signature_from_tool(tool)
    assert extracted_signature == test_signature


def test_get_thought_signature_prioritizes_provider_fields():
    """Test that provider_specific_fields takes priority over tool call ID"""
    signature_in_fields = "signature_from_fields"
    signature_in_id = "signature_from_id"

    encoded_id = _encode_tool_call_id_with_signature("call_abc123", signature_in_id)

    tool = {
        "id": encoded_id,
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "arguments": '{"location": "Paris"}',
        },
        "provider_specific_fields": {"thought_signature": signature_in_fields},
    }

    extracted_signature = _get_thought_signature_from_tool(tool)
    # Should prioritize provider_specific_fields
    assert extracted_signature == signature_in_fields


def test_convert_to_gemini_with_embedded_signature():
    """Test that convert_to_gemini_tool_call_invoke extracts signatures from tool call IDs"""
    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    # Create tool call ID with embedded signature (as OpenAI client would send)
    base_id = "call_abc123"
    encoded_id = _encode_tool_call_id_with_signature(base_id, test_signature)

    # Assistant message as sent by OpenAI client (no provider_specific_fields)
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": encoded_id,  # ID has signature embedded
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ],
    }

    gemini_parts = convert_to_gemini_tool_call_invoke(assistant_message)

    # Verify thought signature is extracted and sent to Gemini
    assert len(gemini_parts) == 1
    assert "function_call" in gemini_parts[0]
    assert "thoughtSignature" in gemini_parts[0]
    assert gemini_parts[0]["thoughtSignature"] == test_signature


def test_openai_client_e2e_flow():
    """
    End-to-end test simulating OpenAI client usage:
    1. LiteLLM receives response from Gemini with thought signature
    2. LiteLLM embeds signature in tool call ID
    3. OpenAI client sends message back with same tool call ID
    4. LiteLLM extracts signature from ID and sends to Gemini
    """
    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    # Step 1: Gemini returns function call with thought signature
    gemini_parts = [
        HttpxPartType(
            functionCall={
                "name": "get_current_temperature",
                "args": {"location": "Paris"},
            },
            thoughtSignature=test_signature,
        )
    ]

    # Step 2: LiteLLM transforms to OpenAI format with embedded signature
    function, tools, _ = VertexGeminiConfig._transform_parts(
        parts=gemini_parts,
        cumulative_tool_call_idx=0,
        is_function_call=False,
    )

    assert tools is not None
    assert len(tools) == 1
    tool_call_id = tools[0]["id"]
    assert THOUGHT_SIGNATURE_SEPARATOR in tool_call_id

    # Step 3: OpenAI client sends back assistant message (preserves tool_call_id)
    openai_assistant_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tool_call_id,  # Preserved from response
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ],
    }

    # Step 4: LiteLLM converts back to Gemini format, extracting signature
    gemini_parts_converted = convert_to_gemini_tool_call_invoke(
        openai_assistant_message
    )

    # Verify signature is preserved through the round trip
    assert len(gemini_parts_converted) == 1
    assert "thoughtSignature" in gemini_parts_converted[0]
    assert gemini_parts_converted[0]["thoughtSignature"] == test_signature


def test_parallel_tool_calls_with_signatures():
    """Test that parallel tool calls preserve signatures correctly"""
    signature1 = "signature_for_first_call"
    # Only first call has signature (Gemini behavior for parallel calls)

    gemini_parts = [
        HttpxPartType(
            functionCall={"name": "get_temperature", "args": {"location": "Paris"}},
            thoughtSignature=signature1,
        ),
        HttpxPartType(
            functionCall={"name": "get_temperature", "args": {"location": "London"}},
            # No signature for second parallel call
        ),
    ]

    function, tools, _ = VertexGeminiConfig._transform_parts(
        parts=gemini_parts,
        cumulative_tool_call_idx=0,
        is_function_call=False,
    )

    assert tools is not None
    assert len(tools) == 2

    # First tool call has signature in ID
    assert THOUGHT_SIGNATURE_SEPARATOR in tools[0]["id"]
    sig1 = _get_thought_signature_from_tool({"id": tools[0]["id"], "type": "function"})
    assert sig1 == signature1

    # Second tool call has no signature in ID
    assert THOUGHT_SIGNATURE_SEPARATOR not in tools[1]["id"]
    sig2 = _get_thought_signature_from_tool({"id": tools[1]["id"], "type": "function"})
    assert sig2 is None
