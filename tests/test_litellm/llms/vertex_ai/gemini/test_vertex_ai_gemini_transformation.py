from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_gemini_tool_call_result,
)
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
    _transform_request_body,
    check_if_part_exists_in_parts,
    _get_highest_media_resolution,
    _extract_max_media_resolution_from_messages,
)
from litellm.types.llms.vertex_ai import BlobType
from litellm.types.utils import Message


def test_check_if_part_exists_in_parts():
    parts = [
        {"text": "Hello", "thought": True},
        {"text": "World", "thought": False},
    ]
    part = {"text": "Hello", "thought": True}
    new_part = {"text": "Hello World", "thought": True}
    assert check_if_part_exists_in_parts(parts, part)
    assert not check_if_part_exists_in_parts(parts, new_part, ["thought"])
    assert check_if_part_exists_in_parts(parts, new_part, ["text"])


def test_check_if_part_exists_in_parts_camel_case_snake_case():
    """Test that function handles both camelCase and snake_case key variations"""
    # Test snake_case to camelCase matching
    parts_with_snake_case = [
        {
            "function_call": {
                "name": "get_current_weather",
                "args": {"location": "San Francisco, CA"},
            }
        },
        {"text": "Some other content"},
    ]

    part_with_camel_case = {
        "functionCall": {
            "name": "get_current_weather",
            "args": {"location": "San Francisco, CA"},
        }
    }

    # Should find match between function_call and functionCall
    assert check_if_part_exists_in_parts(parts_with_snake_case, part_with_camel_case)

    # Test camelCase to snake_case matching
    parts_with_camel_case = [
        {"functionCall": {"name": "calculate_sum", "args": {"a": 1, "b": 2}}}
    ]

    part_with_snake_case = {
        "function_call": {"name": "calculate_sum", "args": {"a": 1, "b": 2}}
    }

    # Should find match between functionCall and function_call
    assert check_if_part_exists_in_parts(parts_with_camel_case, part_with_snake_case)

    # Test no match when values differ
    part_with_different_values = {
        "function_call": {"name": "different_function", "args": {"x": 5}}
    }

    assert not check_if_part_exists_in_parts(
        parts_with_snake_case, part_with_different_values
    )

    # Test multiple keys with mixed casing
    parts_mixed = [
        {
            "function_call": {"name": "test"},
            "thoughtSignature": "reasoning",
            "text": "content",
        }
    ]

    part_mixed_casing = {
        "functionCall": {"name": "test"},
        "thought_signature": "reasoning",
        "text": "content",
    }

    assert check_if_part_exists_in_parts(parts_mixed, part_mixed_casing)


def test_cached_content_respects_modify_params_for_cache_incompatible_fields():
    """Regression: cachedContent drops system/tools/toolConfig only when modify_params=True."""
    import litellm

    cache_name = "projects/p/locations/us-central1/cachedContents/abc123"
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "hi"},
    ]
    optional_params = {
        "tools": [
            {
                "functionDeclarations": [
                    {"name": "get_weather", "description": "Get weather"},
                ]
            }
        ],
        "tool_choice": {"functionCallingConfig": {"mode": "AUTO"}},
    }

    original_modify_params = litellm.modify_params
    try:
        # With modify_params=False (default), keep fields even with cachedContent.
        litellm.modify_params = False
        result = _transform_request_body(
            messages=list(messages),
            model="gemini-2.5-pro",
            optional_params=dict(optional_params),
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=cache_name,
        )
        assert result.get("cachedContent") == cache_name
        assert "system_instruction" in result
        assert "tools" in result
        assert "toolConfig" in result
        assert "contents" in result

        # With modify_params=True, drop cache-incompatible fields.
        litellm.modify_params = True
        result_modify_true = _transform_request_body(
            messages=list(messages),
            model="gemini-2.5-pro",
            optional_params=dict(optional_params),
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=cache_name,
        )
        assert result_modify_true.get("cachedContent") == cache_name
        assert "system_instruction" not in result_modify_true
        assert "tools" not in result_modify_true
        assert "toolConfig" not in result_modify_true
        assert "contents" in result_modify_true

        # Without cache, fields are always included.
        result_no_cache = _transform_request_body(
            messages=list(messages),
            model="gemini-2.5-pro",
            optional_params=dict(optional_params),
            custom_llm_provider="vertex_ai",
            litellm_params={},
            cached_content=None,
        )
        assert "system_instruction" in result_no_cache
        assert "tools" in result_no_cache
        assert "toolConfig" in result_no_cache
    finally:
        litellm.modify_params = original_modify_params


# Tests for issue #14556: Labels field provider-aware filtering
def test_google_genai_excludes_labels():
    """Test that Google GenAI/AI Studio endpoints exclude labels when custom_llm_provider='gemini'"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {"labels": {"project": "test", "team": "ai"}}
    litellm_params = {}

    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="gemini",
        litellm_params=litellm_params,
        cached_content=None,
    )

    # Google GenAI/AI Studio should NOT include labels
    assert "labels" not in result
    assert "contents" in result


def test_vertex_ai_includes_labels():
    """Test that Vertex AI endpoints include labels when custom_llm_provider='vertex_ai'"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {"labels": {"project": "test", "team": "ai"}}
    litellm_params = {}

    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="vertex_ai",
        litellm_params=litellm_params,
        cached_content=None,
    )

    # Vertex AI SHOULD include labels
    assert "labels" in result
    assert result["labels"] == {"project": "test", "team": "ai"}


def test_service_tier_forwarded_to_vertex_ai():
    """Test that service_tier in optional_params is mapped to serviceTier in request body."""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {"service_tier": "flex"}
    litellm_params = {}

    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="vertex_ai",
        litellm_params=litellm_params,
        cached_content=None,
    )

    assert "serviceTier" in result
    assert result["serviceTier"] == "flex"


def test_extra_body_cache_not_forwarded_to_vertex_ai():
    """
    'cache' inside extra_body is a LiteLLM-internal proxy caching control.
    It must NOT be forwarded to the Vertex AI request body.

    Regression test for: "Invalid JSON payload received. Unknown name \"cache\": Cannot find field."
    Vertex AI enforces a strict JSON schema and rejects any unknown field.
    """
    messages = [{"role": "user", "content": "test"}]
    optional_params = {
        "extra_body": {
            "cache": {"use-cache": True, "ttl": 86400},  # LiteLLM-internal
            "some_vertex_param": "value",  # legitimate provider extra
        },
    }
    litellm_params = {}

    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="vertex_ai",
        litellm_params=litellm_params,
        cached_content=None,
    )

    # 'cache' must be stripped — Vertex AI has no such field
    assert "cache" not in result, (
        "extra_body.cache must not be forwarded to Vertex AI. "
        'Vertex AI rejects it with 400: Unknown name "cache": Cannot find field.'
    )

    # Other legitimate extra_body keys should still pass through
    assert "some_vertex_param" in result
    assert result["some_vertex_param"] == "value"

    # Core request fields must be present
    assert "contents" in result


def test_extra_body_tags_not_forwarded_to_vertex_ai():
    """
    'tags' inside extra_body is a LiteLLM-internal param for logging/tracking.
    It must NOT be forwarded to the Vertex AI request body.
    Documented in litellm_proxy.md: "Send tags by including them in the extra_body parameter"
    """
    messages = [{"role": "user", "content": "test"}]
    optional_params = {
        "extra_body": {
            "tags": ["user:alice", "env:prod"],
            "custom_param": "allowed",
        },
    }
    litellm_params = {}

    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="vertex_ai",
        litellm_params=litellm_params,
        cached_content=None,
    )

    assert "tags" not in result
    assert "custom_param" in result
    assert result["custom_param"] == "allowed"


def test_metadata_to_labels_vertex_only():
    """Test that metadata->labels conversion only happens for Vertex AI"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {}
    litellm_params = {
        "metadata": {
            "requester_metadata": {"user": "john_doe", "project": "test-project"}
        }
    }

    # Google GenAI/AI Studio should not include labels from metadata
    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params.copy(),
        custom_llm_provider="gemini",
        litellm_params=litellm_params.copy(),
        cached_content=None,
    )
    assert "labels" not in result

    # Vertex AI should include labels from metadata
    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params.copy(),
        custom_llm_provider="vertex_ai",
        litellm_params=litellm_params.copy(),
        cached_content=None,
    )
    assert "labels" in result
    assert result["labels"] == {"user": "john_doe", "project": "test-project"}


def test_empty_content_handling():
    """Test that empty content strings are properly handled in Gemini message transformation"""
    # Test with empty content in user message
    messages = [{"content": "", "role": "user"}]

    contents = _gemini_convert_messages_with_history(messages=messages)

    # Verify that the content was properly transformed
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert len(contents[0]["parts"]) == 1
    assert "text" in contents[0]["parts"][0]
    assert contents[0]["parts"][0]["text"] == ""


def test_thought_signature_extraction_from_response():
    """Test that thought signatures are extracted from Gemini response parts and stored in provider_specific_fields"""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.types.llms.vertex_ai import HttpxPartType

    # Test case: Single function call with thought signature
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

    # Verify thought signature is stored in provider_specific_fields
    assert tools is not None
    assert len(tools) == 1
    assert "provider_specific_fields" in tools[0]
    assert tools[0]["provider_specific_fields"]["thought_signature"] == test_signature


def test_thought_signature_parallel_function_calls():
    """Test that only the first function call in parallel calls has thought signature"""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.types.llms.vertex_ai import HttpxPartType

    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    # Parallel function calls - only first has signature
    parts_parallel = [
        HttpxPartType(
            functionCall={
                "name": "get_current_temperature",
                "args": {"location": "Paris"},
            },
            thoughtSignature=test_signature,  # First FC has signature
        ),
        HttpxPartType(
            functionCall={
                "name": "get_current_temperature",
                "args": {"location": "London"},
            },
            # Second FC has no signature (parallel call)
        ),
    ]

    function, tools, _ = VertexGeminiConfig._transform_parts(
        parts=parts_parallel,
        cumulative_tool_call_idx=0,
        is_function_call=False,
    )

    # Verify only first tool call has thought signature
    assert tools is not None
    assert len(tools) == 2
    assert "provider_specific_fields" in tools[0]
    assert tools[0]["provider_specific_fields"]["thought_signature"] == test_signature
    # Second tool call should not have thought signature
    assert "provider_specific_fields" not in tools[
        1
    ] or "thought_signature" not in tools[1].get("provider_specific_fields", {})


def test_thought_signature_preservation_in_conversion():
    """Test that thought signatures are preserved when converting assistant messages back to Gemini format"""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_invoke,
    )

    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    # Assistant message with tool calls containing thought signatures
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "arguments": '{"location": "Paris"}',
                },
                "index": 0,
                "provider_specific_fields": {
                    "thought_signature": test_signature,
                },
            },
            {
                "id": "call_def456",
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "arguments": '{"location": "London"}',
                },
                "index": 1,
                # No thought signature for parallel call
            },
        ],
    }

    gemini_parts = convert_to_gemini_tool_call_invoke(assistant_message)

    # Verify thought signature is preserved in first function call part
    assert len(gemini_parts) == 2
    assert "function_call" in gemini_parts[0]
    assert "thoughtSignature" in gemini_parts[0]
    assert gemini_parts[0]["thoughtSignature"] == test_signature

    # Verify second function call part does not have thought signature
    assert "function_call" in gemini_parts[1]
    assert "thoughtSignature" not in gemini_parts[1]


def test_thought_signature_sequential_function_calls():
    """Test that each sequential function call preserves its own thought signature"""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_invoke,
    )

    signature_1 = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"
    signature_2 = "DifferentSignatureForSecondCall1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Sequential function calls - each has its own signature
    # This simulates a multi-step conversation where each step has a signature
    assistant_message_step1 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_step1",
                "type": "function",
                "function": {
                    "name": "check_flight",
                    "arguments": '{"flight": "AA100"}',
                },
                "index": 0,
                "provider_specific_fields": {
                    "thought_signature": signature_1,
                },
            },
        ],
    }

    assistant_message_step2 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_step2",
                "type": "function",
                "function": {
                    "name": "book_taxi",
                    "arguments": '{"destination": "airport"}',
                },
                "index": 0,
                "provider_specific_fields": {
                    "thought_signature": signature_2,
                },
            },
        ],
    }

    gemini_parts_step1 = convert_to_gemini_tool_call_invoke(assistant_message_step1)
    gemini_parts_step2 = convert_to_gemini_tool_call_invoke(assistant_message_step2)

    # Verify each step preserves its own signature
    assert len(gemini_parts_step1) == 1
    assert gemini_parts_step1[0]["thoughtSignature"] == signature_1

    assert len(gemini_parts_step2) == 1
    assert gemini_parts_step2[0]["thoughtSignature"] == signature_2


def test_thought_signature_with_function_call_mode():
    """Test thought signature extraction in function_call mode (is_function_call=True)"""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.types.llms.vertex_ai import HttpxPartType

    test_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    parts_with_signature = [
        HttpxPartType(
            functionCall={
                "name": "get_current_weather",
                "args": {"location": "Tokyo"},
            },
            thoughtSignature=test_signature,
        )
    ]

    function, tools, _ = VertexGeminiConfig._transform_parts(
        parts=parts_with_signature,
        cumulative_tool_call_idx=0,
        is_function_call=True,
    )

    # Verify thought signature is stored in function's provider_specific_fields
    assert function is not None
    # Function should be dict-like (TypedDict or dict)
    assert hasattr(function, "__getitem__") or isinstance(function, dict)
    assert "provider_specific_fields" in function
    assert function["provider_specific_fields"]["thought_signature"] == test_signature
    assert tools is None


def test_dummy_signature_added_for_gemini_3_conversation_history():
    """Test that dummy signatures are added when transferring conversation history from older models (like gemini-2.5-flash) to gemini-3."""
    import base64

    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_invoke,
    )

    # Simulate conversation history from gemini-2.5-flash (no thought signature)
    assistant_message_from_older_model = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "arguments": '{"location": "Paris"}',
                },
                "index": 0,
                # No provider_specific_fields - older model doesn't provide signatures
            },
        ],
    }

    # Convert to Gemini format for gemini-3-pro-preview (should add dummy signature)
    gemini_parts = convert_to_gemini_tool_call_invoke(
        assistant_message_from_older_model, model="gemini-3-pro-preview"
    )

    # Verify dummy signature is added
    assert len(gemini_parts) == 1
    assert "function_call" in gemini_parts[0]
    assert "thoughtSignature" in gemini_parts[0]

    # Verify it's the expected dummy signature (base64 encoded "skip_thought_signature_validator")
    expected_dummy = base64.b64encode(b"skip_thought_signature_validator").decode(
        "utf-8"
    )
    assert gemini_parts[0]["thoughtSignature"] == expected_dummy


def test_dummy_signature_not_added_for_gemini_2_5():
    """Test that dummy signatures are NOT added when target model is not gemini-3."""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_invoke,
    )

    # Simulate conversation history from gemini-2.5-flash (no thought signature)
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "arguments": '{"location": "Paris"}',
                },
                "index": 0,
                # No provider_specific_fields
            },
        ],
    }

    # Convert to Gemini format for gemini-2.5-flash (should NOT add dummy signature)
    gemini_parts = convert_to_gemini_tool_call_invoke(
        assistant_message, model="gemini-2.5-flash"
    )

    # Verify no dummy signature is added for non-gemini-3 models
    assert len(gemini_parts) == 1
    assert "function_call" in gemini_parts[0]
    assert "thoughtSignature" not in gemini_parts[0]


def test_dummy_signature_not_added_when_signature_exists():
    """Test that dummy signatures are NOT added when a real signature already exists."""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_invoke,
    )

    real_signature = "Co4CAdHtim/rWgXbz2Ghp4tShzLeMASrPw6JJyYIC3cbVyZnKzU3uv8/wVzyS2sKRPL2m8QQHHXbNQhEEz500G7n/4ZMmksdTtfQcJMoT76S1DGwhnAiLwTgWCNXs3lEb4M19EVYoWFxhrH5Lr9YMIquoU9U4paydGwvZyIyigamIg4B6WnxrRsf0KZV12gJed0DZuKczvOFtHz3zUnmZRlOiTzd5gBVyQM+5jv1VI8m4WUKd6cN/5a5ZvaA0ggiO6kdVhlpIVs7GczSEVJD8KH4u02X7VSnb7CvykqDntZzV0y8rZFBEFGKrChmeHlWXP4D1IB3F9KQyhuLgWImMzg4BajKVxxMU737JGnNISy5"

    # Assistant message with existing thought signature
    assistant_message_with_signature = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "arguments": '{"location": "Paris"}',
                    "provider_specific_fields": {
                        "thought_signature": real_signature,
                    },
                },
                "index": 0,
            },
        ],
    }

    # Convert to Gemini format for gemini-3-pro-preview
    gemini_parts = convert_to_gemini_tool_call_invoke(
        assistant_message_with_signature, model="gemini-3-pro-preview"
    )

    # Verify real signature is preserved, not replaced with dummy
    assert len(gemini_parts) == 1
    assert "function_call" in gemini_parts[0]
    assert "thoughtSignature" in gemini_parts[0]
    assert gemini_parts[0]["thoughtSignature"] == real_signature


def test_dummy_signature_with_function_call_mode():
    """Test that dummy signatures are added for function_call mode when converting to gemini-3."""
    import base64

    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_invoke,
    )

    # Assistant message with function_call (not tool_calls) and no signature
    assistant_message_function_call = {
        "role": "assistant",
        "content": None,
        "function_call": {
            "name": "get_current_temperature",
            "arguments": '{"location": "Paris"}',
            # No provider_specific_fields
        },
    }

    # Convert to Gemini format for gemini-3-pro-preview
    gemini_parts = convert_to_gemini_tool_call_invoke(
        assistant_message_function_call, model="gemini-3-pro-preview"
    )

    # Verify dummy signature is added
    assert len(gemini_parts) == 1
    assert "function_call" in gemini_parts[0]
    assert "thoughtSignature" in gemini_parts[0]

    # Verify it's the expected dummy signature
    expected_dummy = base64.b64encode(b"skip_thought_signature_validator").decode(
        "utf-8"
    )
    assert gemini_parts[0]["thoughtSignature"] == expected_dummy


# Tests for media_resolution (detail parameter) handling - Issue #17084
class TestMediaResolution:
    """Tests for media_resolution handling in Gemini 2.x models"""

    def test_get_highest_media_resolution_high_wins(self):
        """Test that 'high' resolution takes precedence over 'low'"""
        assert _get_highest_media_resolution("low", "high") == "high"
        assert _get_highest_media_resolution("high", "low") == "high"
        assert _get_highest_media_resolution(None, "high") == "high"
        assert _get_highest_media_resolution("high", None) == "high"

    def test_get_highest_media_resolution_low_over_none(self):
        """Test that 'low' resolution takes precedence over None"""
        assert _get_highest_media_resolution(None, "low") == "low"
        assert _get_highest_media_resolution("low", None) == "low"

    def test_get_highest_media_resolution_same_values(self):
        """Test handling of same resolution values"""
        assert _get_highest_media_resolution("high", "high") == "high"
        assert _get_highest_media_resolution("low", "low") == "low"
        assert _get_highest_media_resolution(None, None) is None

    def test_get_highest_media_resolution_medium(self):
        """Test that 'medium' resolution is correctly ranked between 'low' and 'high'"""
        assert _get_highest_media_resolution("low", "medium") == "medium"
        assert _get_highest_media_resolution("medium", "low") == "medium"
        assert _get_highest_media_resolution("medium", "high") == "high"
        assert _get_highest_media_resolution("high", "medium") == "high"
        assert _get_highest_media_resolution(None, "medium") == "medium"
        assert _get_highest_media_resolution("medium", None) == "medium"

    def test_get_highest_media_resolution_ultra_high(self):
        """Test that 'ultra_high' resolution takes precedence over all others"""
        assert _get_highest_media_resolution("high", "ultra_high") == "ultra_high"
        assert _get_highest_media_resolution("ultra_high", "high") == "ultra_high"
        assert _get_highest_media_resolution("medium", "ultra_high") == "ultra_high"
        assert _get_highest_media_resolution("low", "ultra_high") == "ultra_high"
        assert _get_highest_media_resolution(None, "ultra_high") == "ultra_high"
        assert _get_highest_media_resolution("ultra_high", None) == "ultra_high"

    def test_extract_max_media_resolution_single_image_high(self):
        """Test extraction of media resolution from single image with detail=high"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,abc123",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
        assert _extract_max_media_resolution_from_messages(messages) == "high"

    def test_extract_max_media_resolution_single_image_low(self):
        """Test extraction of media resolution from single image with detail=low"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,abc123",
                            "detail": "low",
                        },
                    },
                ],
            }
        ]
        assert _extract_max_media_resolution_from_messages(messages) == "low"

    def test_extract_max_media_resolution_no_detail(self):
        """Test extraction when no detail parameter is provided"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                ],
            }
        ]
        assert _extract_max_media_resolution_from_messages(messages) is None

    def test_extract_max_media_resolution_multiple_images_mixed(self):
        """Test that highest resolution is returned when multiple images have different details"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,abc123",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,def456",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
        assert _extract_max_media_resolution_from_messages(messages) == "high"

    def test_extract_max_media_resolution_text_only(self):
        """Test extraction from messages with no images"""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well!"},
        ]
        assert _extract_max_media_resolution_from_messages(messages) is None

    def test_transform_request_body_gemini_2x_adds_media_resolution(self):
        """Test that media_resolution is added to generationConfig for Gemini 2.x models"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo=",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

        result = _transform_request_body(
            messages=messages,
            model="gemini-2.5-flash",
            optional_params={},
            custom_llm_provider="gemini",
            litellm_params={},
            cached_content=None,
        )

        assert "generationConfig" in result
        assert "mediaResolution" in result["generationConfig"]
        assert result["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_HIGH"

    def test_transform_request_body_gemini_2x_low_resolution(self):
        """Test that low media_resolution is correctly added for Gemini 2.x"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo=",
                            "detail": "low",
                        },
                    },
                ],
            }
        ]

        result = _transform_request_body(
            messages=messages,
            model="gemini-2.5-flash",
            optional_params={},
            custom_llm_provider="gemini",
            litellm_params={},
            cached_content=None,
        )

        assert "generationConfig" in result
        assert "mediaResolution" in result["generationConfig"]
        assert result["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_LOW"

    def test_transform_request_body_gemini_3_no_global_media_resolution(self):
        """Test that Gemini 3 models don't add media_resolution to generationConfig (they use per-part)"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo=",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

        result = _transform_request_body(
            messages=messages,
            model="gemini-3-pro-preview",
            optional_params={},
            custom_llm_provider="gemini",
            litellm_params={},
            cached_content=None,
        )

        # Gemini 3 should NOT have mediaResolution in generationConfig
        # (it's handled per-part in the content transformation)
        if "generationConfig" in result:
            assert "mediaResolution" not in result["generationConfig"]

    def test_transform_request_body_no_detail_no_media_resolution(self):
        """Test that no mediaResolution is added when detail is not specified"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            }
        ]

        result = _transform_request_body(
            messages=messages,
            model="gemini-2.5-flash",
            optional_params={},
            custom_llm_provider="gemini",
            litellm_params={},
            cached_content=None,
        )

        # When no detail is specified, mediaResolution should not be in generationConfig
        if "generationConfig" in result:
            assert "mediaResolution" not in result["generationConfig"]

    def test_extract_max_media_resolution_file_type_with_detail(self):
        """Test that detail is extracted from file content type, not just image_url"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this file?"},
                    {
                        "type": "file",
                        "file": {
                            "url": "data:image/png;base64,abc123",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
        assert _extract_max_media_resolution_from_messages(messages) == "high"

    def test_extract_max_media_resolution_mixed_image_and_file(self):
        """Test that highest detail is returned across both image_url and file types"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,abc123",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "file",
                        "file": {
                            "url": "data:image/png;base64,def456",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
        assert _extract_max_media_resolution_from_messages(messages) == "high"

    def test_transform_request_body_gemini_1x_no_media_resolution(self):
        """Test that Gemini 1.x models don't get mediaResolution in generationConfig"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo=",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

        result = _transform_request_body(
            messages=messages,
            model="gemini-1.5-pro",
            optional_params={},
            custom_llm_provider="gemini",
            litellm_params={},
            cached_content=None,
        )

        # Gemini 1.x should NOT have mediaResolution (not supported)
        if "generationConfig" in result:
            assert "mediaResolution" not in result["generationConfig"]


# Tests for VideoMetadata support across all Gemini models (Issue #25474)
class TestVideoMetadataAllGeminiModels:
    """Tests that video_metadata (fps, start_offset, end_offset) works for all Gemini models"""

    def _make_video_messages(self, video_metadata: dict) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this video"},
                    {
                        "type": "file",
                        "file": {
                            "file_id": "gs://bucket/video.mp4",
                            "format": "video/mp4",
                            "video_metadata": video_metadata,
                        },
                    },
                ],
            }
        ]

    def _get_file_part(self, contents: list) -> dict:
        for part in contents[0]["parts"]:
            if "file_data" in part:
                return part
        raise AssertionError("No file part found in contents")

    def test_video_metadata_fps_gemini_2_5_flash(self):
        """Gemini 2.5 Flash: fps in video_metadata should be forwarded (Issue #25474)"""
        messages = self._make_video_messages({"fps": 5})
        contents = _gemini_convert_messages_with_history(
            messages=messages, model="gemini-2.5-flash"
        )
        file_part = self._get_file_part(contents)
        assert "video_metadata" in file_part
        assert file_part["video_metadata"]["fps"] == 5

    def test_video_metadata_fps_gemini_2_5_pro(self):
        """Gemini 2.5 Pro: fps in video_metadata should be forwarded (Issue #25474)"""
        messages = self._make_video_messages({"fps": 10})
        contents = _gemini_convert_messages_with_history(
            messages=messages, model="gemini-2.5-pro"
        )
        file_part = self._get_file_part(contents)
        assert "video_metadata" in file_part
        assert file_part["video_metadata"]["fps"] == 10

    def test_video_metadata_offsets_gemini_2_5_flash(self):
        """Gemini 2.5 Flash: start_offset/end_offset converted to camelCase (Issue #25474)"""
        messages = self._make_video_messages(
            {"start_offset": "5s", "end_offset": "30s"}
        )
        contents = _gemini_convert_messages_with_history(
            messages=messages, model="gemini-2.5-flash"
        )
        file_part = self._get_file_part(contents)
        assert "video_metadata" in file_part
        vm = file_part["video_metadata"]
        assert vm["startOffset"] == "5s"
        assert vm["endOffset"] == "30s"

    def test_video_metadata_all_fields_gemini_2_5_flash(self):
        """Gemini 2.5 Flash: all video_metadata fields forwarded correctly (Issue #25474)"""
        messages = self._make_video_messages(
            {"fps": 5, "start_offset": "10s", "end_offset": "60s"}
        )
        contents = _gemini_convert_messages_with_history(
            messages=messages, model="gemini-2.5-flash"
        )
        file_part = self._get_file_part(contents)
        assert "video_metadata" in file_part
        vm = file_part["video_metadata"]
        assert vm["fps"] == 5
        assert vm["startOffset"] == "10s"
        assert vm["endOffset"] == "60s"

    def test_video_metadata_gemini_1_5_pro(self):
        """Gemini 1.5 Pro: video_metadata should also be forwarded (Issue #25474)"""
        messages = self._make_video_messages({"fps": 2})
        contents = _gemini_convert_messages_with_history(
            messages=messages, model="gemini-1.5-pro"
        )
        file_part = self._get_file_part(contents)
        assert "video_metadata" in file_part
        assert file_part["video_metadata"]["fps"] == 2


def test_convert_tool_response_with_base64_image():
    """Test tool response with base64 data URI image."""
    # Create a small test image (1x1 red pixel PNG)
    test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    image_data_uri = f"data:image/png;base64,{test_image_base64}"

    # Create tool message with image
    tool_message = {
        "role": "tool",
        "tool_call_id": "call_test123",
        "content": [
            {
                "type": "text",
                "text": '{"url": "https://example.com", "status": "success"}',
            },
            {"type": "input_image", "image_url": image_data_uri},
        ],
    }

    # Mock last message with tool calls
    last_message_with_tool_calls = {
        "tool_calls": [
            {
                "id": "call_test123",
                "function": {"name": "click_at", "arguments": '{"x": 100, "y": 200}'},
            }
        ]
    }

    # Convert tool response (returns list when image is present)
    result = convert_to_gemini_tool_call_result(
        tool_message, last_message_with_tool_calls
    )

    # Verify results - should be a list with 2 parts (function_response + inline_data)
    assert isinstance(
        result, list
    ), f"Expected list when image present, got {type(result)}"
    assert len(result) == 2, f"Expected 2 parts, got {len(result)}"

    # Find function_response part and inline_data part
    function_response_part = None
    inline_data_part = None
    for part in result:
        if "function_response" in part:
            function_response_part = part
        elif "inline_data" in part:
            inline_data_part = part

    # Check function_response exists
    assert function_response_part is not None, "Missing function_response part"
    function_response = function_response_part["function_response"]
    assert function_response["name"] == "click_at"
    assert "response" in function_response
    # Verify JSON response is parsed correctly
    assert "url" in function_response["response"]
    assert function_response["response"]["url"] == "https://example.com"

    # Check inline_data exists
    assert inline_data_part is not None, "Missing inline_data part"
    inline_data: BlobType = inline_data_part["inline_data"]
    assert "data" in inline_data
    assert "mime_type" in inline_data
    assert inline_data["mime_type"] == "image/png"
    assert inline_data["data"] == test_image_base64


def test_convert_tool_response_with_url_image():
    """Test tool response with HTTP URL image (will download and convert)."""
    import pytest

    # Use a publicly accessible test image URL
    test_image_url = "https://via.placeholder.com/1x1.png"

    tool_message = {
        "role": "tool",
        "tool_call_id": "call_test456",
        "content": [
            {"type": "text", "text": '{"url": "https://example.com"}'},
            {"type": "input_image", "image_url": test_image_url},
        ],
    }

    last_message_with_tool_calls = {
        "tool_calls": [
            {
                "id": "call_test456",
                "function": {
                    "name": "type_text_at",
                    "arguments": '{"x": 300, "y": 400, "text": "hello"}',
                },
            }
        ]
    }

    try:
        result = convert_to_gemini_tool_call_result(
            tool_message, last_message_with_tool_calls
        )

        # Should be a list with 2 parts when image is present
        assert isinstance(
            result, list
        ), f"Expected list when image present, got {type(result)}"
        assert len(result) == 2, f"Expected 2 parts, got {len(result)}"

        # Find parts
        function_response_part = next(p for p in result if "function_response" in p)
        inline_data_part = next(p for p in result if "inline_data" in p)

        # Check function_response exists
        assert function_response_part is not None, "Missing function_response part"
        function_response = function_response_part["function_response"]
        assert function_response["name"] == "type_text_at"

        # Check inline_data exists (URL should be downloaded and converted)
        assert inline_data_part is not None, "Missing inline_data part"
        inline_data: BlobType = inline_data_part["inline_data"]
        assert "data" in inline_data
        assert "mime_type" in inline_data
    except Exception as e:
        # Skip test if URL download fails (no internet connection, etc.)
        pytest.skip(f"Failed to download image from URL: {e}")


def test_convert_tool_response_text_only():
    """Test tool response with only text (no image)."""
    tool_message = {
        "role": "tool",
        "tool_call_id": "call_test789",
        "content": [
            {"type": "text", "text": '{"status": "completed", "result": "success"}'}
        ],
    }

    last_message_with_tool_calls = {
        "tool_calls": [
            {
                "id": "call_test789",
                "function": {"name": "wait_5_seconds", "arguments": "{}"},
            }
        ]
    }

    result = convert_to_gemini_tool_call_result(
        tool_message, last_message_with_tool_calls
    )

    # Should be a single part (no list) when no image
    assert not isinstance(result, list), "Should return single part when no image"

    # Check function_response exists
    assert "function_response" in result
    function_response = result["function_response"]
    assert function_response["name"] == "wait_5_seconds"
    # Verify JSON response is parsed correctly
    assert "status" in function_response["response"]
    assert function_response["response"]["status"] == "completed"

    # Check inline_data does NOT exist (no image provided)
    assert "inline_data" not in result


def test_file_data_field_order():
    """
    Test that file_data fields are in the correct order (mime_type before file_uri).

    The Gemini API is sensitive to field order in the file_data object.
    This test verifies that mime_type comes before file_uri in both:
    1. Dictionary key order
    2. JSON serialization

    Related issue: Gemini API returns 400 INVALID_ARGUMENT when fields are in wrong order.
    """
    import json

    from litellm.llms.vertex_ai.gemini.transformation import _process_gemini_media

    # Test with HTTPS URL and explicit format (audio file)
    file_url = "https://generativelanguage.googleapis.com/v1beta/files/test123"
    format = "audio/mpeg"

    result = _process_gemini_media(image_url=file_url, format=format)

    # Verify the result has file_data
    assert "file_data" in result
    file_data = result["file_data"]

    # Verify both fields are present
    assert "mime_type" in file_data
    assert "file_uri" in file_data
    assert file_data["mime_type"] == "audio/mpeg"
    assert file_data["file_uri"] == file_url

    # Verify field order by checking dictionary keys
    # In Python 3.7+, dict maintains insertion order
    file_data_keys = list(file_data.keys())
    assert file_data_keys.index("mime_type") < file_data_keys.index(
        "file_uri"
    ), "mime_type must come before file_uri in the file_data dict"

    # Also verify by serializing to JSON string
    json_str = json.dumps(file_data)
    mime_type_pos = json_str.find('"mime_type"')
    file_uri_pos = json_str.find('"file_uri"')
    assert (
        mime_type_pos < file_uri_pos
    ), "mime_type must appear before file_uri in JSON serialization"


def test_file_data_field_order_gcs_urls():
    """Test that GCS URLs also maintain correct field order."""
    import json

    from litellm.llms.vertex_ai.gemini.transformation import _process_gemini_media

    # Test with GCS URL
    gcs_url = "gs://bucket/audio.mp3"

    result = _process_gemini_media(image_url=gcs_url)

    # Verify the result has file_data
    assert "file_data" in result
    file_data = result["file_data"]

    # Verify both fields are present
    assert "mime_type" in file_data
    assert "file_uri" in file_data

    # Verify field order
    file_data_keys = list(file_data.keys())
    assert file_data_keys.index("mime_type") < file_data_keys.index(
        "file_uri"
    ), "mime_type must come before file_uri in the file_data dict"


def test_gemini_files_api_uri_without_format():
    """
    Test that Gemini Files API URIs work WITHOUT an explicit format/mime_type.

    When a user uploads a file via the Gemini Files API and then references it
    by URI (https://generativelanguage.googleapis.com/v1beta/files/...),
    the file is already on Google's servers. These URLs return 403 when
    fetched directly, so _process_gemini_media must NOT try to resolve the
    MIME type via HTTP.  Instead it should pass the URI through as file_data
    and let the Gemini API resolve the type from its stored metadata.

    Related issue: https://github.com/BerriAI/litellm/issues/24907
    """
    from litellm.llms.vertex_ai.gemini.transformation import _process_gemini_media

    file_url = "https://generativelanguage.googleapis.com/v1beta/files/37eh7rsw1vfe"

    # Should NOT raise — previously this hit the generic https:// handler
    # which called _get_image_mime_type_from_url() and got a 403.
    result = _process_gemini_media(image_url=file_url)

    assert "file_data" in result
    file_data = result["file_data"]
    assert file_data["file_uri"] == file_url
    # When no format is provided, mime_type should be absent so the
    # Gemini API infers it from the stored file metadata.
    assert "mime_type" not in file_data


def test_gemini_files_api_uri_with_format():
    """
    Test that Gemini Files API URIs correctly forward an explicit format.

    Related issue: https://github.com/BerriAI/litellm/issues/24907
    """
    from litellm.llms.vertex_ai.gemini.transformation import _process_gemini_media

    file_url = "https://generativelanguage.googleapis.com/v1beta/files/n1vhxa28lyaw"

    result = _process_gemini_media(image_url=file_url, format="text/plain")

    assert "file_data" in result
    file_data = result["file_data"]
    assert file_data["file_uri"] == file_url
    assert file_data["mime_type"] == "text/plain"


def test_extract_file_data_with_path_object():
    """
    Test that filename is correctly extracted from Path objects for MIME type detection.

    When uploading files using Path objects (e.g., Path("speech.mp3")), the filename
    must be extracted to enable proper MIME type detection. Without this, files get
    uploaded with 'application/octet-stream' instead of the correct MIME type.

    Related issue: Files uploaded with wrong MIME type cause Gemini API to reject
    requests where the specified format doesn't match the uploaded file's MIME type.
    """
    import os
    import tempfile
    from pathlib import Path

    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        extract_file_data,
    )

    # Create a temporary MP3 file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(b"fake mp3 content")
        tmp_path = tmp.name

    try:
        # Test with Path object
        path_obj = Path(tmp_path)
        extracted = extract_file_data(path_obj)

        # Verify filename was extracted
        assert extracted["filename"] is not None
        assert extracted["filename"].endswith(".mp3")

        # Verify MIME type was correctly detected
        assert (
            extracted["content_type"] == "audio/mpeg"
        ), f"Expected 'audio/mpeg' but got '{extracted['content_type']}'"

        # Verify content was read
        assert extracted["content"] == b"fake mp3 content"

    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def test_extract_file_data_with_pathlib_path():
    """Test that filename is correctly extracted from pathlib.Path inputs.
    Bare str paths are rejected — when this runs in a proxy request handler
    the value is attacker-controlled and opening it as a path is an LFI."""
    import os
    import tempfile
    from pathlib import Path

    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        extract_file_data,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(b"fake wav content")
        tmp_path = Path(tmp.name)

    try:
        extracted = extract_file_data(tmp_path)

        assert extracted["filename"] is not None
        assert extracted["filename"].endswith(".wav")
        assert extracted["content_type"] in [
            "audio/wav",
            "audio/x-wav",
        ], f"Expected 'audio/wav' or 'audio/x-wav' but got '{extracted['content_type']}'"
        assert extracted["content"] == b"fake wav content"
    finally:
        os.unlink(str(tmp_path))


def test_extract_file_data_with_tuple_format():
    """Test that tuple format (with explicit content_type) still works correctly."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        extract_file_data,
    )

    # Test with tuple format: (filename, content, content_type)
    filename = "test_audio.mp3"
    content = b"test audio content"
    content_type = "audio/mpeg"

    extracted = extract_file_data((filename, content, content_type))

    # Verify all fields are correct
    assert extracted["filename"] == filename
    assert extracted["content"] == content
    assert extracted["content_type"] == content_type


def test_extract_file_data_fallback_to_octet_stream():
    """Unknown file types fall back to application/octet-stream."""
    import os
    import tempfile
    from pathlib import Path

    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        extract_file_data,
    )

    with tempfile.NamedTemporaryFile(suffix=".xyz123", delete=False) as tmp:
        tmp.write(b"unknown content")
        tmp_path = Path(tmp.name)

    try:
        extracted = extract_file_data(tmp_path)

        assert extracted["filename"] is not None
        assert extracted["filename"].endswith(".xyz123")
        assert (
            extracted["content_type"] == "application/octet-stream"
        ), f"Expected 'application/octet-stream' for unknown type, got '{extracted['content_type']}'"
    finally:
        os.unlink(str(tmp_path))


def test_convert_tool_response_with_pdf_file():
    """Test tool response with PDF file content using file_data field."""
    # Create a minimal test PDF (base64 encoded)
    test_pdf_base64 = "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFI+PgplbmRvYmoKdHJhaWxlcgo8PC9TaXplIDQvUm9vdCAxIDAgUj4+CnN0YXJ0eHJlZgoyMTYKJSVFT0Y="
    file_data_uri = f"data:application/pdf;base64,{test_pdf_base64}"

    # Create tool message with file
    tool_message = {
        "role": "tool",
        "tool_call_id": "call_pdf_test",
        "content": [
            {"type": "text", "text": '{"status": "success", "pages": 1}'},
            {"type": "file", "file_data": file_data_uri},
        ],
    }

    # Mock last message with tool calls
    last_message_with_tool_calls = {
        "tool_calls": [
            {
                "id": "call_pdf_test",
                "function": {
                    "name": "analyze_document",
                    "arguments": '{"path": "/tmp/doc.pdf"}',
                },
            }
        ]
    }

    # Convert tool response (returns list when file is present)
    result = convert_to_gemini_tool_call_result(
        tool_message, last_message_with_tool_calls
    )

    # Verify results - should be a list with 2 parts (function_response + inline_data)
    assert isinstance(
        result, list
    ), f"Expected list when file present, got {type(result)}"
    assert len(result) == 2, f"Expected 2 parts, got {len(result)}"

    # Find function_response part and inline_data part
    function_response_part = None
    inline_data_part = None
    for part in result:
        if "function_response" in part:
            function_response_part = part
        elif "inline_data" in part:
            inline_data_part = part

    # Check function_response exists
    assert function_response_part is not None, "Missing function_response part"
    function_response = function_response_part["function_response"]
    assert function_response["name"] == "analyze_document"
    assert "response" in function_response
    # Verify JSON response is parsed correctly
    assert "status" in function_response["response"]
    assert function_response["response"]["status"] == "success"

    # Check inline_data exists
    assert inline_data_part is not None, "Missing inline_data part"
    inline_data: BlobType = inline_data_part["inline_data"]
    assert "data" in inline_data
    assert "mime_type" in inline_data
    assert inline_data["mime_type"] == "application/pdf"
    assert inline_data["data"] == test_pdf_base64


def test_convert_tool_response_with_input_file_type():
    """Test tool response with input_file content type (Responses API format)."""
    # Create a minimal test PDF (base64 encoded)
    test_pdf_base64 = "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFI+PgplbmRvYmoKdHJhaWxlcgo8PC9TaXplIDQvUm9vdCAxIDAgUj4+CnN0YXJ0eHJlZgoyMTYKJSVFT0Y="
    file_data_uri = f"data:application/pdf;base64,{test_pdf_base64}"

    # Create tool message with input_file type
    tool_message = {
        "role": "tool",
        "tool_call_id": "call_input_file_test",
        "content": [{"type": "input_file", "file_data": file_data_uri}],
    }

    # Mock last message with tool calls
    last_message_with_tool_calls = {
        "tool_calls": [
            {
                "id": "call_input_file_test",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ]
    }

    # Convert tool response
    result = convert_to_gemini_tool_call_result(
        tool_message, last_message_with_tool_calls
    )

    # Verify results
    assert isinstance(
        result, list
    ), f"Expected list when file present, got {type(result)}"
    assert len(result) == 2, f"Expected 2 parts, got {len(result)}"

    # Find inline_data part
    inline_data_part = None
    for part in result:
        if "inline_data" in part:
            inline_data_part = part

    # Check inline_data exists
    assert inline_data_part is not None, "Missing inline_data part"
    assert inline_data_part["inline_data"]["mime_type"] == "application/pdf"


def test_convert_tool_response_with_nested_file_object():
    """Test tool response with file content using nested file object format."""
    # Create a minimal test PDF (base64 encoded)
    test_pdf_base64 = "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFI+PgplbmRvYmoKdHJhaWxlcgo8PC9TaXplIDQvUm9vdCAxIDAgUj4+CnN0YXJ0eHJlZgoyMTYKJSVFT0Y="
    file_data_uri = f"data:application/pdf;base64,{test_pdf_base64}"

    # Create tool message with nested file object (OpenAI Agents SDK format)
    tool_message = {
        "role": "tool",
        "tool_call_id": "call_nested_test",
        "content": [{"type": "file", "file": {"file_data": file_data_uri}}],
    }

    # Mock last message with tool calls
    last_message_with_tool_calls = {
        "tool_calls": [
            {
                "id": "call_nested_test",
                "function": {"name": "process_document", "arguments": "{}"},
            }
        ]
    }

    # Convert tool response
    result = convert_to_gemini_tool_call_result(
        tool_message, last_message_with_tool_calls
    )

    # Verify results - should be a list with 2 parts
    assert isinstance(
        result, list
    ), f"Expected list when file present, got {type(result)}"
    assert len(result) == 2, f"Expected 2 parts, got {len(result)}"

    # Find inline_data part
    inline_data_part = None
    for part in result:
        if "inline_data" in part:
            inline_data_part = part

    # Check inline_data exists
    assert inline_data_part is not None, "Missing inline_data part"
    inline_data: BlobType = inline_data_part["inline_data"]
    assert "data" in inline_data
    assert "mime_type" in inline_data
    assert inline_data["mime_type"] == "application/pdf"
    assert inline_data["data"] == test_pdf_base64


def test_assistant_message_with_images_field():
    """
    Test that assistant messages with images field are properly converted to Gemini format.

    This handles the case where an assistant message contains generated images in the
    `images` field (e.g., from image generation models like gemini-2.5-flash-image).
    The images should be converted to inline_data parts in the Gemini format.
    """
    # Create a small test image (1x1 red pixel PNG)
    test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    image_data_uri = f"data:image/png;base64,{test_image_base64}"

    # Create messages with assistant message containing images field
    messages = [
        {
            "role": "user",
            "content": "Generate an image of a banana wearing a costume that says LiteLLM",
        },
        {
            "role": "assistant",
            "content": "Here's your banana in a LiteLLM costume!",
            "images": [
                {
                    "image_url": {"url": image_data_uri, "detail": "auto"},
                    "index": 0,
                    "type": "image_url",
                }
            ],
        },
    ]

    # Convert messages to Gemini format
    contents = _gemini_convert_messages_with_history(messages=messages)

    # Verify structure
    assert len(contents) == 2, f"Expected 2 content blocks, got {len(contents)}"

    # Verify user message
    assert contents[0]["role"] == "user"
    assert len(contents[0]["parts"]) == 1
    assert (
        contents[0]["parts"][0]["text"]
        == "Generate an image of a banana wearing a costume that says LiteLLM"
    )

    # Verify assistant message
    assert contents[1]["role"] == "model"
    assert (
        len(contents[1]["parts"]) == 2
    ), f"Expected 2 parts (text + image), got {len(contents[1]['parts'])}"

    # Find text part and inline_data part
    text_part = None
    inline_data_part = None
    for part in contents[1]["parts"]:
        if "text" in part:
            text_part = part
        elif "inline_data" in part:
            inline_data_part = part

    # Verify text part
    assert text_part is not None, "Missing text part in assistant message"
    assert text_part["text"] == "Here's your banana in a LiteLLM costume!"

    # Verify inline_data part (image)
    assert inline_data_part is not None, "Missing inline_data part in assistant message"
    inline_data: BlobType = inline_data_part["inline_data"]
    assert "data" in inline_data
    assert "mime_type" in inline_data
    assert inline_data["mime_type"] == "image/png"
    assert inline_data["data"] == test_image_base64


def test_assistant_message_with_multiple_images():
    """Test that assistant messages with multiple images are properly converted."""
    # Create two test images
    test_image1_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    test_image2_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    image1_data_uri = f"data:image/png;base64,{test_image1_base64}"
    image2_data_uri = f"data:image/jpeg;base64,{test_image2_base64}"

    messages = [
        {"role": "user", "content": "Generate two images"},
        {
            "role": "assistant",
            "content": "Here are your images:",
            "images": [
                {
                    "image_url": {"url": image1_data_uri, "detail": "auto"},
                    "index": 0,
                    "type": "image_url",
                },
                {
                    "image_url": {"url": image2_data_uri, "detail": "high"},
                    "index": 1,
                    "type": "image_url",
                },
            ],
        },
    ]

    # Convert messages to Gemini format
    contents = _gemini_convert_messages_with_history(messages=messages)

    # Verify assistant message has 3 parts (1 text + 2 images)
    assert contents[1]["role"] == "model"
    assert (
        len(contents[1]["parts"]) == 3
    ), f"Expected 3 parts (text + 2 images), got {len(contents[1]['parts'])}"

    # Count inline_data parts
    inline_data_parts = [part for part in contents[1]["parts"] if "inline_data" in part]
    assert (
        len(inline_data_parts) == 2
    ), f"Expected 2 inline_data parts, got {len(inline_data_parts)}"

    # Verify first image
    assert inline_data_parts[0]["inline_data"]["mime_type"] == "image/png"
    assert inline_data_parts[0]["inline_data"]["data"] == test_image1_base64

    # Verify second image
    assert inline_data_parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    assert inline_data_parts[1]["inline_data"]["data"] == test_image2_base64


def test_assistant_message_with_images_using_message_object():
    """Test that Message objects with images field are properly converted."""
    # Create a small test image
    test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    image_data_uri = f"data:image/png;base64,{test_image_base64}"

    # Create messages using Message object (as returned by LiteLLM)
    user_message = {"role": "user", "content": "Generate an image"}

    assistant_message = Message(
        content="Here's your image!",
        role="assistant",
        tool_calls=None,
        function_call=None,
        images=[
            {
                "image_url": {"url": image_data_uri, "detail": "auto"},
                "index": 0,
                "type": "image_url",
            }
        ],
    )

    messages = [user_message, assistant_message]

    # Convert messages to Gemini format
    contents = _gemini_convert_messages_with_history(messages=messages)

    # Verify assistant message has both text and image
    assert contents[1]["role"] == "model"
    assert len(contents[1]["parts"]) == 2

    # Verify image was converted
    inline_data_parts = [part for part in contents[1]["parts"] if "inline_data" in part]
    assert len(inline_data_parts) == 1
    assert inline_data_parts[0]["inline_data"]["mime_type"] == "image/png"
    assert inline_data_parts[0]["inline_data"]["data"] == test_image_base64


def test_assistant_message_with_images_in_conversation_history():
    """
    Test multi-turn conversation where assistant message with images is in history.

    This simulates the real use case where:
    1. User asks for image generation
    2. Assistant generates image (with images field)
    3. User asks follow-up question about the image
    """
    test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    image_data_uri = f"data:image/png;base64,{test_image_base64}"

    messages = [
        {"role": "user", "content": "Generate an image of a cat"},
        {
            "role": "assistant",
            "content": "Here's a cat image:",
            "images": [
                {
                    "image_url": {"url": image_data_uri, "detail": "auto"},
                    "index": 0,
                    "type": "image_url",
                }
            ],
        },
        {"role": "user", "content": "Can you make it more colorful?"},
    ]

    # Convert messages to Gemini format
    contents = _gemini_convert_messages_with_history(messages=messages)

    # Verify structure: user -> model (with image) -> user
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    assert contents[2]["role"] == "user"

    # Verify assistant message has image in history
    inline_data_parts = [part for part in contents[1]["parts"] if "inline_data" in part]
    assert len(inline_data_parts) == 1
    assert inline_data_parts[0]["inline_data"]["mime_type"] == "image/png"


def test_function_response_has_user_role():
    """
    Test that function response ContentType blocks include role="user".

    Gemini API only accepts two roles: "user" and "model". Function responses
    must be sent with role="user". Previously, LiteLLM omitted the role field
    entirely, causing 400 errors from the Gemini API.

    Fixes: https://github.com/BerriAI/litellm/issues/22003
    Fixes: https://github.com/BerriAI/litellm/issues/20690
    """
    messages = [
        {"role": "user", "content": "What is the weather in Berlin?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Berlin"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"temperature": "15°C", "condition": "Cloudy"}',
        },
    ]

    contents = _gemini_convert_messages_with_history(messages=messages)

    # Expect: user -> model (functionCall) -> user (functionResponse)
    assert len(contents) == 3

    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    assert "function_call" in contents[1]["parts"][0]

    # The critical assertion: function response must have role="user"
    assert contents[2]["role"] == "user"
    assert "function_response" in contents[2]["parts"][0]


def test_multi_turn_function_calling_roles():
    """
    Test a full multi-turn function calling conversation produces correct roles.

    Simulates: user asks → model calls tool → tool responds → model answers → user asks again.
    Every content block must have an explicit role of "user" or "model".

    Fixes: https://github.com/BerriAI/litellm/issues/22003
    """
    messages = [
        {"role": "user", "content": "What is the weather in Berlin?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Berlin"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": '{"temperature": "15°C"}',
        },
        {
            "role": "assistant",
            "content": "The weather in Berlin is 15°C.",
        },
        {"role": "user", "content": "And in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_002",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_002",
            "content": '{"temperature": "18°C"}',
        },
    ]

    contents = _gemini_convert_messages_with_history(messages=messages)

    # Every content block must have a valid role
    for i, content in enumerate(contents):
        assert "role" in content, f"Content block {i} missing 'role' field"
        assert content["role"] in (
            "user",
            "model",
        ), f"Content block {i} has invalid role: {content.get('role')}"

    # Verify the function response blocks specifically have role="user"
    for i, content in enumerate(contents):
        for part in content["parts"]:
            if "function_response" in part:
                assert (
                    content["role"] == "user"
                ), f"Content block {i} with function_response has role='{content['role']}', expected 'user'"
