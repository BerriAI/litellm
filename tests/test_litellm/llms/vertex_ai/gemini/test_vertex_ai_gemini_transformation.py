from litellm.llms.vertex_ai.gemini.transformation import (
    check_if_part_exists_in_parts,
    _transform_request_body,
    _gemini_convert_messages_with_history,
)


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



def test_metadata_to_labels_vertex_only():
    """Test that metadata->labels conversion only happens for Vertex AI"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {}
    litellm_params = {
        "metadata": {
            "requester_metadata": {
                "user": "john_doe",
                "project": "test-project"
            }
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
    messages = [
        {
            "content": "",
            "role": "user"
        }
    ]
    
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
            functionCall={"name": "get_current_temperature", "args": {"location": "Paris"}},
            thoughtSignature=test_signature,  # First FC has signature
        ),
        HttpxPartType(
            functionCall={"name": "get_current_temperature", "args": {"location": "London"}},
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
    assert "provider_specific_fields" not in tools[1] or "thought_signature" not in tools[1].get("provider_specific_fields", {})


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
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_invoke,
    )
    import base64

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
    expected_dummy = base64.b64encode(b"skip_thought_signature_validator").decode("utf-8")
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
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_invoke,
    )
    import base64

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
    expected_dummy = base64.b64encode(b"skip_thought_signature_validator").decode("utf-8")
    assert gemini_parts[0]["thoughtSignature"] == expected_dummy
