import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    Delta,
    Function,
    ModelResponseStream,
    PromptTokensDetails,
    StreamingChoices,
    Usage,
)


def test_get_combined_tool_content():
    chunks = [
        ModelResponseStream(
            id="chatcmpl-8478099a-3724-42c7-9194-88d97ffd254b",
            created=1744771912,
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        provider_specific_fields=None,
                        content=None,
                        role="assistant",
                        function_call=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_m87w",
                                function=Function(
                                    arguments='{"location": "San Francisco", "unit": "imperial"}',
                                    name="get_current_weather",
                                ),
                                type="function",
                                index=0,
                            )
                        ],
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            stream_options=None,
        ),
        ModelResponseStream(
            id="chatcmpl-8478099a-3724-42c7-9194-88d97ffd254b",
            created=1744771912,
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        provider_specific_fields=None,
                        content=None,
                        role="assistant",
                        function_call=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_rrns",
                                function=Function(
                                    arguments='{"location": "Tokyo", "unit": "metric"}',
                                    name="get_current_weather",
                                ),
                                type="function",
                                index=1,
                            )
                        ],
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            stream_options=None,
        ),
        ModelResponseStream(
            id="chatcmpl-8478099a-3724-42c7-9194-88d97ffd254b",
            created=1744771912,
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        provider_specific_fields=None,
                        content=None,
                        role="assistant",
                        function_call=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_0k29",
                                function=Function(
                                    arguments='{"location": "Paris", "unit": "metric"}',
                                    name="get_current_weather",
                                ),
                                type="function",
                                index=2,
                            )
                        ],
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            stream_options=None,
        ),
    ]
    chunk_processor = ChunkProcessor(chunks=chunks)

    tool_calls_list = chunk_processor.get_combined_tool_content(chunks)
    assert tool_calls_list == [
        ChatCompletionMessageToolCall(
            id="call_m87w",
            function=Function(
                arguments='{"location": "San Francisco", "unit": "imperial"}',
                name="get_current_weather",
            ),
            type="function",
        ),
        ChatCompletionMessageToolCall(
            id="call_rrns",
            function=Function(
                arguments='{"location": "Tokyo", "unit": "metric"}',
                name="get_current_weather",
            ),
            type="function",
        ),
        ChatCompletionMessageToolCall(
            id="call_0k29",
            function=Function(
                arguments='{"location": "Paris", "unit": "metric"}',
                name="get_current_weather",
            ),
            type="function",
        ),
    ]


def test_cache_read_input_tokens_retained():
    chunk1 = ModelResponseStream(
        id="chatcmpl-95aabb85-c39f-443d-ae96-0370c404d70c",
        created=1745513206,
        model="claude-3-7-sonnet-20250219",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="",
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=5,
            prompt_tokens=11779,
            total_tokens=11784,
            completion_tokens_details=None,
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=None, cached_tokens=11775
            ),
            cache_creation_input_tokens=4,
            cache_read_input_tokens=11775,
        ),
    )

    chunk2 = ModelResponseStream(
        id="chatcmpl-95aabb85-c39f-443d-ae96-0370c404d70c",
        created=1745513207,
        model="claude-3-7-sonnet-20250219",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content=None,
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=214,
            prompt_tokens=0,
            total_tokens=214,
            completion_tokens_details=None,
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=None, cached_tokens=0
            ),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )

    # Use dictionaries directly instead of ModelResponseStream
    chunks = [chunk1, chunk2]
    processor = ChunkProcessor(chunks=chunks)

    usage = processor.calculate_usage(
        chunks=chunks,
        model="claude-3-7-sonnet",
        completion_output="",
    )

    assert usage.cache_creation_input_tokens == 4
    assert usage.cache_read_input_tokens == 11775
    assert usage.prompt_tokens_details.cached_tokens == 11775


def test_stream_chunk_builder_litellm_usage_chunks():
    """
    Validate ChunkProcessor.calculate_usage uses provided usage fields from streaming chunks
    and reconstructs prompt and completion tokens without making any upstream API calls.
    """
    # Prepare two mocked streaming chunks with usage split across them
    chunk1 = ModelResponseStream(
        id="chatcmpl-mocked-usage-1",
        created=1745513206,
        model="gemini/gemini-2.5-flash-lite",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="",
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=0,
            prompt_tokens=50,
            total_tokens=50,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        ),
    )

    chunk2 = ModelResponseStream(
        id="chatcmpl-mocked-usage-1",
        created=1745513207,
        model="gemini/gemini-2.5-flash-lite",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content=None,
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=27,
            prompt_tokens=0,
            total_tokens=27,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        ),
    )

    chunks = [chunk1, chunk2]
    processor = ChunkProcessor(chunks=chunks)

    usage = processor.calculate_usage(
        chunks=chunks, model="gemini/gemini-2.5-flash-lite", completion_output=""
    )

    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 27
    assert usage.total_tokens == 77


# =============================================================================
# Tests for _validate_and_repair_tool_arguments()
# =============================================================================

from litellm.litellm_core_utils.streaming_chunk_builder_utils import (
    _validate_and_repair_tool_arguments,
)


class TestValidateAndRepairToolArguments:
    """Tests for _validate_and_repair_tool_arguments() function.

    This function uses JSONDecoder.raw_decode() to handle malformed JSON
    from streaming providers like Gemini that may send duplicate/overlapping chunks.
    """

    # --- Valid JSON (should pass through unchanged) ---

    def test_valid_simple_json(self):
        """Valid JSON should pass through unchanged."""
        valid = '{"address": "123 Main St"}'
        assert _validate_and_repair_tool_arguments(valid) == valid

    def test_valid_nested_json(self):
        """Deeply nested JSON should work correctly."""
        valid = '{"outer": {"inner": {"deep": {"value": 123}}}}'
        assert _validate_and_repair_tool_arguments(valid) == valid

    def test_valid_json_with_arrays(self):
        """JSON with arrays should work correctly."""
        valid = '{"data": [1, 2, {"nested": [3, 4]}]}'
        assert _validate_and_repair_tool_arguments(valid) == valid

    def test_valid_json_with_escaped_quotes(self):
        """JSON with escaped quotes in strings should work."""
        valid = '{"text": "He said \\"hello\\""}'
        assert _validate_and_repair_tool_arguments(valid) == valid

    def test_valid_json_with_unicode(self):
        """JSON with unicode characters should work."""
        valid = '{"city": "東京", "greeting": "Привет"}'
        assert _validate_and_repair_tool_arguments(valid) == valid

    # --- Malformed JSON (duplicate/concatenated chunks) ---

    def test_duplicate_simple_json(self):
        """Duplicate JSON objects should extract the first one."""
        malformed = '{"address": "School"}{"address": "School"}'
        expected = '{"address": "School"}'
        assert _validate_and_repair_tool_arguments(malformed) == expected

    def test_duplicate_nested_json(self):
        """Duplicate nested JSON should extract the first one."""
        malformed = '{"a": {"b": 1}}{"a": {"b": 1}}'
        expected = '{"a": {"b": 1}}'
        assert _validate_and_repair_tool_arguments(malformed) == expected

    def test_different_concatenated_json(self):
        """Different JSON objects concatenated should return first."""
        malformed = '{"first": true}{"second": false}'
        expected = '{"first": true}'
        assert _validate_and_repair_tool_arguments(malformed) == expected

    def test_json_with_extra_garbage(self):
        """JSON followed by garbage data should extract valid JSON."""
        malformed = '{"valid": true}garbage data here'
        expected = '{"valid": true}'
        assert _validate_and_repair_tool_arguments(malformed) == expected

    # --- Edge cases ---

    def test_empty_string(self):
        """Empty string should return empty object."""
        assert _validate_and_repair_tool_arguments("") == "{}"

    def test_valid_empty_object(self):
        """Empty JSON object should work."""
        assert _validate_and_repair_tool_arguments("{}") == "{}"

    def test_completely_invalid_json(self):
        """Completely invalid JSON should return as-is with warning."""
        invalid = "not json at all"
        result = _validate_and_repair_tool_arguments(invalid)
        assert result == invalid  # Returns as-is

    # --- Long/complex JSON ---

    def test_long_json_object(self):
        """Long JSON with many keys should work correctly."""
        keys = {f"key_{i}": f"value_{i}" for i in range(100)}
        valid = json.dumps(keys)
        assert _validate_and_repair_tool_arguments(valid) == valid

    def test_deeply_nested_10_levels(self):
        """10 levels of nesting should work correctly."""
        nested = '{"l1": {"l2": {"l3": {"l4": {"l5": {"l6": {"l7": {"l8": {"l9": {"l10": "deep"}}}}}}}}}}'
        assert _validate_and_repair_tool_arguments(nested) == nested

    def test_large_array_in_json(self):
        """JSON with large array should work correctly."""
        large = json.dumps({"numbers": list(range(1000))})
        assert _validate_and_repair_tool_arguments(large) == large

    # --- Gemini-specific patterns ---

    def test_gemini_geocode_duplicate_pattern(self):
        """Simulate the actual Gemini geocoding duplicate pattern."""
        # This is the actual pattern causing the bug
        malformed = '{"address": "Cotham Brow School BS6 6DT", "country_code": "GB"}{"address": "Cotham Brow School BS6 6DT", "country_code": "GB"}'
        expected = '{"address": "Cotham Brow School BS6 6DT", "country_code": "GB"}'
        assert _validate_and_repair_tool_arguments(malformed) == expected

    def test_json_with_braces_in_strings(self):
        """Braces inside string values should not confuse parser."""
        valid = '{"regex": "^\\\\{[a-z]+\\\\}$", "note": "contains { and }"}'
        assert _validate_and_repair_tool_arguments(valid) == valid

    def test_json_with_newlines_in_strings(self):
        """JSON with newline characters in strings should work."""
        valid = '{"text": "line1\\nline2\\nline3"}'
        assert _validate_and_repair_tool_arguments(valid) == valid

    def test_triple_concatenation(self):
        """Three concatenated JSON objects should return first."""
        malformed = '{"a": 1}{"b": 2}{"c": 3}'
        expected = '{"a": 1}'
        assert _validate_and_repair_tool_arguments(malformed) == expected
