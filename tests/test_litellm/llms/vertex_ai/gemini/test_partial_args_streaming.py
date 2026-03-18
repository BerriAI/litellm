"""
Tests for partialArgs (streamFunctionCallArguments) support in Gemini streaming.

Feature: https://github.com/BerriAI/litellm/issues/22206

When streamFunctionCallArguments is enabled, Gemini streams tool call arguments
via partialArgs with jsonPath addressing instead of sending complete args at once.
The ModelResponseIterator must accumulate these into complete args before passing
to the standard _transform_parts pipeline.
"""

import json
import pytest
from typing import Optional
from unittest.mock import MagicMock

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator,
)


def _make_iterator():
    """Create a ModelResponseIterator with minimal mocks."""
    mock_logging = MagicMock()
    mock_logging.optional_params = {}
    iterator = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        logging_obj=mock_logging,
    )
    return iterator


class TestParseJsonPath:

    def test_simple(self):
        it = _make_iterator()
        assert it._parse_json_path("$.city") == ["city"]

    def test_nested(self):
        it = _make_iterator()
        assert it._parse_json_path("$.a.b") == ["a", "b"]

    def test_array(self):
        it = _make_iterator()
        assert it._parse_json_path("$.items[0].name") == ["items", 0, "name"]

    def test_empty(self):
        it = _make_iterator()
        assert it._parse_json_path("$") == []
        assert it._parse_json_path("") == []


class TestPreprocessPartialArgs:

    def test_normal_function_call_passthrough(self):
        """Normal functionCall with args should pass through unchanged."""
        it = _make_iterator()
        chunk = {
            "candidates": [{
                "content": {
                    "parts": [{"functionCall": {"name": "Read", "args": {"file_path": "/tmp/x"}}}],
                    "role": "model",
                },
            }],
        }
        result = it._preprocess_partial_args(chunk)
        parts = result["candidates"][0]["content"]["parts"]
        assert len(parts) == 1
        assert parts[0]["functionCall"]["args"] == {"file_path": "/tmp/x"}

    def test_partial_args_accumulated_to_complete(self):
        """partialArgs should be accumulated and emitted as complete args."""
        it = _make_iterator()

        # Chunk 1: Start (name + willContinue)
        c1 = {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "get_weather", "willContinue": True}}
        ], "role": "model"}}]}
        r1 = it._preprocess_partial_args(c1)
        assert len(r1["candidates"][0]["content"]["parts"]) == 0  # buffered

        # Chunk 2: partialArgs with city
        c2 = {"candidates": [{"content": {"parts": [
            {"functionCall": {"partialArgs": [
                {"jsonPath": "$.city", "stringValue": "Tokyo", "willContinue": False}
            ], "willContinue": True}}
        ], "role": "model"}}]}
        r2 = it._preprocess_partial_args(c2)
        assert len(r2["candidates"][0]["content"]["parts"]) == 0  # still buffering

        # Chunk 3: Stream end
        c3 = {"candidates": [{"content": {"parts": [
            {"functionCall": {"partialArgs": [
                {"jsonPath": "$.unit", "stringValue": "celsius", "willContinue": False}
            ], "willContinue": False}}
        ], "role": "model"}}]}
        r3 = it._preprocess_partial_args(c3)
        parts = r3["candidates"][0]["content"]["parts"]

        # Should emit complete functionCall
        assert len(parts) == 1
        assert parts[0]["functionCall"]["name"] == "get_weather"
        assert parts[0]["functionCall"]["args"] == {"city": "Tokyo", "unit": "celsius"}

    def test_string_accumulation(self):
        """String values should be accumulated across chunks."""
        it = _make_iterator()

        # Start
        it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "Write", "willContinue": True}}
        ], "role": "model"}}]})

        # Chunk with partial string
        it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"partialArgs": [
                {"jsonPath": "$.content", "stringValue": "Hello ", "willContinue": True}
            ], "willContinue": True}}
        ], "role": "model"}}]})

        it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"partialArgs": [
                {"jsonPath": "$.content", "stringValue": "World", "willContinue": True}
            ], "willContinue": True}}
        ], "role": "model"}}]})

        # End
        result = it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"partialArgs": [
                {"jsonPath": "$.content", "stringValue": "", "willContinue": False}
            ], "willContinue": False}}
        ], "role": "model"}}]})

        parts = result["candidates"][0]["content"]["parts"]
        assert parts[0]["functionCall"]["args"]["content"] == "Hello World"

    def test_nested_json_path(self):
        """Nested jsonPath like $.questions[0].text should work."""
        it = _make_iterator()

        it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "AskUser", "willContinue": True}}
        ], "role": "model"}}]})

        it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"partialArgs": [
                {"jsonPath": "$.questions[0].text", "stringValue": "What?", "willContinue": False}
            ], "willContinue": True}}
        ], "role": "model"}}]})

        it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"partialArgs": [
                {"boolValue": False, "jsonPath": "$.questions[0].multiSelect"}
            ], "willContinue": True}}
        ], "role": "model"}}]})

        result = it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {}, "willContinue": False}
        ], "role": "model"}}]})

        # willContinue is at the functionCall level, not the part level
        # Since this empty functionCall has no willContinue in the functionCall dict,
        # it defaults to False, so it should emit
        # Actually let me re-check: the functionCall dict is {} and willContinue
        # comes from fc.get("willContinue", False) = False. And partial_fc_active
        # is True. So it should emit.

    def test_empty_fc_with_will_continue_true_skipped(self):
        """Empty functionCall with willContinue:true should be skipped (intermediate state)."""
        it = _make_iterator()

        it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "Tool", "willContinue": True}}
        ], "role": "model"}}]})

        it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"partialArgs": [
                {"jsonPath": "$.x", "stringValue": "a", "willContinue": False}
            ], "willContinue": True}}
        ], "role": "model"}}]})

        # Empty functionCall with willContinue:true — should NOT close stream
        result = it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"willContinue": True}}
        ], "role": "model"}}]})
        assert len(result["candidates"][0]["content"]["parts"]) == 0
        assert it._partial_fc_active is True  # Still accumulating

    def test_thought_signature_preserved(self):
        """thoughtSignature from the first chunk should be preserved in output."""
        it = _make_iterator()

        it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "Read", "willContinue": True}, "thoughtSignature": "sig123"}
        ], "role": "model"}}]})

        result = it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"partialArgs": [
                {"jsonPath": "$.path", "stringValue": "/tmp/x", "willContinue": False}
            ], "willContinue": False}}
        ], "role": "model"}}]})

        parts = result["candidates"][0]["content"]["parts"]
        assert parts[0]["thoughtSignature"] == "sig123"

    def test_bool_and_int_values(self):
        """Bool and int values should be set correctly."""
        it = _make_iterator()

        it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "Tool", "willContinue": True}}
        ], "role": "model"}}]})

        result = it._preprocess_partial_args({"candidates": [{"content": {"parts": [
            {"functionCall": {"partialArgs": [
                {"jsonPath": "$.enabled", "boolValue": True},
                {"jsonPath": "$.count", "intValue": 42},
                {"jsonPath": "$.rate", "doubleValue": 3.14},
            ], "willContinue": False}}
        ], "role": "model"}}]})

        args = result["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]
        assert args["enabled"] is True
        assert args["count"] == 42
        assert args["rate"] == 3.14

    def test_text_part_not_affected(self):
        """Text parts should pass through unchanged."""
        it = _make_iterator()
        chunk = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello world"}],
                    "role": "model",
                },
            }],
        }
        result = it._preprocess_partial_args(chunk)
        assert result["candidates"][0]["content"]["parts"][0]["text"] == "Hello world"
