"""
Test A2A Guardrail Translation Handler

Unit tests for the A2A protocol guardrail handler, covering:
- Text extraction from A2A message parts (input and output formats)
- In-place modification logic for streaming responses
- Defensive handling of malformed or empty inputs
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.llms.a2a.chat.guardrail_translation.handler import A2AGuardrailHandler
from litellm.types.utils import CallTypes


@pytest.fixture
def mock_guardrail():
    """Guardrail mock that echoes input texts."""
    guardrail = MagicMock()
    guardrail.apply_guardrail = AsyncMock(
        side_effect=lambda inputs, **kwargs: {"texts": inputs.get("texts", [])}
    )
    return guardrail


class TestA2AGuardrailHandlerProcessInputMessages:
    """Tests for process_input_messages (pre-call hook)."""

    @pytest.mark.asyncio
    async def test_extracts_and_applies_guardrail_to_text_parts(self, mock_guardrail):
        """Should extract text from kind=text parts and apply guardrail."""
        handler = A2AGuardrailHandler()
        mock_guardrail.apply_guardrail = AsyncMock(
            return_value={"texts": ["guardrailed hello", "guardrailed world"]}
        )

        data = {
            "params": {
                "message": {
                    "parts": [
                        {"kind": "text", "text": "hello"},
                        {"kind": "text", "text": "world"},
                    ]
                }
            }
        }

        result = await handler.process_input_messages(
            data=data,
            guardrail_to_apply=mock_guardrail,
        )

        mock_guardrail.apply_guardrail.assert_called_once()
        call_inputs = mock_guardrail.apply_guardrail.call_args.kwargs["inputs"]
        assert call_inputs["texts"] == ["hello", "world"]

        assert result["params"]["message"]["parts"][0]["text"] == "guardrailed hello"
        assert result["params"]["message"]["parts"][1]["text"] == "guardrailed world"

    @pytest.mark.asyncio
    async def test_skips_empty_parts(self, mock_guardrail):
        """Should skip parts with no text content."""
        handler = A2AGuardrailHandler()

        data = {
            "params": {
                "message": {
                    "parts": [
                        {"kind": "text", "text": ""},
                        {"kind": "model", "model": "gpt-4"},
                    ]
                }
            }
        }

        result = await handler.process_input_messages(
            data=data,
            guardrail_to_apply=mock_guardrail,
        )

        mock_guardrail.apply_guardrail.assert_not_called()
        assert result == data

    @pytest.mark.asyncio
    async def test_returns_unchanged_when_no_parts(self, mock_guardrail):
        """Should return data unchanged when message has no parts."""
        handler = A2AGuardrailHandler()
        data = {"params": {"message": {}}}

        result = await handler.process_input_messages(
            data=data,
            guardrail_to_apply=mock_guardrail,
        )

        mock_guardrail.apply_guardrail.assert_not_called()
        assert result == data


class TestA2AGuardrailHandlerProcessOutputResponse:
    """Tests for process_output_response (post-call, non-streaming)."""

    @pytest.mark.asyncio
    async def test_applies_guardrail_to_direct_message_parts(self, mock_guardrail):
        """Should process result.parts format."""
        mock_guardrail.apply_guardrail = AsyncMock(
            return_value={"texts": ["guardrailed output"]}
        )
        handler = A2AGuardrailHandler()

        response = {
            "result": {
                "kind": "message",
                "parts": [{"kind": "text", "text": "original output"}],
            }
        }

        result = await handler.process_output_response(
            response=response,
            guardrail_to_apply=mock_guardrail,
        )

        assert result["result"]["parts"][0]["text"] == "guardrailed output"

    @pytest.mark.asyncio
    async def test_applies_guardrail_to_nested_message_parts(self, mock_guardrail):
        """Should process result.message.parts format."""
        mock_guardrail.apply_guardrail = AsyncMock(
            return_value={"texts": ["guardrailed nested"]}
        )
        handler = A2AGuardrailHandler()

        response = {
            "result": {
                "message": {
                    "parts": [{"kind": "text", "text": "nested text"}],
                }
            }
        }

        result = await handler.process_output_response(
            response=response,
            guardrail_to_apply=mock_guardrail,
        )

        assert result["result"]["message"]["parts"][0]["text"] == "guardrailed nested"

    @pytest.mark.asyncio
    async def test_applies_guardrail_to_artifact_parts(self, mock_guardrail):
        """Should process result.artifact.parts (streaming artifact-update format)."""
        mock_guardrail.apply_guardrail = AsyncMock(
            return_value={"texts": ["guardrailed artifact"]}
        )
        handler = A2AGuardrailHandler()

        response = {
            "result": {
                "kind": "artifact-update",
                "artifact": {
                    "parts": [{"kind": "text", "text": "artifact text"}],
                },
            }
        }

        result = await handler.process_output_response(
            response=response,
            guardrail_to_apply=mock_guardrail,
        )

        assert result["result"]["artifact"]["parts"][0]["text"] == "guardrailed artifact"


class TestA2AGuardrailHandlerProcessOutputStreamingResponse:
    """
    Tests for process_output_streaming_response.

    IMPORTANT: This method modifies responses_so_far IN-PLACE. It:
    1. Concatenates all text from chunks in order
    2. Applies guardrail once to the combined text
    3. Writes the full guardrailed text into the FIRST chunk that had text
    4. CLEARS all other text parts in subsequent chunks to "" (in-place)
    """

    @pytest.mark.asyncio
    async def test_streaming_combines_text_and_puts_in_first_chunk(self, mock_guardrail):
        """Combined guardrailed text should be placed in first chunk; others cleared."""
        mock_guardrail.apply_guardrail = AsyncMock(
            return_value={"texts": ["COMBINED_GUARDRAILED"]}
        )
        handler = A2AGuardrailHandler()

        chunk1 = {
            "result": {
                "artifact": {"parts": [{"kind": "text", "text": "chunk1 "}]},
            }
        }
        chunk2 = {
            "result": {
                "artifact": {"parts": [{"kind": "text", "text": "chunk2"}]},
            }
        }
        responses_so_far = [chunk1, chunk2]

        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=mock_guardrail,
        )

        # In-place: first chunk gets full guardrailed text
        assert chunk1["result"]["artifact"]["parts"][0]["text"] == "COMBINED_GUARDRAILED"
        # Second chunk's text is cleared
        assert chunk2["result"]["artifact"]["parts"][0]["text"] == ""

        assert result is responses_so_far  # Same list, modified in place

    @pytest.mark.asyncio
    async def test_streaming_handles_ndjson_strings(self, mock_guardrail):
        """Should parse NDJSON strings and write back as NDJSON."""
        mock_guardrail.apply_guardrail = AsyncMock(
            return_value={"texts": ["GUARDRAILED"]}
        )
        handler = A2AGuardrailHandler()

        responses_so_far = [
            '{"result":{"artifact":{"parts":[{"kind":"text","text":"hello"}]}}}\n',
        ]

        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=mock_guardrail,
        )

        # responses_so_far is modified in place; NDJSON string is updated
        parsed = __import__("json").loads(responses_so_far[0].strip())
        assert parsed["result"]["artifact"]["parts"][0]["text"] == "GUARDRAILED"

    @pytest.mark.asyncio
    async def test_streaming_returns_early_when_no_text(self, mock_guardrail):
        """Should return responses_so_far unchanged when no text content."""
        handler = A2AGuardrailHandler()

        responses_so_far = [
            {"result": {"artifact": {"parts": [{"kind": "model", "model": "gpt-4"}]}}},
        ]

        result = await handler.process_output_streaming_response(
            responses_so_far=responses_so_far,
            guardrail_to_apply=mock_guardrail,
        )

        mock_guardrail.apply_guardrail.assert_not_called()
        assert result == responses_so_far


class TestA2AGuardrailHandlerExtractTextsFromResult:
    """Tests for _extract_texts_from_result helper."""

    def test_extracts_from_multiple_formats(self):
        """Should extract text from parts, message.parts, artifact.parts, etc."""
        handler = A2AGuardrailHandler()
        texts: list = []
        mappings: list = []

        result = {
            "parts": [{"kind": "text", "text": "direct"}],
            "message": {"parts": [{"kind": "text", "text": "nested"}]},
            "artifact": {"parts": [{"kind": "text", "text": "artifact"}]},
        }

        handler._extract_texts_from_result(result, texts, mappings)

        assert texts == ["direct", "nested", "artifact"]
        assert len(mappings) == 3


class TestA2AGuardrailHandlerApplyTextToPath:
    """Tests for _apply_text_to_path helper."""

    def test_applies_text_to_nested_path(self):
        """Should navigate path and update part text."""
        handler = A2AGuardrailHandler()
        result = {
            "message": {
                "parts": [
                    {"kind": "text", "text": "old"},
                ]
            }
        }

        handler._apply_text_to_path(
            result=result,
            path=("message", "parts"),
            part_idx=0,
            text="new",
        )

        assert result["message"]["parts"][0]["text"] == "new"


def test_a2a_guardrail_translation_mappings():
    """A2A handler should be registered for send_message and asend_message."""
    from litellm.llms.a2a.chat.guardrail_translation import (
        guardrail_translation_mappings,
    )

    assert CallTypes.send_message in guardrail_translation_mappings
    assert CallTypes.asend_message in guardrail_translation_mappings
    assert (
        guardrail_translation_mappings[CallTypes.send_message] == A2AGuardrailHandler
    )
    assert (
        guardrail_translation_mappings[CallTypes.asend_message] == A2AGuardrailHandler
    )
