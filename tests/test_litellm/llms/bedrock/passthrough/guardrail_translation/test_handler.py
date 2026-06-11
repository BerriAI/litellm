"""
Tests for BedrockPassthroughGuardrailHandler.

Validates that:
- Text content is extracted from Converse messages and system blocks
- apply_guardrail receives the correct texts
- Modified texts are written back in-place; non-text fields are untouched
- Blocking raises and propagates
- Non-converse endpoints skip guardrail execution
"""

import copy
import pytest
from unittest.mock import AsyncMock, MagicMock

from litellm.llms.bedrock.passthrough.guardrail_translation.handler import (
    BedrockPassthroughGuardrailHandler,
    _extract_converse_texts,
    _is_converse_endpoint,
    _write_back_texts,
)
from fastapi import HTTPException


def _make_guardrail(apply_result: dict) -> MagicMock:
    g = MagicMock()
    g.guardrail_name = "test-guard"
    g.apply_guardrail = AsyncMock(return_value=apply_result)
    g.skip_system_message_in_guardrail = False
    g.skip_tool_message_in_guardrail = False
    return g


def _converse_data(endpoint: str = "model/anthropic.claude-3-sonnet/converse") -> dict:
    return {
        "endpoint": endpoint,
        "custom_llm_provider": "bedrock",
        "model": "anthropic.claude-3-sonnet",
        "data": {
            "system": [{"text": "You are helpful."}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "Hello world"},
                        {"toolUse": {"toolUseId": "t1", "name": "search", "input": {}}},
                    ],
                }
            ],
            "inferenceConfig": {"maxTokens": 100},
        },
    }


class TestIsConverseEndpoint:
    def test_converse(self):
        assert _is_converse_endpoint("model/foo/converse") is True

    def test_converse_stream(self):
        assert _is_converse_endpoint("model/foo/converse-stream") is True

    def test_invoke(self):
        assert _is_converse_endpoint("model/foo/invoke") is False

    def test_invoke_with_response_stream(self):
        assert _is_converse_endpoint("model/foo/invoke-with-response-stream") is False


class TestExtractConverseTexts:
    def test_extracts_system_and_message_text(self):
        body = {
            "system": [{"text": "sys text"}],
            "messages": [{"role": "user", "content": [{"text": "user text"}]}],
        }
        texts, mappings = _extract_converse_texts(
            body, skip_system=False, skip_tool=False
        )
        assert texts == ["sys text", "user text"]
        assert mappings[0] == ("system", 0, -1)
        assert mappings[1] == ("message", 0, 0)

    def test_skip_system(self):
        body = {
            "system": [{"text": "sys text"}],
            "messages": [{"role": "user", "content": [{"text": "user text"}]}],
        }
        texts, mappings = _extract_converse_texts(
            body, skip_system=True, skip_tool=False
        )
        assert texts == ["user text"]
        assert all(m[0] == "message" for m in mappings)

    def test_skip_tool_blocks(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "hello"},
                        {"toolUse": {"toolUseId": "1", "name": "fn", "input": {}}},
                        {
                            "toolResult": {
                                "toolUseId": "1",
                                "content": [{"text": "result"}],
                            }
                        },
                    ],
                }
            ]
        }
        texts, _ = _extract_converse_texts(body, skip_system=False, skip_tool=True)
        assert texts == ["hello"]

    def test_non_text_content_blocks_ignored(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"image": {"format": "png", "source": {}}}],
                }
            ]
        }
        texts, _ = _extract_converse_texts(body, skip_system=False, skip_tool=False)
        assert texts == []


class TestWriteBackTexts:
    def test_writes_system_text(self):
        body = {"system": [{"text": "original"}], "messages": []}
        _write_back_texts(body, ["replaced"], [("system", 0, -1)])
        assert body["system"][0]["text"] == "replaced"

    def test_writes_message_text(self):
        body = {"messages": [{"role": "user", "content": [{"text": "original"}]}]}
        _write_back_texts(body, ["replaced"], [("message", 0, 0)])
        assert body["messages"][0]["content"][0]["text"] == "replaced"

    def test_extra_non_text_fields_untouched(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "hello"},
                        {
                            "toolUse": {
                                "toolUseId": "1",
                                "name": "fn",
                                "input": {"key": "val"},
                            }
                        },
                    ],
                }
            ],
            "inferenceConfig": {"maxTokens": 100},
        }
        original = copy.deepcopy(body)
        _write_back_texts(body, ["replaced"], [("message", 0, 0)])
        assert body["messages"][0]["content"][0]["text"] == "replaced"
        assert (
            body["messages"][0]["content"][1] == original["messages"][0]["content"][1]
        )
        assert body["inferenceConfig"] == original["inferenceConfig"]


class TestBedrockPassthroughGuardrailHandlerInput:
    @pytest.mark.asyncio
    async def test_texts_extracted_and_apply_guardrail_called(self):
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data()
        guardrail = _make_guardrail({"texts": ["You are helpful.", "Hello world"]})

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        call_args = guardrail.apply_guardrail.call_args
        assert call_args.kwargs["input_type"] == "request"
        sent_texts = call_args.kwargs["inputs"]["texts"]
        assert "You are helpful." in sent_texts
        assert "Hello world" in sent_texts
        # toolUse block not included
        assert len(sent_texts) == 2

    @pytest.mark.asyncio
    async def test_masking_writes_back_in_place(self):
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data()
        guardrail = _make_guardrail({"texts": ["[REDACTED]", "[REDACTED]"]})

        result = await handler.process_input_messages(
            data=data, guardrail_to_apply=guardrail
        )

        body = result["data"]
        assert body["system"][0]["text"] == "[REDACTED]"
        assert body["messages"][0]["content"][0]["text"] == "[REDACTED]"
        # toolUse unchanged
        assert body["messages"][0]["content"][1].get("toolUse") is not None
        # inferenceConfig untouched
        assert body["inferenceConfig"] == {"maxTokens": 100}

    @pytest.mark.asyncio
    async def test_blocking_guardrail_propagates_exception(self):
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data()
        guardrail = MagicMock()
        guardrail.guardrail_name = "block-guard"
        guardrail.skip_system_message_in_guardrail = False
        guardrail.skip_tool_message_in_guardrail = False
        guardrail.apply_guardrail = AsyncMock(
            side_effect=HTTPException(status_code=400, detail="Blocked")
        )

        with pytest.raises(HTTPException):
            await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

    @pytest.mark.asyncio
    async def test_non_converse_endpoint_skips_apply_guardrail(self):
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data(endpoint="model/anthropic.claude-3-sonnet/invoke")
        guardrail = _make_guardrail({"texts": []})

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        guardrail.apply_guardrail.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_messages_field_skips(self):
        handler = BedrockPassthroughGuardrailHandler()
        data = {
            "endpoint": "model/foo/converse",
            "custom_llm_provider": "bedrock",
            "data": {"system": [{"text": "sys"}]},
        }
        guardrail = _make_guardrail({"texts": []})

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        guardrail.apply_guardrail.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_passed_to_apply_guardrail(self):
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data()
        guardrail = _make_guardrail({"texts": ["You are helpful.", "Hello world"]})

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        call_args = guardrail.apply_guardrail.call_args
        assert call_args.kwargs["inputs"].get("model") == "anthropic.claude-3-sonnet"


class TestBedrockPassthroughGuardrailHandlerOutput:
    def _converse_response(self, text: str = "Model reply") -> dict:
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": text}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

    @pytest.mark.asyncio
    async def test_response_text_extracted_and_apply_guardrail_called(self):
        handler = BedrockPassthroughGuardrailHandler()
        response = self._converse_response("Model reply")
        guardrail = _make_guardrail({"texts": ["Model reply"]})

        await handler.process_output_response(
            response=response, guardrail_to_apply=guardrail
        )

        call_args = guardrail.apply_guardrail.call_args
        assert call_args.kwargs["input_type"] == "response"
        assert call_args.kwargs["inputs"]["texts"] == ["Model reply"]

    @pytest.mark.asyncio
    async def test_response_masking_writes_back(self):
        handler = BedrockPassthroughGuardrailHandler()
        response = self._converse_response("Bad content")
        guardrail = _make_guardrail({"texts": ["[MASKED]"]})

        result = await handler.process_output_response(
            response=response, guardrail_to_apply=guardrail
        )

        assert result["output"]["message"]["content"][0]["text"] == "[MASKED]"
        assert result["stopReason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_non_dict_response_returned_unchanged(self):
        handler = BedrockPassthroughGuardrailHandler()
        guardrail = _make_guardrail({"texts": []})

        result = await handler.process_output_response(
            response="raw string", guardrail_to_apply=guardrail
        )

        assert result == "raw string"
        guardrail.apply_guardrail.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_output_structure_skips(self):
        handler = BedrockPassthroughGuardrailHandler()
        response = {"stopReason": "end_turn"}
        guardrail = _make_guardrail({"texts": []})

        result = await handler.process_output_response(
            response=response, guardrail_to_apply=guardrail
        )

        assert result == {"stopReason": "end_turn"}
        guardrail.apply_guardrail.assert_not_called()
