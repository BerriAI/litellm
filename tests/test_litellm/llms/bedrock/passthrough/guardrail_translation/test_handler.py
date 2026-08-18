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


class GuardrailBlocked(Exception):
    """Stand-in for a guardrail rejecting a request; the handler must let it propagate."""


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
        texts, holders = _extract_converse_texts(body, skip_system=False, skip_tool=False)
        assert texts == ["sys text", "user text"]
        assert holders[0] == (body["system"][0], "text")
        assert holders[1] == (body["messages"][0]["content"][0], "text")

    def test_skip_system(self):
        body = {
            "system": [{"text": "sys text"}],
            "messages": [{"role": "user", "content": [{"text": "user text"}]}],
        }
        texts, holders = _extract_converse_texts(body, skip_system=True, skip_tool=False)
        assert texts == ["user text"]
        assert holders == [(body["messages"][0]["content"][0], "text")]

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

    def test_extracts_nested_tool_result_text_and_json(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "hello"},
                        {
                            "toolResult": {
                                "toolUseId": "1",
                                "content": [
                                    {"text": "blocked tool text"},
                                    {"json": {"k": "blocked json value"}},
                                ],
                            }
                        },
                    ],
                }
            ]
        }
        texts, holders = _extract_converse_texts(body, skip_system=False, skip_tool=False)
        assert texts == ["hello", "blocked tool text", "blocked json value"]
        tool_content = body["messages"][0]["content"][1]["toolResult"]["content"]
        assert holders[1] == (tool_content[0], "text")
        assert holders[2] == (tool_content[1]["json"], "k")

    def test_extracts_tool_use_input_strings(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "1",
                                "name": "lookup",
                                "input": {"query": "blocked input value", "limit": 5},
                            }
                        }
                    ],
                }
            ]
        }
        texts, holders = _extract_converse_texts(body, skip_system=False, skip_tool=False)
        assert texts == ["blocked input value"]
        tool_use_input = body["messages"][0]["content"][0]["toolUse"]["input"]
        assert holders[0] == (tool_use_input, "query")

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

    def test_extracts_tool_config_description_and_schema(self):
        body = {
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "lookup",
                            "description": "blocked tool description",
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": {
                                        "q": {
                                            "type": "string",
                                            "description": "blocked schema description",
                                        }
                                    },
                                }
                            },
                        }
                    }
                ]
            },
        }
        texts, _ = _extract_converse_texts(body, skip_system=False, skip_tool=False)
        assert "blocked tool description" in texts
        assert "blocked schema description" in texts

    def test_tool_config_scanned_even_when_tool_messages_skipped(self):
        body = {
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "toolConfig": {
                "tools": [
                    {"toolSpec": {"name": "fn", "description": "blocked description"}}
                ]
            },
        }
        texts, _ = _extract_converse_texts(body, skip_system=False, skip_tool=True)
        assert "blocked description" in texts

    def test_extracts_additional_model_request_fields(self):
        body = {
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "additionalModelRequestFields": {
                "reasoning_config": {"prompt": "blocked extra field"}
            },
        }
        texts, _ = _extract_converse_texts(body, skip_system=False, skip_tool=False)
        assert "blocked extra field" in texts


class TestWriteBackTexts:
    def test_writes_system_text(self):
        body = {"system": [{"text": "original"}], "messages": []}
        _, holders = _extract_converse_texts(body, skip_system=False, skip_tool=False)
        _write_back_texts(["replaced"], holders)
        assert body["system"][0]["text"] == "replaced"

    def test_writes_message_text(self):
        body = {"messages": [{"role": "user", "content": [{"text": "original"}]}]}
        _, holders = _extract_converse_texts(body, skip_system=False, skip_tool=False)
        _write_back_texts(["replaced"], holders)
        assert body["messages"][0]["content"][0]["text"] == "replaced"

    def test_writes_nested_tool_result_text(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": "1",
                                "content": [{"text": "original"}],
                            }
                        }
                    ],
                }
            ]
        }
        _, holders = _extract_converse_texts(body, skip_system=False, skip_tool=False)
        _write_back_texts(["masked"], holders)
        assert body["messages"][0]["content"][0]["toolResult"]["content"][0]["text"] == "masked"

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
        _, holders = _extract_converse_texts(body, skip_system=False, skip_tool=False)
        _write_back_texts(["replaced"], holders)
        assert body["messages"][0]["content"][0]["text"] == "replaced"
        assert body["messages"][0]["content"][1] == original["messages"][0]["content"][1]
        assert body["inferenceConfig"] == original["inferenceConfig"]

    def test_fewer_guardrailed_texts_logs_warning(self, monkeypatch):
        body = {
            "messages": [
                {"role": "user", "content": [{"text": "a"}, {"text": "b"}]}
            ]
        }
        _, holders = _extract_converse_texts(body, skip_system=False, skip_tool=False)
        assert len(holders) == 2

        warnings = []
        monkeypatch.setattr(
            "litellm.llms.bedrock.passthrough.guardrail_translation.handler.verbose_proxy_logger.warning",
            lambda *args, **kwargs: warnings.append(args),
        )

        _write_back_texts(["masked"], holders)

        assert warnings, "mismatched guardrail output count must not be silently dropped"
        assert body["messages"][0]["content"][0]["text"] == "masked"
        assert body["messages"][0]["content"][1]["text"] == "b"


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

        result = await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

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
        guardrail.apply_guardrail = AsyncMock(side_effect=GuardrailBlocked("Blocked"))

        with pytest.raises(GuardrailBlocked):
            await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

    @pytest.mark.asyncio
    async def test_tool_result_text_scanned_and_masked(self):
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data()
        data["data"]["messages"][0]["content"].append(
            {
                "toolResult": {
                    "toolUseId": "t1",
                    "content": [{"text": "My SSN is 123-45-6789"}],
                }
            }
        )
        guardrail = _make_guardrail(
            {"texts": ["You are helpful.", "Hello world", "[REDACTED]"]}
        )

        result = await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        sent_texts = guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"]
        assert "My SSN is 123-45-6789" in sent_texts
        tool_result = result["data"]["messages"][0]["content"][2]["toolResult"]
        assert tool_result["content"][0]["text"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_tool_result_json_scanned_and_masked(self):
        """A caller can hide blocked text under toolResult.content[].json; the
        guardrail must still see it and write the masked value back in place."""
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data()
        data["data"]["messages"][0]["content"].append(
            {
                "toolResult": {
                    "toolUseId": "t1",
                    "content": [{"json": {"note": "SSN 123-45-6789"}}],
                }
            }
        )
        guardrail = _make_guardrail(
            {"texts": ["You are helpful.", "Hello world", "[REDACTED]"]}
        )

        result = await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        sent_texts = guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"]
        assert "SSN 123-45-6789" in sent_texts
        tool_result = result["data"]["messages"][0]["content"][2]["toolResult"]
        assert tool_result["content"][0]["json"]["note"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_tool_use_input_scanned_and_masked(self):
        """Blocked text hidden in toolUse.input must be scanned and masked."""
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data()
        data["data"]["messages"][0]["content"][1]["toolUse"]["input"] = {
            "query": "email john@example.com"
        }
        guardrail = _make_guardrail(
            {"texts": ["You are helpful.", "Hello world", "[REDACTED]"]}
        )

        result = await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        sent_texts = guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"]
        assert "email john@example.com" in sent_texts
        tool_use = result["data"]["messages"][0]["content"][1]["toolUse"]
        assert tool_use["input"]["query"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_tool_use_input_blocking_propagates(self):
        """A blocking guardrail must reject content hidden in toolUse.input."""
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data()
        data["data"]["messages"][0]["content"][1]["toolUse"]["input"] = {
            "query": "blocked content"
        }
        guardrail = MagicMock()
        guardrail.guardrail_name = "block-guard"
        guardrail.skip_system_message_in_guardrail = False
        guardrail.skip_tool_message_in_guardrail = False
        guardrail.apply_guardrail = AsyncMock(side_effect=GuardrailBlocked("Blocked"))

        with pytest.raises(GuardrailBlocked):
            await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        sent_texts = guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"]
        assert "blocked content" in sent_texts

    @pytest.mark.asyncio
    async def test_tool_config_description_scanned_and_masked(self):
        """Blocked text hidden in toolConfig.tools[].toolSpec.description is still
        forwarded to Bedrock, so the guardrail must see it and mask it in place."""
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data()
        data["data"]["toolConfig"] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "lookup",
                        "description": "email john@example.com",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }
        guardrail = _make_guardrail(
            {"texts": ["You are helpful.", "Hello world", "lookup", "[REDACTED]", "object"]}
        )

        result = await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        sent_texts = guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"]
        assert "email john@example.com" in sent_texts
        tool_spec = result["data"]["toolConfig"]["tools"][0]["toolSpec"]
        assert tool_spec["description"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_tool_config_description_blocking_propagates(self):
        """A blocking guardrail must reject content hidden in a tool description."""
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data()
        data["data"]["toolConfig"] = {
            "tools": [{"toolSpec": {"name": "fn", "description": "blocked content"}}]
        }
        guardrail = MagicMock()
        guardrail.guardrail_name = "block-guard"
        guardrail.skip_system_message_in_guardrail = False
        guardrail.skip_tool_message_in_guardrail = False
        guardrail.apply_guardrail = AsyncMock(side_effect=GuardrailBlocked("Blocked"))

        with pytest.raises(GuardrailBlocked):
            await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        sent_texts = guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"]
        assert "blocked content" in sent_texts

    @pytest.mark.asyncio
    async def test_additional_model_request_fields_scanned_and_masked(self):
        """Blocked text hidden in additionalModelRequestFields is forwarded to
        Bedrock, so the guardrail must scan it and mask it in place."""
        handler = BedrockPassthroughGuardrailHandler()
        data = _converse_data()
        data["data"]["additionalModelRequestFields"] = {"note": "ssn 123-45-6789"}
        guardrail = _make_guardrail(
            {"texts": ["You are helpful.", "Hello world", "[REDACTED]"]}
        )

        result = await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        sent_texts = guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"]
        assert "ssn 123-45-6789" in sent_texts
        assert result["data"]["additionalModelRequestFields"]["note"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_non_converse_endpoint_scans_full_payload(self):
        """Invoke routes must not bypass guardrails: the full request payload is
        scanned so blocking guardrails still see user-controlled text."""
        handler = BedrockPassthroughGuardrailHandler()
        data = {
            "endpoint": "model/anthropic.claude-3-sonnet/invoke",
            "custom_llm_provider": "bedrock",
            "model": "anthropic.claude-3-sonnet",
            "data": {"messages": [{"role": "user", "content": "blocked invoke text"}]},
        }
        guardrail = _make_guardrail({"texts": []})

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        guardrail.apply_guardrail.assert_called_once()
        sent_texts = guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"]
        assert "blocked invoke text" in sent_texts[0]

    @pytest.mark.asyncio
    async def test_non_converse_endpoint_blocking_propagates(self):
        handler = BedrockPassthroughGuardrailHandler()
        data = {
            "endpoint": "model/anthropic.claude-3-sonnet/invoke-with-response-stream",
            "custom_llm_provider": "bedrock",
            "data": {"prompt": "blocked"},
        }
        guardrail = MagicMock()
        guardrail.guardrail_name = "block-guard"
        guardrail.apply_guardrail = AsyncMock(side_effect=GuardrailBlocked("Blocked"))

        with pytest.raises(GuardrailBlocked):
            await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

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

        await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

        call_args = guardrail.apply_guardrail.call_args
        assert call_args.kwargs["input_type"] == "response"
        assert call_args.kwargs["inputs"]["texts"] == ["Model reply"]

    @pytest.mark.asyncio
    async def test_response_masking_writes_back(self):
        handler = BedrockPassthroughGuardrailHandler()
        response = self._converse_response("Bad content")
        guardrail = _make_guardrail({"texts": ["[MASKED]"]})

        result = await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

        assert result["output"]["message"]["content"][0]["text"] == "[MASKED]"
        assert result["stopReason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_response_guardrail_returning_no_texts_preserves_output(self, monkeypatch):
        """A guardrail that returns no texts must leave the response untouched and
        not warn, mirroring the request path's empty-result guard."""
        handler = BedrockPassthroughGuardrailHandler()
        response = self._converse_response("Model reply")
        guardrail = _make_guardrail({"texts": []})

        warnings = []
        monkeypatch.setattr(
            "litellm.llms.bedrock.passthrough.guardrail_translation.handler.verbose_proxy_logger.warning",
            lambda *args, **kwargs: warnings.append(args),
        )

        result = await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

        assert not warnings
        assert result["output"]["message"]["content"][0]["text"] == "Model reply"

    @pytest.mark.asyncio
    async def test_response_reasoning_and_tooluse_extracted_and_masked(self):
        """Model output hidden in reasoningContent.reasoningText.text and
        toolUse.input must be scanned and masked, but the reasoning signature
        must be left untouched."""
        handler = BedrockPassthroughGuardrailHandler()
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "visible"},
                        {
                            "reasoningContent": {
                                "reasoningText": {
                                    "text": "thinking about john@example.com",
                                    "signature": "sig-do-not-touch",
                                }
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "1",
                                "name": "lookup",
                                "input": {"q": "ssn 123-45-6789"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "end_turn",
        }
        guardrail = _make_guardrail(
            {"texts": ["[V]", "[REASON]", "[INPUT]"]}
        )

        result = await handler.process_output_response(
            response=response, guardrail_to_apply=guardrail
        )

        sent_texts = guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"]
        assert "thinking about john@example.com" in sent_texts
        assert "ssn 123-45-6789" in sent_texts
        blocks = result["output"]["message"]["content"]
        assert blocks[0]["text"] == "[V]"
        reasoning_text = blocks[1]["reasoningContent"]["reasoningText"]
        assert reasoning_text["text"] == "[REASON]"
        assert reasoning_text["signature"] == "sig-do-not-touch"
        assert blocks[2]["toolUse"]["input"]["q"] == "[INPUT]"

    @pytest.mark.asyncio
    async def test_response_citations_content_extracted_and_masked(self):
        """citationsContent.content[].text is grounded answer text and must be
        scanned, while citation sources/titles are left untouched."""
        handler = BedrockPassthroughGuardrailHandler()
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "citationsContent": {
                                "content": [{"text": "Contact john@example.com"}],
                                "citations": [
                                    {"source": "https://example.com", "title": "Example"}
                                ],
                            }
                        }
                    ],
                }
            },
            "stopReason": "end_turn",
        }
        guardrail = _make_guardrail({"texts": ["[CITED]"]})

        result = await handler.process_output_response(
            response=response, guardrail_to_apply=guardrail
        )

        sent_texts = guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"]
        assert sent_texts == ["Contact john@example.com"]
        citations = result["output"]["message"]["content"][0]["citationsContent"]
        assert citations["content"][0]["text"] == "[CITED]"
        assert citations["citations"][0]["source"] == "https://example.com"
        assert citations["citations"][0]["title"] == "Example"

    @pytest.mark.asyncio
    async def test_non_dict_response_returned_unchanged(self):
        handler = BedrockPassthroughGuardrailHandler()
        guardrail = _make_guardrail({"texts": []})

        result = await handler.process_output_response(response="raw string", guardrail_to_apply=guardrail)

        assert result == "raw string"
        guardrail.apply_guardrail.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_output_structure_skips(self):
        handler = BedrockPassthroughGuardrailHandler()
        response = {"stopReason": "end_turn"}
        guardrail = _make_guardrail({"texts": []})

        result = await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

        assert result == {"stopReason": "end_turn"}
        guardrail.apply_guardrail.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_converse_response_scanned_via_generic_handler(self):
        """Invoke responses are not Converse-shaped; they must still be scanned
        through the generic passthrough handler rather than skipped."""
        handler = BedrockPassthroughGuardrailHandler()
        response = {"completion": "blocked model output"}
        guardrail = _make_guardrail({"texts": []})

        await handler.process_output_response(
            response=response,
            guardrail_to_apply=guardrail,
            request_data={"endpoint": "model/anthropic.claude-3-sonnet/invoke"},
        )

        guardrail.apply_guardrail.assert_called_once()
        sent_texts = guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"]
        assert "blocked model output" in sent_texts[0]


def _build_event_stream_frame(event_type: str, payload: dict) -> bytes:
    import json
    import struct
    from binascii import crc32 as esm_crc32

    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()

    def _encode_str_header(name: str, value: str) -> bytes:
        name_b = name.encode()
        value_b = value.encode()
        return (
            struct.pack("!B", len(name_b)) + name_b + struct.pack("!B", 7) + struct.pack("!H", len(value_b)) + value_b
        )

    headers_bytes = (
        _encode_str_header(":event-type", event_type)
        + _encode_str_header(":content-type", "application/json")
        + _encode_str_header(":message-type", "event")
    )

    headers_length = len(headers_bytes)
    total_length = 12 + headers_length + len(payload_bytes) + 4
    prelude = struct.pack("!II", total_length, headers_length)
    prelude_crc_val = esm_crc32(prelude) & 0xFFFFFFFF
    prelude_crc_b = struct.pack("!I", prelude_crc_val)
    part_for_msg = prelude_crc_b + headers_bytes + payload_bytes
    msg_crc_val = esm_crc32(part_for_msg, prelude_crc_val) & 0xFFFFFFFF
    msg_crc_b = struct.pack("!I", msg_crc_val)
    return prelude + prelude_crc_b + headers_bytes + payload_bytes + msg_crc_b


class TestDeAnonymizeConverseStream:
    def _make_proxy_logging(self, mock_hook) -> MagicMock:
        proxy_logging_obj = MagicMock()
        proxy_logging_obj.post_call_success_hook = mock_hook
        return proxy_logging_obj

    @pytest.mark.asyncio
    async def test_text_delta_de_anonymized_in_modified_bytes(self):
        import json
        from botocore.eventstream import EventStreamBuffer

        stream_bytes = (
            _build_event_stream_frame("messageStart", {"role": "assistant"})
            + _build_event_stream_frame(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"text": "<PERSON_1> works at <ORG_2>"}},
            )
            + _build_event_stream_frame("contentBlockStop", {"contentBlockIndex": 0})
            + _build_event_stream_frame("messageStop", {"stopReason": "end_turn"})
        )

        de_anon_response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "John Doe works at Acme Corp"}],
                }
            },
            "stopReason": "end_turn",
        }

        async def mock_hook(data, user_api_key_dict, response):
            return de_anon_response

        result = await BedrockPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=stream_bytes,
            proxy_logging_obj=self._make_proxy_logging(mock_hook),
            user_api_key_dict=MagicMock(),
            data={},
        )

        buf = EventStreamBuffer()
        buf.add_data(result)
        texts = [
            json.loads(msg.payload)["delta"]["text"]
            for msg in buf
            if msg.headers.get(":event-type") == "contentBlockDelta"
        ]
        assert "".join(texts) == "John Doe works at Acme Corp"

    @pytest.mark.asyncio
    async def test_tokens_split_across_chunks_reassembled(self):
        import json
        from botocore.eventstream import EventStreamBuffer

        stream_bytes = _build_event_stream_frame(
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"text": "<PERSON_"}},
        ) + _build_event_stream_frame(
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"text": "1> called."}},
        )

        de_anon_response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Alice called."}],
                }
            },
            "stopReason": "end_turn",
        }

        async def mock_hook(data, user_api_key_dict, response):
            assert response["output"]["message"]["content"][0]["text"] == "<PERSON_1> called."
            return de_anon_response

        result = await BedrockPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=stream_bytes,
            proxy_logging_obj=self._make_proxy_logging(mock_hook),
            user_api_key_dict=MagicMock(),
            data={},
        )

        buf = EventStreamBuffer()
        buf.add_data(result)
        texts = [
            json.loads(msg.payload)["delta"]["text"]
            for msg in buf
            if msg.headers.get(":event-type") == "contentBlockDelta"
        ]
        assert "".join(texts) == "Alice called."

    @pytest.mark.asyncio
    async def test_non_text_frames_preserved_unchanged(self):
        from botocore.eventstream import EventStreamBuffer

        stream_bytes = (
            _build_event_stream_frame("messageStart", {"role": "assistant"})
            + _build_event_stream_frame(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"text": "<PERSON_1>"}},
            )
            + _build_event_stream_frame("messageStop", {"stopReason": "end_turn"})
        )

        de_anon_response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Bob"}]}},
            "stopReason": "end_turn",
        }

        async def mock_hook(data, user_api_key_dict, response):
            return de_anon_response

        result = await BedrockPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=stream_bytes,
            proxy_logging_obj=self._make_proxy_logging(mock_hook),
            user_api_key_dict=MagicMock(),
            data={},
        )

        buf = EventStreamBuffer()
        buf.add_data(result)
        event_types = [msg.headers.get(":event-type") for msg in buf]
        assert "messageStart" in event_types
        assert "messageStop" in event_types
        assert event_types.count("contentBlockDelta") == 1

    @pytest.mark.asyncio
    async def test_no_text_deltas_returns_original_bytes(self):
        stream_bytes = _build_event_stream_frame("messageStart", {"role": "assistant"})

        hook_spy = AsyncMock()

        result = await BedrockPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=stream_bytes,
            proxy_logging_obj=self._make_proxy_logging(hook_spy),
            user_api_key_dict=MagicMock(),
            data={},
        )

        hook_spy.assert_not_called()
        assert result is stream_bytes

    @pytest.mark.asyncio
    async def test_text_distributed_proportionally_across_chunks(self):
        import json
        from botocore.eventstream import EventStreamBuffer

        stream_bytes = _build_event_stream_frame(
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"text": "<PERSON_1>"}},
        ) + _build_event_stream_frame(
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"text": "<ORG>"}},
        )

        de_anon_response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "John Acme"}],
                }
            },
            "stopReason": "end_turn",
        }

        async def mock_hook(data, user_api_key_dict, response):
            return de_anon_response

        result = await BedrockPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=stream_bytes,
            proxy_logging_obj=self._make_proxy_logging(mock_hook),
            user_api_key_dict=MagicMock(),
            data={},
        )

        buf = EventStreamBuffer()
        buf.add_data(result)
        texts = [
            json.loads(msg.payload)["delta"]["text"]
            for msg in buf
            if msg.headers.get(":event-type") == "contentBlockDelta"
        ]
        assert "".join(texts) == "John Acme"
        assert all(t != "" for t in texts), f"Expected no empty chunks, got: {texts}"

    @pytest.mark.asyncio
    async def test_trailing_bytes_after_last_frame_preserved(self):
        import json
        from botocore.eventstream import EventStreamBuffer

        trailing = b"\xde\xad\xbe"
        stream_bytes = (
            _build_event_stream_frame(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"text": "<PERSON_1>"}},
            )
            + trailing
        )

        de_anon_response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Jane"}]}},
            "stopReason": "end_turn",
        }

        async def mock_hook(data, user_api_key_dict, response):
            return de_anon_response

        result = await BedrockPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=stream_bytes,
            proxy_logging_obj=self._make_proxy_logging(mock_hook),
            user_api_key_dict=MagicMock(),
            data={},
        )

        assert result.endswith(trailing)
        buf = EventStreamBuffer()
        buf.add_data(result[: -len(trailing)])
        texts = [
            json.loads(msg.payload)["delta"]["text"]
            for msg in buf
            if msg.headers.get(":event-type") == "contentBlockDelta"
        ]
        assert "".join(texts) == "Jane"

    @staticmethod
    def _token_replacing_hook(mapping: dict):
        async def mock_hook(data, user_api_key_dict, response):
            for block in response["output"]["message"]["content"]:
                text = block["text"]
                for token, value in mapping.items():
                    text = text.replace(token, value)
                block["text"] = text
            return response

        return mock_hook

    def _decode_deltas(self, result: bytes) -> list:
        import json
        from botocore.eventstream import EventStreamBuffer

        buf = EventStreamBuffer()
        buf.add_data(result)
        return [
            json.loads(msg.payload)["delta"]
            for msg in buf
            if msg.headers.get(":event-type") == "contentBlockDelta"
        ]

    @pytest.mark.asyncio
    async def test_reasoning_text_delta_de_anonymized(self):
        """Reasoning deltas carry model output; their text must be guardrailed while the reasoning signature is left untouched."""
        stream_bytes = (
            _build_event_stream_frame("messageStart", {"role": "assistant"})
            + _build_event_stream_frame(
                "contentBlockDelta",
                {
                    "contentBlockIndex": 0,
                    "delta": {
                        "reasoningContent": {
                            "text": "thinking about <PERSON_1>",
                            "signature": "sig-do-not-touch",
                        }
                    },
                },
            )
            + _build_event_stream_frame("messageStop", {"stopReason": "end_turn"})
        )

        result = await BedrockPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=stream_bytes,
            proxy_logging_obj=self._make_proxy_logging(
                self._token_replacing_hook({"<PERSON_1>": "Alice"})
            ),
            user_api_key_dict=MagicMock(),
            data={},
        )

        deltas = self._decode_deltas(result)
        assert deltas[0]["reasoningContent"]["text"] == "thinking about Alice"
        assert deltas[0]["reasoningContent"]["signature"] == "sig-do-not-touch"

    @pytest.mark.asyncio
    async def test_tool_use_input_delta_de_anonymized(self):
        """toolUse.input deltas carry model-generated tool arguments and must be guardrailed instead of being forwarded raw."""
        stream_bytes = _build_event_stream_frame(
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"toolUse": {"input": '{"q":"<PERSON_1>"}'}}},
        )

        result = await BedrockPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=stream_bytes,
            proxy_logging_obj=self._make_proxy_logging(
                self._token_replacing_hook({"<PERSON_1>": "Alice"})
            ),
            user_api_key_dict=MagicMock(),
            data={},
        )

        deltas = self._decode_deltas(result)
        assert deltas[0]["toolUse"]["input"] == '{"q":"Alice"}'

    @pytest.mark.asyncio
    async def test_citations_content_delta_de_anonymized(self):
        """citationsContent grounded text must be guardrailed while citation sources are preserved."""
        stream_bytes = _build_event_stream_frame(
            "contentBlockDelta",
            {
                "contentBlockIndex": 0,
                "delta": {
                    "citationsContent": {
                        "content": [{"text": "Contact <PERSON_1>"}],
                        "citations": [{"source": "https://example.com", "title": "Example"}],
                    }
                },
            },
        )

        result = await BedrockPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=stream_bytes,
            proxy_logging_obj=self._make_proxy_logging(
                self._token_replacing_hook({"<PERSON_1>": "Alice"})
            ),
            user_api_key_dict=MagicMock(),
            data={},
        )

        citations = self._decode_deltas(result)[0]["citationsContent"]
        assert citations["content"][0]["text"] == "Contact Alice"
        assert citations["citations"][0]["source"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_text_and_reasoning_deltas_de_anonymized_independently(self):
        """Distinct delta kinds must each be guardrailed and written back into their own field without bleeding the de-anonymized text across kinds."""
        captured = {}

        async def mock_hook(data, user_api_key_dict, response):
            captured["texts"] = [
                b["text"] for b in response["output"]["message"]["content"]
            ]
            mapping = {"<PERSON_1>": "Alice", "<ORG_2>": "Acme"}
            for block in response["output"]["message"]["content"]:
                text = block["text"]
                for token, value in mapping.items():
                    text = text.replace(token, value)
                block["text"] = text
            return response

        stream_bytes = _build_event_stream_frame(
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"text": "Hi <PERSON_1>"}},
        ) + _build_event_stream_frame(
            "contentBlockDelta",
            {"contentBlockIndex": 1, "delta": {"reasoningContent": {"text": "works at <ORG_2>"}}},
        )

        result = await BedrockPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=stream_bytes,
            proxy_logging_obj=self._make_proxy_logging(mock_hook),
            user_api_key_dict=MagicMock(),
            data={},
        )

        assert "Hi <PERSON_1>" in captured["texts"]
        assert "works at <ORG_2>" in captured["texts"]
        deltas = self._decode_deltas(result)
        assert deltas[0]["text"] == "Hi Alice"
        assert deltas[1]["reasoningContent"]["text"] == "works at Acme"

    @pytest.mark.asyncio
    async def test_reasoning_signature_only_frame_left_unmodified(self):
        """A reasoning delta carrying only a signature has no guardrailable text; it must be forwarded untouched and the guardrail must not run."""
        stream_bytes = _build_event_stream_frame(
            "contentBlockDelta",
            {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}}},
        )
        hook_spy = AsyncMock()

        result = await BedrockPassthroughGuardrailHandler.de_anonymize_event_stream(
            body_bytes=stream_bytes,
            proxy_logging_obj=self._make_proxy_logging(hook_spy),
            user_api_key_dict=MagicMock(),
            data={},
        )

        hook_spy.assert_not_called()
        assert result is stream_bytes
