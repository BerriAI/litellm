"""
Tests for the Google GenAI generateContent guardrail translation handler.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.llms.gemini.google_genai.guardrail_translation.handler import (
    GoogleGenAIGenerateContentHandler,
)
from litellm.types.utils import CallTypes


class GuardrailBlockedError(Exception):
    pass


def _mock_guardrail(returned_texts):
    guardrail = MagicMock()
    guardrail.apply_guardrail = AsyncMock(return_value={"texts": returned_texts})
    return guardrail


@pytest.mark.asyncio
async def test_input_contents_text_is_guardrailed_and_written_back():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = _mock_guardrail(["masked question"])
    data = {
        "model": "gemini-2.5-flash",
        "contents": [{"role": "user", "parts": [{"text": "raw question"}]}],
    }

    result = await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

    call_kwargs = guardrail.apply_guardrail.call_args.kwargs
    assert call_kwargs["inputs"]["texts"] == ["raw question"]
    assert call_kwargs["inputs"]["model"] == "gemini-2.5-flash"
    assert call_kwargs["input_type"] == "request"
    assert result["contents"][0]["parts"][0]["text"] == "masked question"


@pytest.mark.asyncio
async def test_input_system_instruction_text_is_scanned_and_written_back():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = _mock_guardrail(["masked instruction", "masked question"])
    data = {
        "model": "gemini-2.5-flash",
        "systemInstruction": {"role": "system", "parts": [{"text": "prohibited instruction"}]},
        "contents": [{"role": "user", "parts": [{"text": "benign question"}]}],
    }

    result = await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"] == [
        "prohibited instruction",
        "benign question",
    ]
    assert result["systemInstruction"]["parts"][0]["text"] == "masked instruction"
    assert result["contents"][0]["parts"][0]["text"] == "masked question"


@pytest.mark.asyncio
async def test_input_config_nested_snake_case_system_instruction_is_scanned():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = _mock_guardrail(["clean"])
    instruction_part = SimpleNamespace(text="prohibited instruction")
    data = {
        "contents": [],
        "config": SimpleNamespace(system_instruction=SimpleNamespace(parts=[instruction_part])),
    }

    await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"] == ["prohibited instruction"]
    assert instruction_part.text == "clean"


@pytest.mark.asyncio
async def test_input_without_text_skips_guardrail():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = _mock_guardrail([])
    data = {"model": "gemini-2.5-flash", "contents": [{"role": "user", "parts": [{"inlineData": {}}]}]}

    result = await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

    guardrail.apply_guardrail.assert_not_called()
    assert result is data


@pytest.mark.asyncio
async def test_output_dict_response_text_is_guardrailed_and_written_back():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = _mock_guardrail(["masked answer"])
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "harmful answer"}]},
                "finishReason": "STOP",
            }
        ]
    }

    result = await handler.process_output_response(
        response=response,
        guardrail_to_apply=guardrail,
        request_data={"model": "gemini-2.5-flash"},
    )

    call_kwargs = guardrail.apply_guardrail.call_args.kwargs
    assert call_kwargs["inputs"]["texts"] == ["harmful answer"]
    assert call_kwargs["input_type"] == "response"
    assert call_kwargs["request_data"]["response"] is response
    assert result["candidates"][0]["content"]["parts"][0]["text"] == "masked answer"


@pytest.mark.asyncio
async def test_output_object_response_text_is_guardrailed_and_written_back():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = _mock_guardrail(["masked answer"])
    part = SimpleNamespace(text="harmful answer")
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason="STOP")]
    )

    await handler.process_output_response(response=response, guardrail_to_apply=guardrail)

    assert guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"] == ["harmful answer"]
    assert part.text == "masked answer"


@pytest.mark.asyncio
async def test_output_without_text_skips_guardrail():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = _mock_guardrail([])

    result = await handler.process_output_response(response={"candidates": []}, guardrail_to_apply=guardrail)

    guardrail.apply_guardrail.assert_not_called()
    assert result == {"candidates": []}


@pytest.mark.asyncio
async def test_output_blocking_guardrail_exception_propagates():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = MagicMock()
    guardrail.apply_guardrail = AsyncMock(side_effect=GuardrailBlockedError("blocked"))
    response = {"candidates": [{"content": {"parts": [{"text": "harmful answer"}]}}]}

    with pytest.raises(GuardrailBlockedError):
        await handler.process_output_response(response=response, guardrail_to_apply=guardrail)


@pytest.mark.asyncio
async def test_streaming_dict_chunks_accumulate_text():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = _mock_guardrail(["clean"])
    chunks = [
        {"candidates": [{"content": {"parts": [{"text": "harmful "}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "answer"}]}}]},
    ]

    result = await handler.process_output_streaming_response(
        responses_so_far=chunks,
        guardrail_to_apply=guardrail,
    )

    call_kwargs = guardrail.apply_guardrail.call_args.kwargs
    assert call_kwargs["inputs"]["texts"] == ["harmful answer"]
    assert call_kwargs["input_type"] == "response"
    assert result is chunks


@pytest.mark.asyncio
async def test_streaming_raw_sse_chunks_accumulate_text_across_split_frames():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = _mock_guardrail(["clean"])
    frame_one = "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "harmful "}]}}]}) + "\r\n\r\n"
    frame_two = "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "answer"}]}}]}) + "\r\n\r\n"
    split_at = len(frame_one) // 2
    chunks = [frame_one[:split_at], frame_one[split_at:] + frame_two[:5], frame_two[5:].encode("utf-8")]

    await handler.process_output_streaming_response(
        responses_so_far=chunks,
        guardrail_to_apply=guardrail,
    )

    assert guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"] == ["harmful answer"]


@pytest.mark.asyncio
async def test_streaming_blocking_guardrail_exception_propagates():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = MagicMock()
    guardrail.apply_guardrail = AsyncMock(side_effect=GuardrailBlockedError("blocked"))
    chunks = [{"candidates": [{"content": {"parts": [{"text": "harmful"}]}}]}]

    with pytest.raises(GuardrailBlockedError):
        await handler.process_output_streaming_response(responses_so_far=chunks, guardrail_to_apply=guardrail)


@pytest.mark.asyncio
async def test_streaming_without_text_skips_guardrail():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = _mock_guardrail([])

    result = await handler.process_output_streaming_response(responses_so_far=[], guardrail_to_apply=guardrail)

    guardrail.apply_guardrail.assert_not_called()
    assert result == []


def test_generate_content_call_types_are_registered():
    from litellm.llms.gemini.google_genai.guardrail_translation import (
        guardrail_translation_mappings,
    )

    for call_type in (
        CallTypes.generate_content,
        CallTypes.agenerate_content,
        CallTypes.generate_content_stream,
        CallTypes.agenerate_content_stream,
    ):
        assert guardrail_translation_mappings[call_type] is GoogleGenAIGenerateContentHandler


def test_discovery_finds_generate_content_handler():
    from litellm.llms import load_guardrail_translation_mappings

    mappings = load_guardrail_translation_mappings()
    assert mappings[CallTypes.agenerate_content] is GoogleGenAIGenerateContentHandler
    assert mappings[CallTypes.agenerate_content_stream] is GoogleGenAIGenerateContentHandler
