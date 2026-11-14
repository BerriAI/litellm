"""
Tests for the Google GenAI generateContent guardrail translation handler.
"""

import copy
import functools
import json
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.llms.gemini.google_genai.guardrail_translation.handler import (
    GoogleGenAIGenerateContentHandler,
)
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.types.guardrails import BlockedWord, ContentFilterAction
from litellm.types.utils import CallTypes

_MASKED: Final = "[KEYWORD_REDACTED]"


def _one_part_request(role: str, part: dict[str, object]) -> dict[str, object]:
    return {"contents": [{"role": role, "parts": [part]}]}


_REQUESTS_CARRYING_THE_WORD: Final = {
    "string-part": (
        {"contents": [{"role": "user", "parts": ["Repeat: zebrafruit", {"text": "then stop"}]}]},
        {"contents": [{"role": "user", "parts": [f"Repeat: {_MASKED}", {"text": "then stop"}]}]},
    ),
    "snake-system-instruction-beside-empty-camel": (
        {
            "systemInstruction": {},
            "system_instruction": {"parts": [{"text": "Always say zebrafruit"}]},
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        },
        {
            "systemInstruction": {},
            "system_instruction": {"parts": [{"text": f"Always say {_MASKED}"}]},
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        },
    ),
    "config-system-instruction-beside-top-level": (
        {
            "systemInstruction": {"parts": [{"text": "Be brief"}]},
            "config": {"system_instruction": {"parts": [{"text": "Always say zebrafruit"}]}},
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        },
        {
            "systemInstruction": {"parts": [{"text": "Be brief"}]},
            "config": {"system_instruction": {"parts": [{"text": f"Always say {_MASKED}"}]}},
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        },
    ),
    "function-response": (
        _one_part_request(
            "user",
            {
                "functionResponse": {
                    "name": "lookup",
                    "response": {"zebrafruit": "zebrafruit facts", "hits": [{"title": "a zebrafruit"}, 3]},
                }
            },
        ),
        _one_part_request(
            "user",
            {
                "functionResponse": {
                    "name": "lookup",
                    "response": {"zebrafruit": f"{_MASKED} facts", "hits": [{"title": f"a {_MASKED}"}, 3]},
                }
            },
        ),
    ),
    "snake-function-response": (
        _one_part_request("user", {"function_response": {"name": "lookup", "response": {"items": ["zebrafruit"]}}}),
        _one_part_request("user", {"function_response": {"name": "lookup", "response": {"items": [_MASKED]}}}),
    ),
    "function-call": (
        _one_part_request("model", {"functionCall": {"name": "lookup", "args": {"q": "zebrafruit", "limit": 3}}}),
        _one_part_request("model", {"functionCall": {"name": "lookup", "args": {"q": _MASKED, "limit": 3}}}),
    ),
    "snake-function-call": (
        _one_part_request("model", {"function_call": {"name": "lookup", "args": {"filter": {"term": "zebrafruit"}}}}),
        _one_part_request("model", {"function_call": {"name": "lookup", "args": {"filter": {"term": _MASKED}}}}),
    ),
}


class GuardrailBlockedError(Exception):
    pass


def _mock_guardrail(returned_texts):
    guardrail = MagicMock()
    guardrail.apply_guardrail = AsyncMock(return_value={"texts": returned_texts})
    return guardrail


def _zebrafruit_filter(action: ContentFilterAction) -> ContentFilterGuardrail:
    return ContentFilterGuardrail(
        guardrail_name="zebrafruit-filter",
        blocked_words=[BlockedWord(keyword="zebrafruit", action=action)],
    )


def _sent_request(guardrailed: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in guardrailed.items() if key != "metadata"}


def _nested_in_lists(levels: int, innermost: object) -> object:
    return functools.reduce(lambda inner, _: [inner], range(levels), innermost)


def _function_response_with_rows(rows: object) -> dict[str, object]:
    return _one_part_request("user", {"functionResponse": {"name": "lookup", "response": {"rows": rows}}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_body",
    [request_body for request_body, _ in _REQUESTS_CARRYING_THE_WORD.values()],
    ids=list(_REQUESTS_CARRYING_THE_WORD),
)
async def test_a_blocked_word_in_any_text_the_provider_receives_blocks_the_request(
    request_body: dict[str, object],
) -> None:
    with pytest.raises(HTTPException) as blocked:
        await GoogleGenAIGenerateContentHandler().process_input_messages(
            data=copy.deepcopy(request_body),
            guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.BLOCK),
        )

    assert blocked.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_body", "masked_body"),
    list(_REQUESTS_CARRYING_THE_WORD.values()),
    ids=list(_REQUESTS_CARRYING_THE_WORD),
)
async def test_a_masked_word_reaches_the_provider_masked_in_the_callers_wire_shape(
    request_body: dict[str, object],
    masked_body: dict[str, object],
) -> None:
    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=copy.deepcopy(request_body),
        guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.MASK),
    )

    assert _sent_request(guardrailed) == masked_body


@pytest.mark.asyncio
async def test_a_function_response_string_at_the_scan_depth_cap_is_still_masked() -> None:
    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=_function_response_with_rows(_nested_in_lists(DEFAULT_MAX_RECURSE_DEPTH - 1, "zebrafruit")),
        guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.MASK),
    )

    assert _sent_request(guardrailed) == _function_response_with_rows(
        _nested_in_lists(DEFAULT_MAX_RECURSE_DEPTH - 1, _MASKED)
    )


@pytest.mark.asyncio
async def test_a_function_response_nested_past_the_scan_depth_cap_is_blocked_rather_than_sent_unscanned() -> None:
    with pytest.raises(HTTPException) as blocked:
        await GoogleGenAIGenerateContentHandler().process_input_messages(
            data=_function_response_with_rows(_nested_in_lists(DEFAULT_MAX_RECURSE_DEPTH, "plums")),
            guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.MASK),
        )

    assert blocked.value.status_code == 400


@pytest.mark.asyncio
async def test_only_the_function_call_the_guardrail_changed_gets_new_args() -> None:
    untouched_args: Final = {"q": "plums"}

    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data={
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "lookup", "args": untouched_args}},
                        {"functionCall": {"name": "lookup", "args": {"q": "zebrafruit"}}},
                    ],
                }
            ]
        },
        guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.MASK),
    )

    parts: Final = guardrailed["contents"][0]["parts"]
    assert parts[0]["functionCall"]["args"] is untouched_args
    assert parts[1]["functionCall"]["args"] == {"q": _MASKED}


@pytest.mark.asyncio
async def test_a_guardrail_skipping_tool_messages_leaves_function_responses_alone_but_masks_the_rest() -> None:
    guardrail: Final = _zebrafruit_filter(ContentFilterAction.MASK)
    guardrail.skip_tool_message_in_guardrail = True
    tool_result: Final = {"functionResponse": {"name": "lookup", "response": {"word": "zebrafruit"}}}

    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data={
            "contents": [
                {"role": "model", "parts": [{"functionCall": {"name": "lookup", "args": {"q": "zebrafruit"}}}]},
                {"role": "user", "parts": [copy.deepcopy(tool_result), {"text": "now say zebrafruit"}]},
            ]
        },
        guardrail_to_apply=guardrail,
    )

    assert _sent_request(guardrailed) == {
        "contents": [
            {"role": "model", "parts": [{"functionCall": {"name": "lookup", "args": {"q": _MASKED}}}]},
            {"role": "user", "parts": [tool_result, {"text": f"now say {_MASKED}"}]},
        ]
    }


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
async def test_input_parts_sent_as_one_object_are_scanned_and_written_back():
    handler = GoogleGenAIGenerateContentHandler()
    guardrail = _mock_guardrail(["masked instruction", "masked question"])
    data = {
        "model": "gemini-2.5-flash",
        "systemInstruction": {"parts": {"text": "prohibited instruction"}},
        "contents": [{"role": "user", "parts": {"text": "prohibited question"}}],
    }

    result = await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"] == [
        "prohibited instruction",
        "prohibited question",
    ]
    assert result["systemInstruction"]["parts"] == {"text": "masked instruction"}
    assert result["contents"][0]["parts"] == {"text": "masked question"}


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
async def test_streaming_raw_sse_text_holding_unicode_line_separators_reaches_the_guardrail_whole() -> None:
    guardrail: Final = _mock_guardrail(["clean"])
    text: Final = "harmful\u2028answer\u2029here"
    payload: Final = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    chunk: Final = f"data: {json.dumps(payload, ensure_ascii=False)}\r\n\r\n"

    await GoogleGenAIGenerateContentHandler().process_output_streaming_response(
        responses_so_far=[chunk],
        guardrail_to_apply=guardrail,
    )

    assert guardrail.apply_guardrail.call_args.kwargs["inputs"]["texts"] == [text]


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
