"""
Tests for the Google GenAI generateContent guardrail translation handler.
"""

import copy
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from itertools import chain
from types import SimpleNamespace
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.base_llm.guardrail_translation.utils import UnappliableRequestRewrite
from litellm.llms.gemini.google_genai.guardrail_translation.handler import (
    GoogleGenAIGenerateContentHandler,
)
from litellm.types.llms.openai import ChatCompletionToolCallChunk
from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs


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


def _masking_card_numbers(tool_calls: Sequence[ChatCompletionToolCallChunk]) -> list[object]:
    return [
        {
            **tool_call,
            "function": {
                **tool_call["function"],
                "arguments": tool_call["function"]["arguments"].replace("4111", "[CARD]"),
            },
        }
        for tool_call in tool_calls
    ]


class _ToolCallGuardrail(CustomGuardrail):
    def __init__(
        self,
        scan_tool_calls: bool = True,
        rewrite: Callable[[Sequence[ChatCompletionToolCallChunk]], list[object]] = list,
    ) -> None:
        super().__init__(guardrail_name="tool-call-guardrail")
        self.scan_tool_calls_as_text = scan_tool_calls
        self.rewrite: Final = rewrite
        self.seen: Final[list[GenericGuardrailAPIInputs]] = []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: object | None = None,
    ) -> GenericGuardrailAPIInputs:
        self.seen.append(inputs)
        tool_calls: Final = inputs.get("tool_calls")
        if tool_calls is None:
            return inputs
        return {**inputs, "tool_calls": self.rewrite(tool_calls)}


FUNCTION_CALL_PART: Final = {"functionCall": {"name": "send_payment", "args": {"card": "4111 1111", "amount": 12.5}}}


async def _scan_request(guardrail: _ToolCallGuardrail, part: Mapping[str, object] = FUNCTION_CALL_PART) -> None:
    await GoogleGenAIGenerateContentHandler().process_input_messages(
        data={"model": "gemini-2.5-flash", "contents": [{"role": "model", "parts": [part]}]},
        guardrail_to_apply=guardrail,
    )


async def _scan_response(guardrail: _ToolCallGuardrail, part: Mapping[str, object] = FUNCTION_CALL_PART) -> None:
    await GoogleGenAIGenerateContentHandler().process_output_response(
        response={"candidates": [{"content": {"role": "model", "parts": [part]}}]},
        guardrail_to_apply=guardrail,
    )


async def _scan_sse_stream(guardrail: _ToolCallGuardrail, part: Mapping[str, object] = FUNCTION_CALL_PART) -> None:
    frame: Final = "data: " + json.dumps({"candidates": [{"content": {"parts": [part]}}]}) + "\r\n\r\n"
    await GoogleGenAIGenerateContentHandler().process_output_streaming_response(
        responses_so_far=[frame[:20], frame[20:].encode("utf-8")], guardrail_to_apply=guardrail
    )


async def _scan_object_stream(
    guardrail: _ToolCallGuardrail, part: Mapping[str, object] = FUNCTION_CALL_PART
) -> None:
    function_call: Final = part.get("functionCall")
    sdk_part: Final = SimpleNamespace(
        text=part.get("text"),
        function_call=SimpleNamespace(**function_call) if isinstance(function_call, dict) else None,
    )
    await GoogleGenAIGenerateContentHandler().process_output_streaming_response(
        responses_so_far=[SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[sdk_part]))])],
        guardrail_to_apply=guardrail,
    )


SCANS: Final = (
    pytest.param(_scan_request, id="request"),
    pytest.param(_scan_response, id="response"),
    pytest.param(_scan_sse_stream, id="sse-stream"),
    pytest.param(_scan_object_stream, id="object-stream"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("scan", SCANS)
async def test_function_call_parts_reach_the_guardrail_as_tool_calls_when_opted_in(
    scan: Callable[[_ToolCallGuardrail], object],
) -> None:
    guardrail: Final = _ToolCallGuardrail()

    await scan(guardrail)

    (inputs,) = guardrail.seen
    (tool_call,) = inputs["tool_calls"]
    assert tool_call["function"]["name"] == "send_payment"
    assert json.loads(tool_call["function"]["arguments"]) == {"card": "4111 1111", "amount": 12.5}


@pytest.mark.asyncio
@pytest.mark.parametrize("scan", SCANS)
async def test_text_without_function_calls_reaches_an_opted_in_guardrail(
    scan: Callable[[_ToolCallGuardrail, Mapping[str, object]], Awaitable[None]],
) -> None:
    guardrail: Final = _ToolCallGuardrail()

    await scan(guardrail, {"text": "Pay the invoice"})

    (inputs,) = guardrail.seen
    assert inputs["texts"] == ["Pay the invoice"]
    assert not inputs.get("tool_calls")


@pytest.mark.asyncio
@pytest.mark.parametrize("scan", SCANS)
async def test_function_call_parts_are_not_scanned_by_default(scan: Callable[[_ToolCallGuardrail], object]) -> None:
    guardrail: Final = _ToolCallGuardrail(scan_tool_calls=False)

    await scan(guardrail)

    assert guardrail.seen == []


@pytest.mark.asyncio
async def test_masked_function_call_args_are_written_back_and_unmasked_ones_are_left_alone() -> None:
    untouched_args: Final = {"query": "invoices"}
    sdk_function_call: Final = SimpleNamespace(name="charge", args={"card": "4111 2222"})
    data: Final = {
        "model": "gemini-2.5-flash",
        "contents": [
            {"role": "model", "parts": [{"functionCall": {"name": "search", "args": untouched_args}}]},
            {"role": "model", "parts": [{"function_call": {"name": "send_payment", "args": {"card": "4111 1111"}}}]},
            {"role": "model", "parts": [SimpleNamespace(text=None, function_call=sdk_function_call)]},
        ],
    }

    await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=data, guardrail_to_apply=_ToolCallGuardrail(rewrite=_masking_card_numbers)
    )

    assert data["contents"][0]["parts"][0]["functionCall"]["args"] is untouched_args
    assert data["contents"][1]["parts"][0]["function_call"]["args"] == {"card": "[CARD] 1111"}
    assert sdk_function_call.args == {"card": "[CARD] 2222"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rewrite",
    [
        pytest.param(lambda tool_calls: [], id="dropped"),
        pytest.param(
            lambda tool_calls: [{**call, "function": {**call["function"], "arguments": "[1]"}} for call in tool_calls],
            id="not-an-object",
        ),
    ],
)
async def test_function_call_rewrite_the_request_cannot_carry_is_rejected(
    rewrite: Callable[[Sequence[ChatCompletionToolCallChunk]], list[object]],
) -> None:
    data: Final = {
        "model": "gemini-2.5-flash",
        "contents": [{"role": "model", "parts": [copy.deepcopy(FUNCTION_CALL_PART)]}],
    }

    with pytest.raises(UnappliableRequestRewrite):
        await GoogleGenAIGenerateContentHandler().process_input_messages(
            data=data, guardrail_to_apply=_ToolCallGuardrail(rewrite=rewrite)
        )

    assert data["contents"][0]["parts"][0] == FUNCTION_CALL_PART


STREAMED_FUNCTION_CALL_PARTS: Final = (
    {"functionCall": {"name": "send_payment", "willContinue": True}},
    {
        "functionCall": {
            "partialArgs": [{"jsonPath": "$.card", "stringValue": "4111 ", "willContinue": True}],
            "willContinue": True,
        }
    },
    {"functionCall": {"partialArgs": [{"jsonPath": "$.card", "stringValue": "1111"}]}},
    {"functionCall": {"name": "search", "args": {"query": "invoices"}}},
)
SNAKE_CASE_FIELDS: Final = {"functionCall": "function_call", "partialArgs": "partial_args", "willContinue": "will_continue"}


def _sse_frames(parts: Sequence[object]) -> list[object]:
    return ["data: " + json.dumps({"candidates": [{"content": {"parts": [part]}}]}) + "\r\n\r\n" for part in parts]


def _sdk_value(value: object) -> object:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{SNAKE_CASE_FIELDS.get(key, key): item if key == "args" else _sdk_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_sdk_value(item) for item in value]
    return value


def _sdk_chunks(parts: Sequence[object]) -> list[object]:
    return [
        SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[_sdk_value({"text": None, **part})]))])
        for part in parts
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [_sse_frames, _sdk_chunks], ids=["sse-stream", "object-stream"])
async def test_streamed_partial_function_call_args_reach_the_guardrail_whole(
    chunks: Callable[[Sequence[object]], list[object]],
) -> None:
    guardrail: Final = _ToolCallGuardrail()

    await GoogleGenAIGenerateContentHandler().process_output_streaming_response(
        responses_so_far=chunks(STREAMED_FUNCTION_CALL_PARTS), guardrail_to_apply=guardrail
    )

    (inputs,) = guardrail.seen
    assert [(call["function"]["name"], call["function"]["arguments"]) for call in inputs["tool_calls"]] == [
        ("send_payment", json.dumps({"$.card": "4111 1111"})),
        ("search", json.dumps({"query": "invoices"})),
    ]


HIDDEN_CARD: Final = "4111-1111-1111-1111"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("next_call", "next_value"),
    [
        pytest.param({"args": {"card": "safe"}}, "safe", id="own-args"),
        pytest.param({"name": "export_records"}, "export_records", id="own-name"),
    ],
)
async def test_a_call_with_its_own_name_or_args_after_will_continue_hides_nothing_from_the_scan(
    next_call: Mapping[str, object], next_value: str
) -> None:
    first_call: Final = {"name": "send_payment", "willContinue": True, "args": {"card": HIDDEN_CARD}}
    data: Final = {
        "model": "gemini-2.5-flash",
        "contents": [{"role": "model", "parts": [{"functionCall": first_call}, {"functionCall": dict(next_call)}]}],
    }
    guardrail: Final = _ToolCallGuardrail()

    await GoogleGenAIGenerateContentHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    (inputs,) = guardrail.seen
    scanned: Final = json.dumps(inputs["tool_calls"])
    assert HIDDEN_CARD in scanned
    assert next_value in scanned


def _partial_card_text(parts: Sequence[Mapping[str, object]]) -> str:
    partial_args: Final = chain.from_iterable(part["functionCall"].get("partialArgs", ()) for part in parts)
    return "".join(partial_arg["stringValue"] for partial_arg in partial_args)


@pytest.mark.asyncio
async def test_masked_partial_function_call_args_in_the_request_are_written_back() -> None:
    data: Final = {
        "model": "gemini-2.5-flash",
        "contents": [{"role": "model", "parts": copy.deepcopy(list(STREAMED_FUNCTION_CALL_PARTS))}],
    }

    await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=data, guardrail_to_apply=_ToolCallGuardrail(rewrite=_masking_card_numbers)
    )

    parts: Final = data["contents"][0]["parts"]
    assert _partial_card_text(parts) == "[CARD] 1111"
    assert parts[0] == STREAMED_FUNCTION_CALL_PARTS[0]
    assert parts[3] == STREAMED_FUNCTION_CALL_PARTS[3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("partial_args", "masked_partial_args"),
    [
        pytest.param(
            [{"jsonPath": "$.card", "numberValue": 4111}],
            [{"jsonPath": "$.card", "stringValue": "[CARD]"}],
            id="camel-case",
        ),
        pytest.param(
            [
                {"json_path": "$.card", "number_value": 4111, "will_continue": True},
                {"json_path": "$.card", "number_value": 1111},
            ],
            [
                {"json_path": "$.card", "will_continue": True, "string_value": "[CARD]1111"},
                {"json_path": "$.card", "string_value": ""},
            ],
            id="snake-case-split",
        ),
    ],
)
async def test_masked_number_in_partial_function_call_args_is_written_back_as_a_string(
    partial_args: list[dict[str, object]], masked_partial_args: list[dict[str, object]]
) -> None:
    data: Final = {
        "model": "gemini-2.5-flash",
        "contents": [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "send_payment", "willContinue": True}},
                    {"functionCall": {"partialArgs": copy.deepcopy(partial_args)}},
                ],
            }
        ],
    }

    await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=data, guardrail_to_apply=_ToolCallGuardrail(rewrite=_masking_card_numbers)
    )

    assert data["contents"][0]["parts"][1]["functionCall"]["partialArgs"] == masked_partial_args
