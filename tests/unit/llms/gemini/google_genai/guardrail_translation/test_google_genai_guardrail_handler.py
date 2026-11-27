"""
Tests for the Google GenAI generateContent guardrail translation handler.
"""

import copy
import functools
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from itertools import chain
from types import SimpleNamespace
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.guardrail_translation.utils import UnappliableRequestRewrite
from litellm.llms.gemini.google_genai.guardrail_translation.handler import (
    GoogleGenAIGenerateContentHandler,
)
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.types.guardrails import BlockedWord, ContentFilterAction, ContentFilterPattern
from litellm.types.llms.openai import ChatCompletionToolCallChunk
from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs

_MASKED: Final = "[KEYWORD_REDACTED]"
_DEEPER_THAN_ANY_SCAN_CAP: Final = 150


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
                    "response": {_MASKED: f"{_MASKED} facts", "hits": [{"title": f"a {_MASKED}"}, 3]},
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
    "executable-code": (
        _one_part_request("model", {"executableCode": {"language": "PYTHON", "code": "print('zebrafruit')"}}),
        _one_part_request("model", {"executableCode": {"language": "PYTHON", "code": f"print('{_MASKED}')"}}),
    ),
    "snake-executable-code": (
        _one_part_request("model", {"executable_code": {"language": "PYTHON", "code": "print('zebrafruit')"}}),
        _one_part_request("model", {"executable_code": {"language": "PYTHON", "code": f"print('{_MASKED}')"}}),
    ),
    "code-execution-result": (
        _one_part_request("user", {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "zebrafruit"}}),
        _one_part_request("user", {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": _MASKED}}),
    ),
    "snake-code-execution-result": (
        _one_part_request("user", {"code_execution_result": {"outcome": "OUTCOME_OK", "output": "zebrafruit"}}),
        _one_part_request("user", {"code_execution_result": {"outcome": "OUTCOME_OK", "output": _MASKED}}),
    ),
    "tool-call": (
        _one_part_request("model", {"toolCall": {"toolType": "GOOGLE_SEARCH_WEB", "args": {"query": "zebrafruit"}}}),
        _one_part_request("model", {"toolCall": {"toolType": "GOOGLE_SEARCH_WEB", "args": {"query": _MASKED}}}),
    ),
    "snake-tool-call": (
        _one_part_request("model", {"tool_call": {"tool_type": "GOOGLE_SEARCH_WEB", "args": {"query": "zebrafruit"}}}),
        _one_part_request("model", {"tool_call": {"tool_type": "GOOGLE_SEARCH_WEB", "args": {"query": _MASKED}}}),
    ),
    "tool-response": (
        _one_part_request(
            "user", {"toolResponse": {"toolType": "GOOGLE_SEARCH_WEB", "response": {"hits": ["zebrafruit"]}}}
        ),
        _one_part_request("user", {"toolResponse": {"toolType": "GOOGLE_SEARCH_WEB", "response": {"hits": [_MASKED]}}}),
    ),
    "snake-tool-response": (
        _one_part_request(
            "user", {"tool_response": {"tool_type": "GOOGLE_SEARCH_WEB", "response": {"hits": ["zebrafruit"]}}}
        ),
        _one_part_request(
            "user", {"tool_response": {"tool_type": "GOOGLE_SEARCH_WEB", "response": {"hits": [_MASKED]}}}
        ),
    ),
    "snake-function-call": (
        _one_part_request("model", {"function_call": {"name": "lookup", "args": {"filter": {"term": "zebrafruit"}}}}),
        _one_part_request("model", {"function_call": {"name": "lookup", "args": {"filter": {"term": _MASKED}}}}),
    ),
}


class GuardrailBlockedError(Exception):
    pass


def _mock_guardrail(returned_texts):
    guardrail = MagicMock(skip_system_message_in_guardrail=False, skip_tool_message_in_guardrail=False)
    guardrail.apply_guardrail = AsyncMock(return_value={"texts": returned_texts})
    return guardrail


class _TruncatingGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(guardrail_name="truncator")

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        return GenericGuardrailAPIInputs(texts=[text[:-1] for text in inputs.get("texts", [])])


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
async def test_a_clean_tool_result_nested_deeper_than_any_scan_cap_reaches_the_provider_unchanged() -> None:
    sent: Final = _function_response_with_rows(_nested_in_lists(_DEEPER_THAN_ANY_SCAN_CAP, "plums"))

    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=copy.deepcopy(sent),
        guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.BLOCK),
    )

    assert _sent_request(guardrailed) == sent


@pytest.mark.asyncio
async def test_a_number_nested_deeper_than_any_scan_cap_keeps_its_type_when_another_value_is_masked() -> None:
    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=_function_response_with_rows(["zebrafruit", _nested_in_lists(_DEEPER_THAN_ANY_SCAN_CAP, 3)]),
        guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.MASK),
    )

    assert _sent_request(guardrailed) == _function_response_with_rows(
        [_MASKED, _nested_in_lists(_DEEPER_THAN_ANY_SCAN_CAP, 3)]
    )


@pytest.mark.asyncio
async def test_a_masked_word_deep_in_a_function_response_is_masked_where_it_sits() -> None:
    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=_function_response_with_rows(_nested_in_lists(_DEEPER_THAN_ANY_SCAN_CAP, "zebrafruit")),
        guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.MASK),
    )

    assert _sent_request(guardrailed) == _function_response_with_rows(
        _nested_in_lists(_DEEPER_THAN_ANY_SCAN_CAP, _MASKED)
    )


def _card_filter(action: ContentFilterAction) -> ContentFilterGuardrail:
    return ContentFilterGuardrail(
        guardrail_name="card-filter",
        patterns=[ContentFilterPattern(pattern_type="regex", pattern=r"\b4\d{15}\b", name="card", action=action)],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [[{"card": 4111111111111111}], ["first line\n4111111111111111"], ["first line\r\n\t4111111111111111"]],
    ids=["json-number", "line-start", "after-tab"],
)
async def test_a_card_number_in_a_function_response_is_blocked(rows: list[object]) -> None:
    with pytest.raises(HTTPException) as blocked:
        await GoogleGenAIGenerateContentHandler().process_input_messages(
            data=_function_response_with_rows(rows),
            guardrail_to_apply=_card_filter(ContentFilterAction.BLOCK),
        )

    assert blocked.value.status_code == 400


@pytest.mark.asyncio
async def test_a_masked_card_number_replaces_the_json_number_and_other_values_keep_their_type() -> None:
    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=_function_response_with_rows(
            [{"card": 4111111111111111, "count": 3, "rate": 2.5, "ok": True, "id": "7", "note": 'C:\\new "x"\nok'}]
        ),
        guardrail_to_apply=_card_filter(ContentFilterAction.MASK),
    )

    sent_rows: Final = guardrailed["contents"][0]["parts"][0]["functionResponse"]["response"]["rows"]
    assert sent_rows == [
        {"card": sent_rows[0]["card"], "count": 3, "rate": 2.5, "ok": True, "id": "7", "note": 'C:\\new "x"\nok'}
    ]
    assert isinstance(sent_rows[0]["card"], str)
    assert "4111111111111111" not in sent_rows[0]["card"]
    assert type(sent_rows[0]["count"]) is int


def _without_note_in_reverse_key_order(text: str) -> str:
    fields: Final = json.loads(text)
    return json.dumps({key: fields[key] for key in reversed(fields) if key != "note"})


class _RestructuringGuardrail(CustomGuardrail):
    def __init__(self, rewrite: Callable[[str], str] = _without_note_in_reverse_key_order) -> None:
        super().__init__(guardrail_name="restructurer")
        self.rewrite: Final = rewrite

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        return GenericGuardrailAPIInputs(texts=[self.rewrite(text) for text in inputs["texts"]])


@pytest.mark.asyncio
async def test_numbers_keep_their_type_when_a_guardrail_reorders_and_drops_tool_result_fields() -> None:
    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=_one_part_request(
            "user",
            {
                "functionResponse": {
                    "name": "lookup",
                    "response": {"count": 3, "rate": 2.5, "id": "7", "note": "drop me", "rows": [1, {"n": 2}]},
                }
            },
        ),
        guardrail_to_apply=_RestructuringGuardrail(),
    )

    sent_response: Final = guardrailed["contents"][0]["parts"][0]["functionResponse"]["response"]
    assert sent_response == {"rows": [1, {"n": 2}], "id": "7", "rate": 2.5, "count": 3}
    assert type(sent_response["count"]) is int


def _without_zebrafruit_rows(text: str) -> str:
    fields: Final = json.loads(text.replace("plumcot", _MASKED))
    return json.dumps({**fields, "rows": [row for row in fields["rows"] if row != "zebrafruit"]})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "kept_rows"),
    [
        ([1, "zebrafruit", 2.5, {"n": 3}], [1, 2.5, {"n": 3}]),
        (["1", 1, "zebrafruit"], ["1", 1]),
        ([1, "1", "zebrafruit", 1], [1, "1", 1]),
        ([{"n": 3, "s": "plumcot"}, "zebrafruit", 4], [{"n": 3, "s": _MASKED}, 4]),
    ],
)
async def test_list_items_keep_their_type_when_a_guardrail_drops_a_tool_result_list_item(
    rows: list[object], kept_rows: list[object]
) -> None:
    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=_function_response_with_rows(rows),
        guardrail_to_apply=_RestructuringGuardrail(rewrite=_without_zebrafruit_rows),
    )

    assert _sent_request(guardrailed) == _function_response_with_rows(kept_rows)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_a_guardrail_dropping_one_item_from_a_long_tool_result_list_keeps_the_rest_without_stalling() -> None:
    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=_function_response_with_rows([1] * 8000 + ["zebrafruit"]),
        guardrail_to_apply=_RestructuringGuardrail(rewrite=_without_zebrafruit_rows),
    )

    assert _sent_request(guardrailed) == _function_response_with_rows([1] * 8000)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_a_guardrail_dropping_both_ends_of_a_long_tool_result_list_sends_the_rest_without_stalling() -> None:
    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=_function_response_with_rows(["zebrafruit", *[1] * 8000, "zebrafruit"]),
        guardrail_to_apply=_RestructuringGuardrail(rewrite=_without_zebrafruit_rows),
    )

    sent_rows: Final = guardrailed["contents"][0]["parts"][0]["functionResponse"]["response"]["rows"]
    assert [str(row) for row in sent_rows] == ["1"] * 8000


@pytest.mark.asyncio
async def test_a_non_ascii_blocked_word_in_a_function_response_is_blocked() -> None:
    accented_filter: Final = ContentFilterGuardrail(
        guardrail_name="accented-filter",
        blocked_words=[BlockedWord(keyword="crème brûlée", action=ContentFilterAction.BLOCK)],
    )

    with pytest.raises(HTTPException) as blocked:
        await GoogleGenAIGenerateContentHandler().process_input_messages(
            data=_function_response_with_rows(["order the crème brûlée"]),
            guardrail_to_apply=accented_filter,
        )

    assert blocked.value.status_code == 400


@pytest.mark.asyncio
async def test_a_tool_result_the_guardrail_leaves_alone_keeps_the_callers_object() -> None:
    response: Final = {"rows": [1, 2.5, None, "plums"]}

    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=_one_part_request("user", {"functionResponse": {"name": "lookup", "response": response}}),
        guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.MASK),
    )

    assert guardrailed["contents"][0]["parts"][0]["functionResponse"]["response"] is response


@pytest.mark.asyncio
async def test_a_guardrail_that_breaks_a_tool_result_json_blocks_the_request_instead_of_sending_it() -> None:
    request: Final = _function_response_with_rows(["plums"])
    original: Final = copy.deepcopy(request)

    with pytest.raises(UnappliableRequestRewrite) as rejected:
        await GoogleGenAIGenerateContentHandler().process_input_messages(
            data=request,
            guardrail_to_apply=_TruncatingGuardrail(),
        )

    assert rejected.value.guardrail_name == "truncator"
    assert request["contents"] == original["contents"]


@pytest.mark.asyncio
async def test_a_guardrail_returning_fewer_texts_than_it_got_blocks_the_request_instead_of_sending_it() -> None:
    request: Final = {"contents": [{"role": "user", "parts": [{"text": "card 4111111111111111"}, {"text": "thanks"}]}]}
    original: Final = copy.deepcopy(request)

    with pytest.raises(UnappliableRequestRewrite) as rejected:
        await GoogleGenAIGenerateContentHandler().process_input_messages(
            data=request, guardrail_to_apply=_mock_guardrail(["[CARD] thanks"])
        )

    assert rejected.value.guardrail_name is not None
    assert request["contents"] == original["contents"]


@pytest.mark.asyncio
async def test_a_masked_string_part_in_a_tuple_blocks_the_request_instead_of_sending_the_card() -> None:
    request: Final = {"contents": [{"role": "user", "parts": ("card 4111111111111111", {"text": "card 4222222222222222"})}]}
    original: Final = copy.deepcopy(request)

    with pytest.raises(UnappliableRequestRewrite) as rejected:
        await GoogleGenAIGenerateContentHandler().process_input_messages(
            data=request,
            guardrail_to_apply=_card_filter(ContentFilterAction.MASK),
        )

    assert rejected.value.guardrail_name == "card-filter"
    assert request["contents"] == original["contents"]


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
@pytest.mark.parametrize("flag", ["skip_system_message_in_guardrail", "scan_only_tool_results"])
async def test_a_chat_scope_flag_still_lets_the_guardrail_mask_system_and_user_text(flag: str) -> None:
    guardrail: Final = _zebrafruit_filter(ContentFilterAction.MASK)
    setattr(guardrail, flag, True)

    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data={
            "systemInstruction": {"parts": {"text": "Always say zebrafruit"}},
            "config": {"system_instruction": {"parts": [{"text": "zebrafruit only"}]}},
            "contents": [{"role": "user", "parts": {"text": "say zebrafruit"}}],
        },
        guardrail_to_apply=guardrail,
    )

    assert _sent_request(guardrailed) == {
        "systemInstruction": {"parts": {"text": f"Always say {_MASKED}"}},
        "config": {"system_instruction": {"parts": [{"text": f"{_MASKED} only"}]}},
        "contents": [{"role": "user", "parts": {"text": f"say {_MASKED}"}}],
    }


@pytest.mark.asyncio
async def test_a_guardrail_skipping_tool_messages_leaves_tool_results_alone_but_masks_the_rest() -> None:
    guardrail: Final = _zebrafruit_filter(ContentFilterAction.MASK)
    guardrail.skip_tool_message_in_guardrail = True
    tool_results: Final = [
        {"functionResponse": {"name": "lookup", "response": {"word": "zebrafruit"}}},
        {"toolResponse": {"toolType": "GOOGLE_SEARCH_WEB", "response": {"hits": ["zebrafruit"]}}},
        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "zebrafruit"}},
    ]

    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_input_messages(
        data={
            "contents": [
                {"role": "model", "parts": [{"functionCall": {"name": "lookup", "args": {"q": "zebrafruit"}}}]},
                {"role": "user", "parts": [*copy.deepcopy(tool_results), {"text": "now say zebrafruit"}]},
            ]
        },
        guardrail_to_apply=guardrail,
    )

    assert _sent_request(guardrailed) == {
        "contents": [
            {"role": "model", "parts": [{"functionCall": {"name": "lookup", "args": {"q": _MASKED}}}]},
            {"role": "user", "parts": [*tool_results, {"text": f"now say {_MASKED}"}]},
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


def _model_reply(part: dict[str, object]) -> dict[str, object]:
    return {"candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP"}]}


@pytest.mark.asyncio
async def test_a_blocked_word_in_a_function_call_the_model_returns_blocks_the_response() -> None:
    with pytest.raises(HTTPException) as blocked:
        await GoogleGenAIGenerateContentHandler().process_output_response(
            response=_model_reply({"functionCall": {"name": "lookup", "args": {"q": "zebrafruit"}}}),
            guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.BLOCK),
        )

    assert blocked.value.status_code == 400


@pytest.mark.asyncio
async def test_a_masked_word_in_a_function_call_the_model_returns_reaches_the_client_masked() -> None:
    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_output_response(
        response=_model_reply({"functionCall": {"name": "lookup", "args": {"q": "zebrafruit", "limit": 3}}}),
        guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.MASK),
    )

    assert guardrailed == _model_reply({"functionCall": {"name": "lookup", "args": {"q": _MASKED, "limit": 3}}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ordered",
    [lambda text, code: [text, code], lambda text, code: [code, text]],
    ids=["text_first", "code_first"],
)
async def test_a_guardrail_replacing_a_reply_with_one_text_replaces_its_first_text_as_on_chat(
    ordered: Callable[[dict[str, object], dict[str, object]], list[dict[str, object]]],
) -> None:
    code_part: Final = {"executableCode": {"language": "PYTHON", "code": "print(1)"}}
    reply: Final = {"candidates": [{"content": {"role": "model", "parts": ordered({"text": "zebrafruit"}, code_part)}}]}

    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_output_response(
        response=reply, guardrail_to_apply=_mock_guardrail(["Blocked by policy"])
    )

    assert guardrailed["candidates"][0]["content"]["parts"] == ordered({"text": "Blocked by policy"}, code_part)


class _LastTextOnlyGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(guardrail_name="last-text-only")

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if "zebrafruit" in (inputs.get("texts") or [""])[-1]:
            raise GuardrailBlockedError("blocked")
        return inputs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ordered",
    [lambda text, code: [text, code], lambda text, code: [code, text]],
    ids=["text_first", "code_first"],
)
async def test_a_guardrail_scanning_only_the_last_text_like_hiddenlayer_still_sees_a_reply_text_beside_code(
    ordered: Callable[[dict[str, object], dict[str, object]], list[dict[str, object]]],
) -> None:
    code_part: Final = {"executableCode": {"language": "PYTHON", "code": "print(1)"}}
    reply: Final = {"candidates": [{"content": {"role": "model", "parts": ordered({"text": "zebrafruit"}, code_part)}}]}

    with pytest.raises(GuardrailBlockedError):
        await GoogleGenAIGenerateContentHandler().process_output_response(
            response=reply, guardrail_to_apply=_LastTextOnlyGuardrail()
        )


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
async def test_a_blocked_word_split_between_a_streamed_text_part_and_a_code_part_does_not_block() -> None:
    chunks: Final = [
        {"candidates": [{"content": {"parts": [{"text": "zebra"}]}}]},
        {"candidates": [{"content": {"parts": [{"executableCode": {"language": "PYTHON", "code": "fruit = 1"}}]}}]},
    ]

    result: Final = await GoogleGenAIGenerateContentHandler().process_output_streaming_response(
        responses_so_far=chunks, guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.BLOCK)
    )

    assert result is chunks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "as_stream_item",
    [lambda reply: reply, lambda reply: f"data: {json.dumps(reply)}\r\n\r\n"],
    ids=["response_chunk", "raw_sse"],
)
async def test_a_blocked_word_whose_streamed_text_parts_sit_around_a_code_part_blocks_the_stream(
    as_stream_item: Callable[[dict[str, object]], object],
) -> None:
    streamed: Final = [
        as_stream_item(_model_reply({"text": "zebra"})),
        as_stream_item(_model_reply({"executableCode": {"language": "PYTHON", "code": "pass"}})),
        as_stream_item(_model_reply({"text": "fruit"})),
    ]

    with pytest.raises(HTTPException) as blocked:
        await GoogleGenAIGenerateContentHandler().process_output_streaming_response(
            responses_so_far=streamed,
            guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.BLOCK),
        )

    assert blocked.value.status_code == 400


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
@pytest.mark.parametrize(
    "as_stream_item",
    [lambda reply: reply, lambda reply: f"data: {json.dumps(reply)}\r\n\r\n"],
    ids=["response_chunk", "raw_sse"],
)
async def test_a_blocked_word_in_a_function_call_the_model_streams_blocks_the_stream(
    as_stream_item: Callable[[dict[str, object]], object],
) -> None:
    streamed: Final = [
        as_stream_item(_model_reply({"text": "Looking it up."})),
        as_stream_item(_model_reply({"functionCall": {"name": "lookup", "args": {"q": "zebrafruit"}}})),
    ]

    with pytest.raises(HTTPException) as blocked:
        await GoogleGenAIGenerateContentHandler().process_output_streaming_response(
            responses_so_far=streamed,
            guardrail_to_apply=_zebrafruit_filter(ContentFilterAction.BLOCK),
        )

    assert blocked.value.status_code == 400


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
@pytest.mark.parametrize("scan_tool_calls", [True, False], ids=["opted-in", "default"])
async def test_function_call_parts_reach_the_guardrail_as_tool_calls(
    scan: Callable[[_ToolCallGuardrail], object], scan_tool_calls: bool
) -> None:
    guardrail: Final = _ToolCallGuardrail(scan_tool_calls=scan_tool_calls)

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


_REQUEST_WITH_EVERY_PART_KIND: Final = {
    "systemInstruction": {"parts": {"text": "system words"}},
    "contents": [
        {"role": "user", "parts": {"text": "user words"}},
        {
            "role": "model",
            "parts": [
                {"executableCode": {"language": "PYTHON", "code": "print('model code')"}},
                {"toolCall": {"toolType": "GOOGLE_SEARCH_WEB", "args": {"query": "server call"}}},
                FUNCTION_CALL_PART,
            ],
        },
        {
            "role": "user",
            "parts": [
                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "code output"}},
                {"functionResponse": {"name": "send_payment", "response": {"status": "paid"}}},
                {"toolResponse": {"toolType": "GOOGLE_SEARCH_WEB", "response": {"hits": ["search hit"]}}},
            ],
        },
    ],
}
_MESSAGE_SCAN_TEXTS: Final = ["system words", "user words", "print('model code')"]
_TOOL_RESULT_SCAN_TEXTS: Final = ["code output", '{"status": "paid"}', '{"hits": ["search hit"]}']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scan_tool_calls", "scanned_texts", "scanned_tool_calls"),
    [
        pytest.param(False, [*_MESSAGE_SCAN_TEXTS, *_TOOL_RESULT_SCAN_TEXTS], [], id="tool-results-only"),
        pytest.param(
            True,
            [*_MESSAGE_SCAN_TEXTS, '{"query": "server call"}', *_TOOL_RESULT_SCAN_TEXTS],
            ["send_payment"],
            id="plus-tool-calls-as-text",
        ),
    ],
)
async def test_a_guardrail_scanning_only_tool_results_skips_tool_calls_but_still_sees_every_text(
    scan_tool_calls: bool, scanned_texts: list[str], scanned_tool_calls: list[str]
) -> None:
    guardrail: Final = _ToolCallGuardrail(scan_tool_calls=scan_tool_calls)
    guardrail.scan_only_tool_results = True

    await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=copy.deepcopy(_REQUEST_WITH_EVERY_PART_KIND), guardrail_to_apply=guardrail
    )

    (inputs,) = guardrail.seen
    assert inputs["texts"] == scanned_texts
    assert [tool_call["function"]["name"] for tool_call in inputs.get("tool_calls") or []] == scanned_tool_calls


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
async def test_function_call_rewrite_the_request_cannot_carry_is_rejected() -> None:
    data: Final = {
        "model": "gemini-2.5-flash",
        "contents": [{"role": "model", "parts": [copy.deepcopy(FUNCTION_CALL_PART)]}],
    }

    with pytest.raises(UnappliableRequestRewrite):
        await GoogleGenAIGenerateContentHandler().process_input_messages(
            data=data,
            guardrail_to_apply=_ToolCallGuardrail(
                rewrite=lambda tool_calls: [
                    {**call, "function": {**call["function"], "arguments": "[1]"}} for call in tool_calls
                ]
            ),
        )

    assert data["contents"][0]["parts"][0] == FUNCTION_CALL_PART


@pytest.mark.asyncio
async def test_a_request_tool_call_the_guardrail_drops_still_reaches_the_provider_as_on_chat() -> None:
    data: Final = {
        "model": "gemini-2.5-flash",
        "contents": [{"role": "model", "parts": [copy.deepcopy(FUNCTION_CALL_PART)]}],
    }

    await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=data, guardrail_to_apply=_ToolCallGuardrail(rewrite=lambda tool_calls: [])
    )

    assert data["contents"][0]["parts"][0] == FUNCTION_CALL_PART


@pytest.mark.asyncio
async def test_a_reply_tool_call_the_guardrail_drops_still_reaches_the_client_as_on_chat() -> None:
    reply: Final = _model_reply(copy.deepcopy(FUNCTION_CALL_PART))

    guardrailed: Final = await GoogleGenAIGenerateContentHandler().process_output_response(
        response=reply, guardrail_to_apply=_ToolCallGuardrail(rewrite=lambda tool_calls: [])
    )

    assert guardrailed == _model_reply(FUNCTION_CALL_PART)


SECOND_FUNCTION_CALL_PART: Final = {"functionCall": {"name": "search", "args": {"query": "invoices"}}}


def _first_call_masked_second_dropped(tool_calls: Sequence[ChatCompletionToolCallChunk]) -> list[object]:
    first: Final = tool_calls[0]
    return [{**first, "function": {**first["function"], "arguments": json.dumps({"card": "[MASKED]", "amount": 12.5})}}]


def _first_call_dropped(tool_calls: Sequence[ChatCompletionToolCallChunk]) -> list[object]:
    return list(tool_calls[1:])


TWO_FUNCTION_CALL_PARTS: Final = (FUNCTION_CALL_PART, SECOND_FUNCTION_CALL_PART)


async def _guardrail_two_request_calls(guardrail: _ToolCallGuardrail) -> None:
    await GoogleGenAIGenerateContentHandler().process_input_messages(
        data={"model": "gemini-2.5-flash", "contents": [{"role": "model", "parts": copy.deepcopy(TWO_FUNCTION_CALL_PARTS)}]},
        guardrail_to_apply=guardrail,
    )


async def _guardrail_two_reply_calls(guardrail: _ToolCallGuardrail) -> None:
    await GoogleGenAIGenerateContentHandler().process_output_response(
        response={"candidates": [{"content": {"role": "model", "parts": copy.deepcopy(TWO_FUNCTION_CALL_PARTS)}}]},
        guardrail_to_apply=guardrail,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("guardrail_two_calls", [_guardrail_two_request_calls, _guardrail_two_reply_calls])
@pytest.mark.parametrize("rewrite", [_first_call_dropped, _first_call_masked_second_dropped])
async def test_a_guardrail_dropping_one_of_several_tool_calls_is_rejected_instead_of_sending_a_wrong_call(
    guardrail_two_calls: Callable[[_ToolCallGuardrail], Awaitable[None]],
    rewrite: Callable[[Sequence[ChatCompletionToolCallChunk]], list[object]],
) -> None:
    with pytest.raises(UnappliableRequestRewrite):
        await guardrail_two_calls(_ToolCallGuardrail(rewrite=rewrite))



RENAMED_FUNCTION_CALL_PART: Final = {
    "functionCall": {"name": "review_payment", "args": {"card": "4111 1111", "amount": 12.5}}
}


def _renamed_to_review_payment(tool_calls: Sequence[ChatCompletionToolCallChunk]) -> list[object]:
    return [{**tool_call, "function": {**tool_call["function"], "name": "review_payment"}} for tool_call in tool_calls]


@pytest.mark.asyncio
async def test_a_tool_call_the_guardrail_renames_goes_out_under_the_new_name_as_on_chat() -> None:
    data: Final = {"model": "gemini-2.5-flash", "contents": [{"role": "model", "parts": [copy.deepcopy(FUNCTION_CALL_PART)]}]}

    await GoogleGenAIGenerateContentHandler().process_input_messages(
        data=data, guardrail_to_apply=_ToolCallGuardrail(rewrite=_renamed_to_review_payment)
    )
    guardrailed_reply: Final = await GoogleGenAIGenerateContentHandler().process_output_response(
        response=_model_reply(copy.deepcopy(FUNCTION_CALL_PART)),
        guardrail_to_apply=_ToolCallGuardrail(rewrite=_renamed_to_review_payment),
    )

    assert data["contents"][0]["parts"] == [RENAMED_FUNCTION_CALL_PART]
    assert guardrailed_reply == _model_reply(RENAMED_FUNCTION_CALL_PART)

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
        pytest.param(
            [{"json_path": "$.card", "stringValue": "4111"}],
            [{"json_path": "$.card", "string_value": "[CARD]"}],
            id="mixed-case-string",
        ),
        pytest.param(
            [
                {"jsonPath": "$.card", "stringValue": "4111 "},
                {"jsonPath": "$.note", "stringValue": "rent"},
                {"jsonPath": "$.card", "stringValue": "1111"},
            ],
            [
                {"jsonPath": "$.card", "stringValue": "[CARD] 1111"},
                {"jsonPath": "$.note", "stringValue": "rent"},
                {"jsonPath": "$.card", "stringValue": ""},
            ],
            id="interleaved-paths",
        ),
    ],
)
async def test_masked_partial_function_call_arg_is_written_back_as_the_only_value(
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
