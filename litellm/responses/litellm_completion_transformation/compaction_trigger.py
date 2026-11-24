"""
A Responses ``compaction_trigger`` asks for exactly one ``compaction`` output item. Chat-completions providers have
no such item, so the bridge asks the model for a summary in the same call and returns it as the summary-only
compaction item it already replays on later turns.
"""

from collections.abc import Mapping
from typing import Final

from openai.types.responses.easy_input_message_param import EasyInputMessageParam

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.anthropic.messages.context_management.constants import (
    COMPACT_DEFAULT_INSTRUCTIONS,
    COMPACT_NO_TOOL_CALLS_SUFFIX,
    text_in_summary_tags,
)
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams
from litellm.types.utils import Choices, Message, ModelResponse, ModelResponseStream

from .compaction_items import input_items, summary_only_compaction_block

COMPACTION_TRIGGER_TYPE: Final = "compaction_trigger"
SUMMARY_PROMPT: Final = COMPACT_DEFAULT_INSTRUCTIONS + COMPACT_NO_TOOL_CALLS_SUFFIX


def _is_compaction_trigger(item: object) -> bool:
    fields: Final = as_str_mapping(item)
    return fields is not None and fields.get("type") == COMPACTION_TRIGGER_TYPE


def ends_in_compaction_trigger(input: object) -> bool:
    items: Final = input_items(input)
    return bool(items) and _is_compaction_trigger(items[-1])


def with_trailing_compaction_trigger_as_summary_prompt(
    input: str | ResponseInputParam,
) -> str | ResponseInputParam:
    if isinstance(input, str) or not ends_in_compaction_trigger(input):
        return input
    return [*input[:-1], EasyInputMessageParam(role="user", type="message", content=SUMMARY_PROMPT)]


def without_a_forced_tool(request: ResponsesAPIOptionalRequestParams) -> ResponsesAPIOptionalRequestParams:
    """A forced tool choice would make the model call a tool instead of writing the summary"""
    tool_choice: Final[object] = request.get(  # pyright: ignore[reportUnknownMemberType]  # ToolChoice has an untyped member
        "tool_choice"
    )
    if tool_choice is None or tool_choice in ("auto", "none"):
        return request
    return {**request, "tool_choice": "auto"}


def summary_from_reply(text: str | None) -> str | None:
    """The text inside the reply's ``<summary>`` block, or the whole reply when the model left the tags out"""
    if not text:
        return None
    tagged: Final = text_in_summary_tags(text)
    summary: Final = (text if tagged is None else tagged).strip()
    return summary or None


def _no_summary_error(model: str | None, llm_provider: str | None) -> litellm.APIError:
    return litellm.APIError(
        status_code=502,
        message=f"{model} returned no summary text for the compaction_trigger request",
        llm_provider=llm_provider or "",
        model=model or "",
    )


def _summary_or_error(text: str | None, model: str | None, llm_provider: str | None) -> str:
    summary: Final = summary_from_reply(text)
    if summary is None:
        raise _no_summary_error(model, llm_provider)
    return summary


def _summary_only_message(message: Message, summary: str) -> Message:
    return message.model_copy(
        update={
            "content": None,
            "tool_calls": None,
            "function_call": None,
            "provider_specific_fields": {
                **(message.provider_specific_fields or {}),
                "compaction_blocks": [dict(summary_only_compaction_block(summary))],
            },
        }
    )


def _compaction_reply(response: ModelResponse, text: str | None, llm_provider: str | None) -> ModelResponse:
    first: Final = response.choices[0] if response.choices else None
    if not isinstance(first, Choices):
        raise _no_summary_error(response.model, llm_provider)
    summary: Final = _summary_or_error(text, response.model, llm_provider)
    return response.model_copy(
        update={
            "choices": [
                first.model_copy(update={"message": _summary_only_message(first.message, summary)}),
                *response.choices[1:],
            ]
        }
    )


def as_compaction_reply(response: ModelResponse, *, llm_provider: str | None) -> ModelResponse:
    """
    The model's summary turn as a reply that carries only the summary, in the ``compaction_blocks`` provider field
    the bridge turns into one ``compaction`` output item. Text and tool calls are dropped, so no message or
    function call item comes back next to it. Usage, the id and reasoning stay as the model sent them
    """
    first: Final = response.choices[0] if response.choices else None
    text: Final = first.message.content if isinstance(first, Choices) else None
    return _compaction_reply(response, text, llm_provider)


class CompactionSummaryStream:
    """
    Collects the summary a streamed summary turn writes. ``take`` keeps text and tool call deltas out of the
    stream, so no message or function call item is opened, and ``reply`` puts the collected summary on the
    assembled reply the way ``as_compaction_reply`` does
    """

    def __init__(self, *, llm_provider: str | None) -> None:
        self._llm_provider: Final = llm_provider
        self._text: str = ""

    def take(self, chunk: ModelResponseStream) -> ModelResponseStream:
        if not chunk.choices:
            return chunk
        first: Final = chunk.choices[0]
        delta: Final = first.delta
        if isinstance(delta.content, str):
            self._text += delta.content
        if not (delta.content or delta.tool_calls or delta.function_call):
            return chunk
        answerless: Final = delta.model_copy(update={"content": None, "tool_calls": None, "function_call": None})
        return chunk.model_copy(
            update={"choices": [first.model_copy(update={"delta": answerless}), *chunk.choices[1:]]}
        )

    def provider_specific_fields(self, model: str | None) -> Mapping[str, object]:
        summary: Final = _summary_or_error(self._text, model, self._llm_provider)
        return {"compaction_blocks": [dict(summary_only_compaction_block(summary))]}

    def reply(self, response: ModelResponse) -> ModelResponse:
        return _compaction_reply(response, self._text, self._llm_provider)
