"""Mistral chat-completion response JSON -> IR ``ChatResponse``.

v1's ``MistralConfig.transform_response`` (LIVE on the httpx path) applies
two raw-body pre-steps, then the shared ``convert_to_model_response_object``
(fresh ModelResponse, model None -> BARE wire model). v2 mirrors the same
chain: the two pre-steps here, then the shared openai parser (probed
in-process at HEAD):

- ``_handle_empty_content_response``: a message ``content == ""`` becomes
  None — and it runs FIRST, so a content LIST that later flattens to ""
  STAYS ``""`` (the blocks_no_text pin);
- ``_handle_content_list_to_str_conversion``: a content LIST is collapsed —
  thinking blocks' text items join with ``\\n`` into ``reasoning_content``
  (set only when truthy), and the LAST text block WINS as ``content``
  (v1 overwrites, never joins — the two_text_blocks pin); blocks of any
  other type are silently ignored.

Malformed list shapes are loud boundary errors naming v1's raise: non-dict
blocks and a non-list ``thinking`` value (v1's unguarded ``.get`` chains),
a non-str thinking ``text`` (v1's ``"\\n".join`` TypeError), and an EMPTY
content list (v1's truthiness gate skips the collapse and
``convert_to_model_response_object`` raises "Invalid response object" on
``Message(content=[])`` — verifier-wave2b-beta F3).
"""

from __future__ import annotations

from expression import Error, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import ChatRequest, ChatResponse, PlainJson
from ..openai_compat.response import parse_response as openai_parse_response

_ParseResult = Result[ChatResponse, TranslationError]


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(
        BoundaryError.of(Block.of_seq([f"mistral {reason} (v1 raises)"]))
    )


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    pre_stepped = _with_mistral_pre_steps(raw)
    if isinstance(pre_stepped, TranslationError):
        return Error(pre_stepped)
    return openai_parse_response(pre_stepped, request)


def _with_mistral_pre_steps(raw: PlainJson) -> PlainJson | TranslationError:
    if not isinstance(raw, dict):
        return raw  # the openai parser owns the non-object boundary error
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return raw
    reshaped: list[PlainJson] = []
    for choice in choices:
        stepped = _pre_step_choice(choice)
        if isinstance(stepped, TranslationError):
            return stepped
        reshaped = [*reshaped, stepped]
    return {**raw, "choices": reshaped}


def _pre_step_choice(choice: PlainJson) -> PlainJson | TranslationError:
    if not isinstance(choice, dict):
        return choice
    message = choice.get("message")
    if not isinstance(message, dict):
        return choice
    content = message.get("content")
    if content == "":
        return {**choice, "message": {**message, "content": None}}
    if not isinstance(content, list):
        return choice
    if not content:
        # v1's truthiness gate SKIPS the collapse on an empty list and the
        # un-collapsed Message(content=[]) raises out of
        # convert_to_model_response_object ("Invalid response object") —
        # the old isinstance-only arm collapsed [] to "" and SERVED it.
        return _boundary("response content list is empty")
    collapsed = _collapse_content_list(content)
    if isinstance(collapsed, TranslationError):
        return collapsed
    text, reasoning = collapsed
    stepped: dict[str, PlainJson] = {**message, "content": text}
    if reasoning:
        stepped = {**stepped, "reasoning_content": reasoning}
    return {**choice, "message": stepped}


def _collapse_content_list(
    content: list[PlainJson],
) -> tuple[str, str] | TranslationError:
    thinking_content = ""
    text_content = ""
    for block in content:
        if not isinstance(block, dict):
            return _boundary("response content block is not an object")
        block_type = block.get("type")
        if block_type == "thinking":
            collapsed = _collapse_thinking(block)
            if isinstance(collapsed, TranslationError):
                return collapsed
            thinking_content = collapsed
        elif block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                return _boundary("response text block is not a string")
            text_content = text  # v1 overwrites: the LAST text block wins
    return text_content, thinking_content


def _collapse_thinking(block: dict[str, PlainJson]) -> str | TranslationError:
    thinking = block.get("thinking", [])
    if not isinstance(thinking, list):
        return _boundary("response thinking value is not a list")
    texts: list[str] = []
    for item in thinking:
        if not isinstance(item, dict):
            return _boundary("response thinking item is not an object")
        if item.get("type") == "text":
            text = item.get("text", "")
            if not isinstance(text, str):
                # the old arm coerced to "" and SERVED; v1's "\n".join
                # raises TypeError out of the transform
                return _boundary("response thinking text is not a string")
            texts = [*texts, text]
    return "\n".join(texts)
