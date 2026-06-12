"""Cohere v2 ``/v2/chat`` response JSON -> IR ``ChatResponse``.

Mirrors ``CohereV2ChatConfig.transform_response`` (LIVE on the httpx path),
which mutates a FRESH ``ModelResponse``. The normalized chat-completion body
rides ``ChatResponse.wire`` and the seam's ``openai`` construction arm
reproduces v1's assembly byte-for-byte (probed in-process at HEAD; the wire
body deliberately carries NO ``id``/``created`` so the ambient envelope
stays litellm's — v1 ignores the cohere response id and stamps
``int(time.time())``):

- content: ``"".join(text of message.content items)`` when the key is
  present and non-null (an EMPTY list joins to ``""``, not None); ``None``
  when absent/null.
- ``finish_reason`` is ALWAYS ``"stop"`` — v1 never reads the wire
  ``finish_reason``; the fresh Choices default survives even for tool-call
  responses (quirk pinned by the differential rows).
- tool calls: each wire entry rides verbatim plus an ``index``; text content
  is DISCARDED when tool_calls are present (v1 replaces the whole message
  with ``Message(tool_calls=..., content=None, annotations=...)`` — the
  explicit ``annotations`` kwarg, ``None`` included, is reproduced).
- citations -> OpenAI ``url_citation`` annotations (one per document
  source; ``url`` falls back to ``source:{id}``).
- usage from ``usage.tokens.input_tokens/output_tokens`` (defaults 0; NOT
  billed_units), ``total = prompt + completion``.

Fail-closed arms reproduce v1's raises: a non-dict body (v1's TypedDict
splat raises -> CohereError 422), a missing ``message`` or ``usage`` key
(v1 KeyErrors out of the transform), non-dict content items (v1
AttributeErrors on ``.get``).
"""

from __future__ import annotations

import json

from expression import Error, Nothing, Ok, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import (
    ChatRequest,
    ChatResponse,
    ContentBlock,
    JsonBlob,
    PlainJson,
    ResponseUsage,
    Text,
    ToolUse,
)

_ParseResult = Result[ChatResponse, TranslationError]


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    if not isinstance(raw, dict):
        return Error(
            _boundary("cohere v2 response is not an object (v1 raises CohereError 422)")
        )
    message = raw.get("message")
    if "message" not in raw or not isinstance(message, dict):
        return Error(
            _boundary("cohere v2 response has no 'message' object (v1 KeyErrors)")
        )
    usage = raw.get("usage")
    if "usage" not in raw or not isinstance(usage, dict):
        return Error(
            _boundary("cohere v2 response has no 'usage' object (v1 KeyErrors)")
        )
    text = _joined_text(message.get("content"))
    if isinstance(text, TranslationError):
        return Error(text)
    annotations = _annotations(message.get("citations"))
    tool_calls = _indexed_tool_calls(message.get("tool_calls"))
    if isinstance(tool_calls, TranslationError):
        return Error(tool_calls)
    tokens_raw = usage.get("tokens", {})
    if not isinstance(tokens_raw, dict):
        return Error(
            _boundary(
                "cohere v2 usage.tokens is present but not an object (v1 "
                "AttributeErrors on .get)"
            )
        )
    prompt = _int_token(tokens_raw, "input_tokens")
    if isinstance(prompt, TranslationError):
        return Error(prompt)
    completion = _int_token(tokens_raw, "output_tokens")
    if isinstance(completion, TranslationError):
        return Error(completion)
    wire_message = _wire_message(text, tool_calls, annotations)
    body: dict[str, PlainJson] = {
        "object": "chat.completion",
        "model": request.model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": wire_message}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }
    blocks = _semantic_blocks(text, tool_calls)
    if isinstance(blocks, TranslationError):
        return Error(blocks)
    return Ok(
        ChatResponse(
            id="",  # v1 keeps the ambient chatcmpl id; the cohere wire id is ignored
            model=request.model,
            content=Block.of_seq(blocks),
            finish="stop",  # v1 never reads the wire finish_reason (fresh-Choices default)
            usage=ResponseUsage(
                input_tokens=prompt,
                output_tokens=completion,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                cache_creation=Nothing,
                total_tokens=Some(prompt + completion),
            ),
            synthesized_json_content=False,
            wire=Some(JsonBlob(value=body)),
        )
    )


def _joined_text(content: PlainJson) -> str | None | TranslationError:
    if content is None:
        return None
    if not isinstance(content, list):
        return _boundary("cohere v2 message.content is not a list (v1 raises)")
    parts: list[str] = []
    for item in content:
        if item is None:
            continue  # v1 filters None items
        if not isinstance(item, dict):
            return _boundary(
                "cohere v2 content item is not an object (v1 AttributeErrors)"
            )
        part = item.get("text", "")
        if not isinstance(part, str):
            return _boundary(
                "cohere v2 content item text is not a string (v1's "
                "str.join raises TypeError)"
            )
        parts = [*parts, part]
    return "".join(parts)


def _int_token(tokens: dict[str, PlainJson], key: str) -> int | TranslationError:
    """v1 reads ``usage.tokens.get(key, 0)`` and feeds the value into
    ``prompt + completion`` and ``Usage(...)``: an ABSENT key is 0; ints and
    integral floats serve (pydantic's lax coercion); anything else raises in
    v1 (a present null/str hits the ``+`` TypeError or the Usage validation
    — incl. the str+str concatenation corner, which v2 deliberately leaves
    to v1) — typed boundary error here."""
    if key not in tokens:
        return 0
    value = tokens.get(key)
    if isinstance(value, (bool, int)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return _boundary(
        f"cohere v2 usage.tokens.{key} is not an integer: {value!r} (v1 raises)"
    )


def _indexed_tool_calls(
    tool_calls: PlainJson,
) -> list[dict[str, PlainJson]] | None | TranslationError:
    if tool_calls is None or tool_calls == []:
        return None
    if not isinstance(tool_calls, list):
        return _boundary("cohere v2 message.tool_calls is not a list (v1 raises)")
    indexed: list[dict[str, PlainJson]] = []
    for index, tool in enumerate(tool_calls):
        if not isinstance(tool, dict):
            return _boundary("cohere v2 tool_call is not an object (v1 raises)")
        indexed = [*indexed, {**tool, "index": index}]
    return indexed


def _wire_message(
    text: str | None,
    tool_calls: list[dict[str, PlainJson]] | None,
    annotations: list[PlainJson] | None,
) -> dict[str, PlainJson]:
    if tool_calls is not None:
        # v1 REPLACES the message: Message(tool_calls=..., content=None,
        # annotations=...) — text content is lost, annotations explicit.
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": list(tool_calls),
            "annotations": annotations,
        }
    message: dict[str, PlainJson] = {"role": "assistant", "content": text}
    if annotations:
        message = {**message, "annotations": annotations}
    return message


def _annotations(citations: PlainJson) -> list[PlainJson] | None:
    """Mirror ``_translate_citations_to_openai_annotations`` (one annotation
    per document source; non-document sources skipped; sources-less
    citations skipped). Falsy citations (None/[]) yield None — v1 only
    attaches annotations when the citations list is truthy."""
    if not isinstance(citations, list) or not citations:
        return None
    annotations: list[PlainJson] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        sources = citation.get("sources")
        if not isinstance(sources, list) or not sources:
            continue
        for source in sources:
            if not isinstance(source, dict):
                continue
            document = source.get("document")
            if source.get("type") != "document" or not isinstance(document, dict):
                continue
            url = source.get("url") or f"source:{source.get('id', 'unknown')}"
            annotations = [
                *annotations,
                {
                    "type": "url_citation",
                    "url_citation": {
                        "start_index": citation.get("start", 0),
                        "end_index": citation.get("end", 0),
                        "title": document.get("title", ""),
                        "url": url,
                    },
                },
            ]
    return annotations or None


def _semantic_blocks(
    text: str | None, tool_calls: list[dict[str, PlainJson]] | None
) -> list[ContentBlock] | TranslationError:
    blocks: list[ContentBlock] = []
    if tool_calls is None and isinstance(text, str) and text:
        blocks = [ContentBlock.of_text(Text(text=text, cache=Nothing))]
    for call in tool_calls or []:
        function_raw = call.get("function")
        function = function_raw if isinstance(function_raw, dict) else {}
        name = function.get("name")
        identifier = call.get("id")
        arguments = function.get("arguments")
        raw_args = arguments if isinstance(arguments, str) else ""
        try:
            parsed: PlainJson = json.loads(raw_args) if raw_args else {}
        except ValueError:
            parsed = {}
        blocks = [
            *blocks,
            ContentBlock.of_tool_use(
                ToolUse(
                    id=identifier if isinstance(identifier, str) else "",
                    name=name if isinstance(name, str) else "",
                    arguments=JsonBlob(value=parsed),
                    cache=Nothing,
                    arguments_raw=Some(raw_args) if raw_args else Nothing,
                )
            ),
        ]
    return blocks
