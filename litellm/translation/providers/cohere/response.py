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
AttributeErrors on ``.get``), the hostile citation shapes (truthy non-list
citations/sources, non-dict citation/source/document entries — v1's
unguarded ``.get`` chain AttributeErrors on each), and a tool_call without
a ``function`` key (v1's Message constructor TypeErrors). Citation VALUE
types the typed construction rejects (non-int start/end, non-str
title/url) fall back instead — v1 attaches the unvalidated TypedDict via
plain setattr and SERVES it, bytes v2's pydantic seam cannot reproduce.
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
    if isinstance(annotations, TranslationError):
        return Error(annotations)
    tool_calls = _indexed_tool_calls(message.get("tool_calls"))
    if isinstance(tool_calls, TranslationError):
        return Error(tool_calls)
    counts = _usage_counts(usage)
    if isinstance(counts, TranslationError):
        return Error(counts)
    prompt, completion = counts
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


def _usage_counts(usage: dict[str, PlainJson]) -> tuple[int, int] | TranslationError:
    tokens_raw = usage.get("tokens", {})
    if not isinstance(tokens_raw, dict):
        return _boundary(
            "cohere v2 usage.tokens is present but not an object (v1 "
            "AttributeErrors on .get)"
        )
    prompt = _int_token(tokens_raw, "input_tokens")
    if isinstance(prompt, TranslationError):
        return prompt
    completion = _int_token(tokens_raw, "output_tokens")
    if isinstance(completion, TranslationError):
        return completion
    return prompt, completion


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
        if "function" not in tool:
            return _boundary(
                "cohere v2 tool_call has no 'function' key (v1's Message "
                "constructor raises TypeError; typed here so the raise "
                "never escapes the seam untyped)"
            )
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


def _annotations(citations: PlainJson) -> list[PlainJson] | None | TranslationError:
    """Mirror ``_translate_citations_to_openai_annotations`` FAIL-CLOSED (one
    annotation per document source; non-document and document-less sources
    skipped; sources-less citations skipped). Falsy citations (None/[]/"")
    yield None — v1 only attaches annotations when the value is truthy. Every
    shape v1's unguarded ``.get`` chain AttributeErrors on is a typed
    boundary error here (truthy non-list citations, non-dict citation entry,
    truthy non-list sources, non-dict source, present non-dict document
    under type "document"), and annotation VALUE types the seam's typed
    Message construction validates (start/end int, title/url str) fall back
    where v1 serves the unvalidated TypedDict via plain setattr."""
    if not citations:
        return None
    if not isinstance(citations, list):
        return _boundary(
            "cohere v2 message.citations is truthy but not a list (v1 "
            "iterates it and AttributeErrors on citation.get)"
        )
    annotations: list[PlainJson] = []
    for citation in citations:
        if not isinstance(citation, dict):
            return _boundary(
                "cohere v2 citation entry is not an object (v1 "
                "AttributeErrors on citation.get)"
            )
        sources = citation.get("sources", [])
        if not sources:
            continue  # v1's falsy gate: the citation is skipped, served
        if not isinstance(sources, list):
            return _boundary(
                "cohere v2 citation.sources is truthy but not a list (v1 "
                "iterates it and AttributeErrors on source.get)"
            )
        for source in sources:
            annotation = _source_annotation(citation, source)
            if isinstance(annotation, TranslationError):
                return annotation
            if annotation is not None:
                annotations = [*annotations, annotation]
    return annotations or None


def _source_annotation(
    citation: dict[str, PlainJson], source: PlainJson
) -> PlainJson | None | TranslationError:
    if not isinstance(source, dict):
        return _boundary(
            "cohere v2 citation source is not an object (v1 AttributeErrors "
            "on source.get)"
        )
    if source.get("type") != "document" or "document" not in source:
        return None  # v1 skips non-document and document-less sources, served
    document = source["document"]
    if not isinstance(document, dict):
        return _boundary(
            "cohere v2 citation document is not an object (v1 "
            "AttributeErrors on document.get)"
        )
    start = citation.get("start", 0)
    end = citation.get("end", 0)
    title = document.get("title", "")
    url = source.get("url") or f"source:{source.get('id', 'unknown')}"
    if not _is_int(start) or not _is_int(end) or not isinstance(title, str):
        return _boundary(
            "cohere v2 citation start/end/title value type fails the typed "
            "annotation construction (v1 SERVES the unvalidated TypedDict "
            "via plain setattr; v2's Message validation would raise)"
        )
    if not isinstance(url, str):
        return _boundary(
            "cohere v2 citation source url is truthy but not a string (v1 "
            "SERVES the unvalidated TypedDict via plain setattr; v2's "
            "Message validation would raise)"
        )
    return {
        "type": "url_citation",
        "url_citation": {
            "start_index": start,
            "end_index": end,
            "title": title,
            "url": url,
        },
    }


def _is_int(value: PlainJson) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _semantic_blocks(
    text: str | None, tool_calls: list[dict[str, PlainJson]] | None
) -> list[ContentBlock] | TranslationError:
    """The IR blocks are deliberately LENIENT where the wire body beside them
    is strict (critic-wave2b-beta N3): v1 SERVES non-str tool id/name and
    unparseable argument strings VERBATIM on the wire (probed — only a
    missing ``function`` key raises, gated in ``_indexed_tool_calls``), so
    failing closed here would fall back on shapes v1 serves. Nothing reads
    ``ChatResponse.content`` on the openai construction arm today; a future
    consumer must re-derive these defaults against the wire truth."""
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
