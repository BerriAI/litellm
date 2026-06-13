"""databricks response JSON -> IR ``ChatResponse``.

Mirrors ``DatabricksConfig.transform_response`` + ``_transform_dbrx_choices``
(probed in-process at HEAD). v1 parses a ``DatabricksResponse`` TypedDict onto
a fresh ``ModelResponse``; the normalized chat-completion body rides
``ChatResponse.wire`` and the seam's ``openai`` construction arm reproduces
v1's assembly (the cohere/ollama_chat shape — the body carries the wire's
own id/created, so the ambient envelope is overwritten exactly like v1):

- ``model`` = ``databricks/{wire model}`` (the prefix is v1's
  ``litellm_params.custom_llm_provider or "databricks"`` — DB-R7: a custom
  provider riding the databricks config keeps its OWN prefix; the dispatch
  provider here is "databricks", so the literal is correct, and the override
  is a documented fork obligation; the wire model is ``response.model or ""``);
- ``id``/``created`` copied verbatim (a missing ``id``/``created`` KeyErrors in
  v1 — a drift from researcher-5's "DatabricksException" claim: the
  ``DatabricksResponse(**json)`` TypedDict construction does NOT validate, so a
  malformed body raises a RAW ``KeyError`` on the first missing required key
  — pinned as a v1-raise fallback row);
- ``usage`` = ``Usage(**usage)``;
- per choice, ``_transform_dbrx_choices``: a content block-list is flattened to
  a joined text string (``extract_content_str``), ``reasoning``/``summary``
  blocks become ``reasoning_content`` (concatenated) + ``thinking_blocks``
  (missing signature -> ``""``), citations become
  ``provider_specific_fields.citations`` with a ``supported_text`` mirror of
  the block's text, tool_calls ride verbatim, logprobs is forced None;
- UNKNOWN top-level keys are DROPPED (only id/created/model/usage/choices come
  from the wire; ``object``/``system_fingerprint`` are ModelResponse defaults,
  the wire values are NOT carried — probed).

The json_mode single-tool -> content rewrite is NOT reached from v2: it fires
only when ``json_mode=True``, and v2 falls back on every response_format-on-
claude request (the json_tool_call machinery is unported), so json_mode is
always absent at parse time. The arm is a request-side fallback, pinned there.
"""

from __future__ import annotations

from typing import NamedTuple

from expression import Error, Nothing, Ok, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import (
    ChatRequest,
    ChatResponse,
    ContentBlock,
    FinishReason,
    JsonBlob,
    PlainJson,
    ResponseUsage,
    Text,
    ToolUse,
)

_ParseResult = Result[ChatResponse, TranslationError]

_IR_FINISH: frozenset[str] = frozenset(
    {"stop", "length", "tool_calls", "content_filter"}
)


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    if not isinstance(raw, dict):
        return Error(_boundary("databricks response is not an object (v1 raises)"))
    for key in ("id", "created", "usage", "choices"):
        if key not in raw:
            return Error(
                _boundary(
                    f"databricks response missing required key {key!r}: v1's "
                    "DatabricksResponse access raises a raw KeyError (the "
                    "TypedDict construction does not validate — probed)"
                )
            )
    usage = _usage(raw["usage"])
    if isinstance(usage, TranslationError):
        return Error(usage)
    wire_model = raw.get("model") or ""
    if not isinstance(wire_model, str):
        return Error(_boundary("databricks response model is not a string (v1 raises)"))
    choices = _choices(raw["choices"])
    if isinstance(choices, TranslationError):
        return Error(choices)
    model = f"databricks/{wire_model}"
    body: dict[str, PlainJson] = {
        "id": raw["id"],
        "created": raw["created"],
        "model": model,
        "object": "chat.completion",
        "choices": [choice.wire for choice in choices],
        "usage": usage.wire,
    }
    return Ok(
        ChatResponse(
            id=raw["id"] if isinstance(raw["id"], str) else "",
            model=model,
            content=_first_blocks(choices),
            finish=_first_finish(choices),
            usage=usage.ir,
            synthesized_json_content=False,
            wire=Some(JsonBlob(value=body)),
        )
    )


class _Usage(NamedTuple):
    wire: PlainJson
    ir: ResponseUsage


def _usage(raw: PlainJson) -> _Usage | TranslationError:
    if not isinstance(raw, dict):
        return _boundary(
            "databricks usage is not an object (v1's Usage(**usage) raises)"
        )
    prompt = _int(raw.get("prompt_tokens"))
    completion = _int(raw.get("completion_tokens"))
    total = _int(raw.get("total_tokens"))
    if prompt is None or completion is None:
        return _boundary(
            "databricks usage missing integer prompt_tokens/completion_tokens "
            "(v1's Usage validation raises)"
        )
    total_value = total if total is not None else prompt + completion
    return _Usage(
        wire=dict(raw),
        ir=ResponseUsage(
            input_tokens=prompt,
            output_tokens=completion,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            cache_creation=Nothing,
            total_tokens=Some(total_value),
        ),
    )


def _int(value: PlainJson) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


class _Choice(NamedTuple):
    wire: PlainJson
    blocks: Block[ContentBlock]
    finish: FinishReason


def _choices(raw: PlainJson) -> list[_Choice] | TranslationError:
    if not isinstance(raw, list):
        return _boundary("databricks choices is not a list (v1 raises)")
    out: list[_Choice] = []
    for entry in raw:
        choice = _choice(entry)
        if isinstance(choice, TranslationError):
            return choice
        out = [*out, choice]
    return out


def _choice(raw: PlainJson) -> _Choice | TranslationError:
    if not isinstance(raw, dict):
        return _boundary("databricks choice is not an object (v1 raises)")
    if "message" not in raw or "index" not in raw or "finish_reason" not in raw:
        return _boundary(
            "databricks choice missing index/message/finish_reason (v1's "
            "TypedDict access raises a raw KeyError)"
        )
    message = raw["message"]
    if not isinstance(message, dict):
        return _boundary("databricks choice message is not an object (v1 raises)")
    rebuilt = _message(message)
    if isinstance(rebuilt, TranslationError):
        return rebuilt
    finish_raw = raw["finish_reason"]
    wire: dict[str, PlainJson] = {
        "index": raw["index"],
        "finish_reason": finish_raw,
        "message": rebuilt,
        "logprobs": None,
    }
    return _Choice(wire=wire, blocks=_blocks(rebuilt), finish=_ir_finish(finish_raw))


def _message(message: dict[str, PlainJson]) -> dict[str, PlainJson] | TranslationError:
    content = message.get("content")
    content_str = _content_str(content)
    if isinstance(content_str, TranslationError):
        return content_str
    out: dict[str, PlainJson] = {
        "role": "assistant",
        "content": content_str,
        "tool_calls": message.get("tool_calls"),
    }
    reasoning, thinking_blocks = _reasoning(content)
    if reasoning is not None:
        out = {
            **out,
            "reasoning_content": reasoning,
            "thinking_blocks": thinking_blocks,
        }
    citations = _citations(content)
    if citations is not None:
        out = {**out, "provider_specific_fields": {"citations": citations}}
    return out


def _content_str(content: PlainJson) -> PlainJson | TranslationError:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                parts = [*parts, str(text) if text is not None else ""]
        return "".join(parts)
    return _boundary(
        "databricks message content is neither null/str/list (v1's "
        "extract_content_str raises Exception)"
    )


def _reasoning(content: PlainJson) -> tuple[str | None, list[PlainJson] | None]:
    if not isinstance(content, list):
        return None, None
    reasoning: str | None = None
    blocks: list[PlainJson] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if not isinstance(summary, list):
            continue
        for entry in summary:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text", "")
            reasoning = (reasoning or "") + (text if isinstance(text, str) else "")
            blocks = [
                *blocks,
                {
                    "type": "thinking",
                    "thinking": text if text is not None else "",
                    "signature": entry.get("signature", ""),
                },
            ]
    return reasoning, (blocks or None)


def _citations(content: PlainJson) -> list[PlainJson] | None:
    if not isinstance(content, list):
        return None
    citations: list[PlainJson] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        item_citations = item.get("citations")
        if item_citations and isinstance(item_citations, list):
            citations = [
                *citations,
                [
                    {**citation, "supported_text": text}
                    for citation in item_citations
                    if isinstance(citation, dict)
                ],
            ]
    return citations or None


def _ir_finish(finish: PlainJson) -> FinishReason:
    if finish == "length":
        return "length"
    if finish == "tool_calls":
        return "tool_calls"
    if finish == "content_filter":
        return "content_filter"
    return "stop"


def _blocks(message: dict[str, PlainJson]) -> Block[ContentBlock]:
    """Lenient IR content beside the strict wire body (the cohere/ollama N3
    rule): nothing reads ``ChatResponse.content`` on the openai construction
    arm; the wire body is the parity surface."""
    out: list[ContentBlock] = []
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    if isinstance(content, str) and content and not tool_calls:
        out = [ContentBlock.of_text(Text(text=content, cache=Nothing))]
    for call in tool_calls if isinstance(tool_calls, list) else []:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        function_map = function if isinstance(function, dict) else {}
        identifier = call.get("id")
        name = function_map.get("name")
        arguments = function_map.get("arguments")
        out = [
            *out,
            ContentBlock.of_tool_use(
                ToolUse(
                    id=identifier if isinstance(identifier, str) else "",
                    name=name if isinstance(name, str) else "",
                    arguments=JsonBlob(
                        value=arguments if isinstance(arguments, str) else ""
                    ),
                    cache=Nothing,
                )
            ),
        ]
    return Block.of_seq(out)


def _first_blocks(choices: list[_Choice]) -> Block[ContentBlock]:
    return choices[0].blocks if choices else Block.empty()


def _first_finish(choices: list[_Choice]) -> FinishReason:
    return choices[0].finish if choices else "stop"
