"""Ollama ``/api/chat`` response JSON -> IR ``ChatResponse``.

Mirrors ``OllamaChatConfig.transform_response`` (probed in-process at HEAD),
which mutates a fresh ``ModelResponse``: the normalized chat-completion body
rides ``ChatResponse.wire`` and the seam's ``openai`` construction arm
reproduces v1's assembly byte-for-byte (the cohere shape — the body carries
NO id/created, so the ambient envelope stays litellm's; v1 re-stamps
``created = int(time.time())``, the same frozen ambient value):

- ``model`` = ``ollama_chat/{REQUEST model}`` (the wire ``model`` key is
  IGNORED — quirk-pinned);
- finish from ``map_finish_reason(done_reason or "stop")`` via the
  in-package mirror of v1's table (unknown values -> "stop"; the
  ``function_call`` passthrough value is IR-unrepresentable -> typed
  fallback, v1 serves it);
- message rides VERBATIM into the body (extra keys included — v1's
  ``Message(**message)`` forwards them) after the remap: a ``thinking``
  key (any value) becomes ``reasoning_content``; else a STRING content is
  think-tag split (``reasoning_content`` = inner text, content = the
  REMAINDER — note the request-side munge keeps the full string instead);
  v1 then sets ``reasoning_content: None`` when the regex misses, mirrored
  verbatim so both Message constructions see identical bytes;
- tool_calls present (truthy) force finish "tool_calls"; the entries ride
  verbatim — the seam's ``Message(**...)`` mints missing ids (bare uuid4)
  and re-dumps dict arguments with ``", "``/``": "`` separators exactly
  like v1's, because BOTH sides run the same validation;
- usage = ``Usage(prompt_eval_count, eval_count, sum)``; a body missing
  EITHER count falls back typed — v1 fills the gap with local
  ``litellm.token_counter`` estimates (deterministic, but a v1-stack call;
  deps hook deferred per the wave-3 dossier).

Fail-closed arms reproduce v1's raises: non-object body, missing/null/
non-object ``message`` (TypeError), a final content that is not a string —
``None``/missing included: v1's eval-count default EAGERLY evaluates
``token_counter(text=content)`` even when ``eval_count`` is present, so a
null content raises ValueError REGARDLESS of counts (probed) — non-str
truthy content (the think regex TypeErrors), malformed tool_call entries
(Message construction raises on both sides; typed here so the raise never
escapes the seam), and unhashable ``done_reason`` values (v1's dict lookup
TypeErrors).
"""

from __future__ import annotations

import json
from types import MappingProxyType

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
from .serialize import THINK_TAG_RE

_ParseResult = Result[ChatResponse, TranslationError]

FINISH_REASON_MIRROR: MappingProxyType[str, str] = MappingProxyType(
    {
        # The in-package mirror of v1's core_helpers._FINISH_REASON_MAP
        # (the package cannot import the v1 stack); the response gate's
        # mirror test pins every row against v1's live table. Unmapped
        # strings default to "stop" (v1 logs a warning and serves "stop").
        "stop_sequence": "stop",
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "refusal": "content_filter",
        "compaction": "length",
        "COMPLETE": "stop",
        "ERROR_TOXIC": "content_filter",
        "ERROR": "stop",
        "eos_token": "stop",
        "eos": "stop",
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "FINISH_REASON_UNSPECIFIED": "stop",
        "MALFORMED_FUNCTION_CALL": "stop",
        "LANGUAGE": "content_filter",
        "OTHER": "content_filter",
        "BLOCKLIST": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "SPII": "content_filter",
        "IMAGE_SAFETY": "content_filter",
        "IMAGE_PROHIBITED_CONTENT": "content_filter",
        "TOO_MANY_TOOL_CALLS": "stop",
        "MALFORMED_RESPONSE": "stop",
        "network_error": "stop",
        "sensitive": "content_filter",
        "guardrail_intervened": "content_filter",
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_calls",
        "function_call": "function_call",
        "content_filter": "content_filter",
        "content_filtered": "content_filter",
    }
)

_IR_FINISH: frozenset[str] = frozenset(
    {"stop", "length", "tool_calls", "content_filter"}
)


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    if not isinstance(raw, dict):
        return Error(_boundary("ollama response is not an object (v1 raises)"))
    message = raw.get("message")
    if not isinstance(message, dict):
        return Error(
            _boundary(
                "ollama response 'message' is missing or not an object (v1's "
                "Message(**message) raises TypeError)"
            )
        )
    remapped = _remapped_message(message)
    if isinstance(remapped, TranslationError):
        return Error(remapped)
    tool_calls = remapped.get("tool_calls")
    tool_reason = _tool_calls_reason(tool_calls)
    if tool_reason is not None:
        return Error(_boundary(tool_reason))
    finish = _finish(raw.get("done_reason"), tool_calls)
    if isinstance(finish, TranslationError):
        return Error(finish)
    counts = _usage_counts(raw)
    if isinstance(counts, TranslationError):
        return Error(counts)
    prompt, completion = counts
    body: dict[str, PlainJson] = {
        "object": "chat.completion",
        "model": f"ollama_chat/{request.model}",
        "choices": [{"index": 0, "finish_reason": finish, "message": remapped}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }
    return Ok(
        ChatResponse(
            id="",  # v1 keeps the ambient chatcmpl id (no wire id exists)
            model=f"ollama_chat/{request.model}",
            content=_semantic_blocks(remapped),
            finish=_ir_finish(finish),
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


def _remapped_message(
    message: dict[str, PlainJson],
) -> dict[str, PlainJson] | TranslationError:
    if "thinking" in message:
        remapped = {
            **{key: value for key, value in message.items() if key != "thinking"},
            "reasoning_content": message["thinking"],
        }
    else:
        content = message.get("content")
        if content is None:
            remapped = dict(message)
        elif not isinstance(content, str):
            return _boundary(
                "ollama message content is not a string (v1's think-tag "
                "regex raises TypeError)"
            )
        else:
            matched = THINK_TAG_RE.match(content)
            remapped = {
                **message,
                "reasoning_content": matched.group(1) if matched else None,
                "content": matched.group(2) if matched else content,
            }
    final_content = remapped.get("content")
    if not isinstance(final_content, str):
        return _boundary(
            "ollama message content is missing or null: v1's eval-count "
            "default eagerly runs token_counter(text=content) and raises "
            "ValueError even when eval_count is present (probed)"
        )
    return remapped


def _tool_calls_reason(tool_calls: PlainJson) -> str | None:
    """Pre-check the shapes both sides' ``Message(**...)`` validation raises
    on, so the raise never escapes the seam untyped (the cohere F8 rule).
    Value-level validation stays with Message itself — identical on both
    sides by construction."""
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, list):
        return "ollama message tool_calls is not a list (v1's Message raises)"
    for call in tool_calls:
        if not isinstance(call, dict):
            return "ollama tool_call entry is not an object (v1's Message raises)"
        function = call.get("function")
        if not isinstance(function, dict):
            return (
                "ollama tool_call has no 'function' object (v1's Message "
                "construction raises)"
            )
        arguments = function.get("arguments")
        if arguments is not None and not isinstance(arguments, (str, dict)):
            return (
                "ollama tool_call arguments are neither a string nor an "
                "object (v1's Function validation raises)"
            )
    return None


def _finish(
    done_reason: PlainJson, tool_calls: PlainJson
) -> str | TranslationError:
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        # v1 forces "tool_calls" whenever Message.tool_calls is truthy,
        # overriding the mapped done_reason
        return "tool_calls"
    if not done_reason:
        return "stop"  # v1's `done_reason or "stop"` falsy gate
    if isinstance(done_reason, str):
        mapped = FINISH_REASON_MIRROR.get(done_reason, "stop")
        if mapped not in _IR_FINISH:
            return _boundary(
                f"ollama done_reason {done_reason!r} maps to {mapped!r}, "
                "which the IR cannot carry (v1 serves it verbatim)"
            )
        return mapped
    if isinstance(done_reason, (list, dict)):
        return _boundary(
            "unhashable ollama done_reason (v1's finish-reason table lookup "
            "raises TypeError)"
        )
    return "stop"  # v1's map .get(non-str hashable) misses and defaults


def _ir_finish(finish: str) -> FinishReason:
    if finish == "length":
        return "length"
    if finish == "tool_calls":
        return "tool_calls"
    if finish == "content_filter":
        return "content_filter"
    return "stop"


def _usage_counts(raw: dict[str, PlainJson]) -> tuple[int, int] | TranslationError:
    if "prompt_eval_count" not in raw or "eval_count" not in raw:
        return _boundary(
            "ollama response missing prompt_eval_count/eval_count: v1 fills "
            "the gap with local litellm.token_counter estimates (a v1-stack "
            "call; deps hook deferred — typed fallback per the wave-3 dossier)"
        )
    prompt = _int_count(raw, "prompt_eval_count")
    if isinstance(prompt, TranslationError):
        return prompt
    completion = _int_count(raw, "eval_count")
    if isinstance(completion, TranslationError):
        return completion
    return prompt, completion


def _int_count(raw: dict[str, PlainJson], key: str) -> int | TranslationError:
    value = raw.get(key)
    if isinstance(value, (bool, int)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return _boundary(
        f"ollama {key} is not an integer: {value!r} (v1's raw addition or "
        "Usage validation raises)"
    )


def _semantic_blocks(message: dict[str, PlainJson]) -> Block[ContentBlock]:
    """Deliberately LENIENT beside the strict wire body (the cohere N3 rule):
    nothing reads ``ChatResponse.content`` on the openai construction arm
    today; the wire body is the parity surface."""
    blocks: list[ContentBlock] = []
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    if isinstance(content, str) and content and not tool_calls:
        blocks = [ContentBlock.of_text(Text(text=content, cache=Nothing))]
    for call in tool_calls if isinstance(tool_calls, list) else []:
        call_map = call if isinstance(call, dict) else {}
        function_raw = call_map.get("function")
        function = function_raw if isinstance(function_raw, dict) else {}
        arguments = function.get("arguments")
        parsed: PlainJson
        if isinstance(arguments, dict):
            parsed = arguments
        elif isinstance(arguments, str):
            try:
                parsed = json.loads(arguments) if arguments else {}
            except ValueError:
                parsed = {}
        else:
            parsed = {}
        identifier = call_map.get("id")
        name = function.get("name")
        blocks = [
            *blocks,
            ContentBlock.of_tool_use(
                ToolUse(
                    id=identifier if isinstance(identifier, str) else "",
                    name=name if isinstance(name, str) else "",
                    arguments=JsonBlob(value=parsed),
                    cache=Nothing,
                )
            ),
        ]
    return Block.of_seq(blocks)
