"""The ONE httpx chunk-dialect normalizer behind every dict-path stream parser.

v1's httpx providers decode streams through ``BaseModelResponseIterator``
subclasses whose ``chunk_parser`` rebuilds a ``ModelResponseStream`` from the
wire dict — the shared consequences (extras and ``system_fingerprint``
dropped, the dict-path tool_call ``type: "function"`` default, usage withheld
from content/finish chunks and attached only to the ``choices: []`` tail,
the reasoning -> reasoning_content rewrite) are ONE mechanism here,
parameterized by a frozen ``HttpxChunkPolicy`` instead of pasted per provider
(critic-wave2a M2 — the openai N3 -> azure N1 -> grok M3 ``make_parse_line``
trajectory, applied one level up at the chunk seam). Consumers compose
``make_parse_event(policy)`` and feed it to ``make_parse_line``:

- xai: ``reasoning="rename"`` (the base handler pops ``delta.reasoning`` into
  ``reasoning_content``), value-check error chunks, no required envelope
  keys, the xai usage fold hook (reasoning fold + total normalize).
- cometapi: ``reasoning="copy_both"`` (v1 assigns WITHOUT popping, so both
  keys reach the Delta), key-presence error chunks, strict
  id/created/model/choices envelope (v1 KeyErrors -> CometAPIException).
- openrouter (wave-2b-alpha): ``reasoning="unconditional"`` — every delta
  gains ``reasoning_content`` (None when ``reasoning`` is absent; a native
  ``reasoning_content`` clobbered), key-presence error chunks, strict
  id/created/model/choices envelope (v1 KeyErrors -> OpenRouterException).
- wave 2b remaining (groq: rename with pop semantics == "rename"): extend
  ``ReasoningMode`` with the new Literal value and its arm IN THE SAME
  COMMIT as the consumer and its differential rows — never a third copy of
  this file's machinery, and never an arm without a consumer (the
  placeholder-arm rule).

Shapes a v2-sent request cannot trigger (multiple choices, ``function_call``
deltas, logprobs, unknown delta/tool_call keys) are loud error values.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from expression import Error, Ok, Result
from expression.collections import Block
from typing_extensions import assert_never

from ...errors import BoundaryError, TranslationError
from ...ir import JsonBlob, PlainJson, StreamEvent

_EventResult = Result[StreamEvent | None, TranslationError]
ParseEvent = Callable[[PlainJson], _EventResult]

ReasoningMode = Literal["rename", "copy_both", "unconditional"]
"""How a wire ``delta.reasoning`` reaches the emitted delta: "rename" emits
``reasoning_content`` only (xai; groq's pop has the same output), "copy_both"
keeps the original key beside the copy (cometapi). Native
``reasoning_content`` deltas pass through verbatim in those two modes.
"unconditional" is openrouter's variant (wave-2b-alpha, added with its
consumer per the no-consumer-no-arm rule): EVERY delta gains
``reasoning_content = delta.get("reasoning")`` — ``None`` when ``reasoning``
is absent, the original ``reasoning`` key kept beside it when present, and a
native wire ``reasoning_content`` CLOBBERED by the assignment (v1
openrouter's chunk_parser, replay-pinned)."""

_DELTA_KEYS = (
    "content",
    "function_call",
    "refusal",
    "role",
    "tool_calls",
    "reasoning",
    "reasoning_content",
)


@dataclass(frozen=True)
class StrictEnvelope:
    """Envelope keys whose ABSENCE is loud (v1 chunk_parsers that subscript
    the chunk raise KeyError) PLUS the reason naming that v1 raise — one
    value, so a strict envelope can never ship without its v1-raise naming
    (critic-longtail NIT-3: the pair was a docstring-only contract)."""

    keys: tuple[str, ...]
    reason: Callable[[Sequence[str]], str]


@dataclass(frozen=True)
class HttpxChunkPolicy:
    reasoning: ReasoningMode
    error_on_key_presence: bool = False
    """True mirrors a v1 ``if "error" in chunk:`` raise (cometapi); False
    mirrors the value check (xai's ``get("error") is not None``)."""
    strict_envelope: StrictEnvelope | None = None
    """The required-keys envelope with its v1-raise naming (cometapi); None
    for parsers that ``.get`` everything."""
    fold_usage: Callable[[PlainJson], PlainJson | TranslationError] | None = None
    """Per-chunk usage rewrite (the xai reasoning fold); None passes a dict
    usage through verbatim. Either way usage is attached ONLY to the
    ``choices: []`` tail — v1's wrapper strips it from every emitted
    content/finish chunk and re-synthesizes the final usage chunk."""


BASE_HANDLER_POLICY = HttpxChunkPolicy(reasoning="rename")
"""The compat_httpx FAMILY policy — v1's BASE
``OpenAIChatCompletionStreamingHandler`` rebuild (reasoning rename,
value-checked error chunks, no required envelope keys, wire usage verbatim
on the ``choices: []`` tail only). ONE name for the shared truth
(critic-wave2b-alpha NIT-1: it was declared six times): compat_httpx and the
five base-handler own modules (deepseek, hosted_vllm, fireworks_ai,
snowflake, huggingface) all compose ``make_parse_event(BASE_HANDLER_POLICY)``
— a family-policy fix is a one-site edit here, and each consumer's docstring
stays the provider-specific pinned truth (v1 = the base handler)."""


def _envelope_error(
    policy: HttpxChunkPolicy, event: dict[str, PlainJson]
) -> TranslationError | None:
    error_present = (
        "error" in event
        if policy.error_on_key_presence
        else event.get("error") is not None
    )
    if error_present:
        return _boundary(f"provider stream error: {event.get('error')!r}")
    if policy.strict_envelope is None:
        return None
    missing = [key for key in policy.strict_envelope.keys if key not in event]
    if not missing:
        return None
    return _boundary(policy.strict_envelope.reason(missing))


def make_parse_event(policy: HttpxChunkPolicy) -> ParseEvent:
    def parse_event(event: PlainJson) -> _EventResult:
        if not isinstance(event, dict):
            return Error(_boundary("stream chunk is not an object"))
        envelope_error = _envelope_error(policy, event)
        if envelope_error is not None:
            return Error(envelope_error)
        choices = event.get("choices")
        if not isinstance(choices, list):
            return Error(_boundary("stream chunk 'choices' is missing"))
        if len(choices) > 1:
            return Error(
                TranslationError.of_unsupported(
                    "multiple stream choices (n > 1); unreachable for v2-sent requests"
                )
            )
        usage = _usage_value(policy, event.get("usage"))
        if isinstance(usage, TranslationError):
            return Error(usage)
        normalized_choices: list[PlainJson] = []
        if len(choices) == 1:
            normalized = _normalize_choice(policy, choices[0])
            if isinstance(normalized, TranslationError):
                return Error(normalized)
            normalized_choices = [normalized]
        identifier = event.get("id")
        chunk: dict[str, PlainJson] = {
            # No extras passthrough and no system_fingerprint: the v1
            # chunk_parser rebuilds keep only id/created/usage/model/choices.
            "id": identifier if isinstance(identifier, str) else None,
            "system_fingerprint": None,
            "choices": normalized_choices,
            "usage": usage if len(normalized_choices) == 0 else None,
        }
        return Ok(StreamEvent.of_wire_chunk(JsonBlob(value=chunk)))

    return parse_event


def _usage_value(
    policy: HttpxChunkPolicy, raw_usage: PlainJson
) -> PlainJson | TranslationError:
    if policy.fold_usage is not None:
        return policy.fold_usage(raw_usage)
    return raw_usage if isinstance(raw_usage, dict) else None


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def _string_or_none(value: PlainJson) -> PlainJson:
    return value if isinstance(value, str) else None


def _normalize_choice(
    policy: HttpxChunkPolicy, choice: PlainJson
) -> PlainJson | TranslationError:
    if not isinstance(choice, dict):
        return _boundary("stream choice is not an object")
    extra_keys = set(choice.keys()) - {"index", "delta", "logprobs", "finish_reason"}
    if extra_keys:
        return TranslationError.of_unsupported(
            f"stream choice keys {sorted(extra_keys)!r}; unreachable for v2-sent requests"
        )
    if choice.get("logprobs") is not None:
        return TranslationError.of_unsupported(
            "stream logprobs; unreachable for v2-sent requests"
        )
    finish = choice.get("finish_reason")
    if finish is not None and not isinstance(finish, str):
        return _boundary("stream finish_reason is not a string")
    if finish == "function_call":
        return TranslationError.of_unsupported(
            "legacy function_call stream finish; the v2 surface cannot send 'functions'"
        )
    raw_delta = choice.get("delta")
    delta = raw_delta if isinstance(raw_delta, dict) else {}
    normalized_delta = _normalize_delta(policy, delta)
    if isinstance(normalized_delta, TranslationError):
        return normalized_delta
    if finish is not None and _delta_bears_content(normalized_delta):
        return TranslationError.of_unsupported(
            "finish chunk with a non-empty delta; v1's wrapper interleaves it"
        )
    index = choice.get("index")
    return {
        "index": index if isinstance(index, int) else 0,
        "delta": normalized_delta,
        "logprobs": None,
        "finish_reason": finish,
    }


def _normalize_delta(
    policy: HttpxChunkPolicy, delta: dict[str, PlainJson]
) -> dict[str, PlainJson] | TranslationError:
    extra_keys = set(delta.keys()) - set(_DELTA_KEYS)
    if extra_keys:
        return TranslationError.of_unsupported(
            f"stream delta keys {sorted(extra_keys)!r}; unreachable for v2-sent requests"
        )
    if delta.get("function_call") is not None:
        return TranslationError.of_unsupported(
            "legacy function_call stream delta; the v2 surface cannot send 'functions'"
        )
    tool_calls = delta.get("tool_calls")
    normalized_calls: PlainJson = None
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            return _boundary("stream delta 'tool_calls' is not an array")
        gathered: list[PlainJson] = []
        for call in tool_calls:
            normalized = _normalize_tool_call(call)
            if isinstance(normalized, TranslationError):
                return normalized
            gathered = [*gathered, normalized]
        normalized_calls = gathered
    # provider_specific_fields: None always (the dict-path wrapper stamps the
    # null provider field on content-bearing deltas); refusal and the
    # reasoning keys ride set-only, mirroring Delta's set-field serialization
    # on the dict path — v1 FORWARDS a refusal that rides a role/content
    # delta and swallows refusal-only deltas (verifier-grok F1).
    base: dict[str, PlainJson] = {
        "content": _string_or_none(delta.get("content")),
        "function_call": None,
        "provider_specific_fields": None,
        "role": _string_or_none(delta.get("role")),
        "tool_calls": normalized_calls,
    }
    if "refusal" in delta:
        base = {**base, "refusal": _string_or_none(delta.get("refusal"))}
    reasoning_error = _non_string_reasoning_error(policy.reasoning, delta)
    if reasoning_error is not None:
        return reasoning_error
    return _reasoning_tail(policy.reasoning, delta, base)


def _non_string_reasoning_error(
    mode: ReasoningMode, delta: dict[str, PlainJson]
) -> TranslationError | None:
    """verifier-wave2b-alpha F3: v1's chunk_parsers serve a non-string
    reasoning value verbatim, then the stream CRASHES at the end (APIError
    out of the chunk builder's reasoning join) — never serve what v1 raises
    on, so the coercion-to-None that silently swallowed the chunk is now a
    loud error. The native ``reasoning_content`` key is exempt in the
    unconditional mode only: there v1's assignment clobbers it to None
    before it can reach the join (replay-pinned), so v1 never crashes."""
    checked = (
        ("reasoning",)
        if mode == "unconditional"
        else ("reasoning", "reasoning_content")
    )
    for key in checked:
        value = delta.get(key)
        if value is not None and not isinstance(value, str):
            return _boundary(
                f"non-string stream delta {key!r} ({type(value).__name__}): "
                "v1 serves the verbatim value, then raises APIError at stream "
                "end joining reasoning for the chunk builder"
            )
    return None


def _reasoning_tail(
    mode: ReasoningMode, delta: dict[str, PlainJson], base: dict[str, PlainJson]
) -> dict[str, PlainJson]:
    # every ReasoningMode member named, assert_never on the residual: a new
    # Literal value without an arm here is a pyright error at this line, not
    # silent rename semantics (critic-wave2b-alpha MAJOR-3, the UsageStyle/M2
    # precedent — "rename" used to be the implicit else)
    if mode == "unconditional":
        value = _string_or_none(delta.get("reasoning"))
        if "reasoning" in delta:
            return {**base, "reasoning": value, "reasoning_content": value}
        return {**base, "reasoning_content": value}
    if mode == "copy_both":
        if "reasoning" in delta:
            value = _string_or_none(delta.get("reasoning"))
            # v1 cometapi assigns without popping: both keys reach the Delta
            return {**base, "reasoning": value, "reasoning_content": value}
        return _native_reasoning_tail(delta, base)
    if mode == "rename":
        if "reasoning" in delta:
            return {
                **base,
                "reasoning_content": _string_or_none(delta.get("reasoning")),
            }
        return _native_reasoning_tail(delta, base)
    assert_never(mode)


def _native_reasoning_tail(
    delta: dict[str, PlainJson], base: dict[str, PlainJson]
) -> dict[str, PlainJson]:
    """A native wire ``reasoning_content`` passes through verbatim in the
    rename and copy_both modes (the unconditional mode CLOBBERS it instead)."""
    if "reasoning_content" in delta:
        return {
            **base,
            "reasoning_content": _string_or_none(delta.get("reasoning_content")),
        }
    return base


def _normalize_tool_call(call: PlainJson) -> PlainJson | TranslationError:
    if not isinstance(call, dict):
        return _boundary("stream tool_call is not an object")
    extra_keys = set(call.keys()) - {"index", "id", "function", "type"}
    if extra_keys:
        return TranslationError.of_unsupported(
            f"stream tool_call keys {sorted(extra_keys)!r}; unreachable for v2-sent requests"
        )
    raw_function = call.get("function")
    function = raw_function if isinstance(raw_function, dict) else {}
    index = call.get("index")
    return {
        "index": index if isinstance(index, int) else 0,
        "id": _string_or_none(call.get("id")),
        "function": {
            "arguments": _string_or_none(function.get("arguments")),
            "name": _string_or_none(function.get("name")),
        },
        # dict-path default: ChatCompletionDeltaToolCall fills "function" when
        # the wire omits the key (the SDK path keeps None there)
        "type": _string_or_none(call.get("type")) if "type" in call else "function",
    }


def _delta_bears_content(delta: dict[str, PlainJson]) -> bool:
    """Does a NORMALIZED delta carry payload v1's wrapper would interleave
    as its own chunk ahead of the finish chunk? Non-empty content,
    tool_calls — and the refusal/reasoning keys (verifier-wave2b-alpha F1:
    a reasoning-bearing finish delta used to be served as an EMPTY finish
    chunk, silently dropping text v1 serves; now it takes the loud
    finish-chunk fallback above). Those keys count only on non-None values:
    the unconditional mode stamps ``reasoning_content: None`` onto every
    delta, including the bare finish chunks v1 serves."""
    content = delta.get("content")
    if isinstance(content, str) and len(content) > 0:
        return True
    if delta.get("tool_calls") is not None:
        return True
    return any(
        delta.get(key) is not None
        for key in ("refusal", "reasoning", "reasoning_content")
    )
