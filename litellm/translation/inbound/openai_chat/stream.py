"""IR stream events -> OpenAI chat-completion chunk bodies.

A pure fold: ``step(state, event)`` returns the next state plus zero or more
chunk bodies. The shapes mirror what v1's CustomStreamWrapper emits for each
provider iterator, so the fold carries a ``dialect``:

- ``anthropic`` (also bedrock_invoke, whose decoder wraps the anthropic
  parser): the first content-bearing chunk carries ``role: "assistant"``,
  thinking deltas ride ``reasoning_content`` + ``thinking_blocks`` (signature
  always present) + ``provider_specific_fields.thinking_blocks``;
- ``bedrock_converse``: thinking deltas mirror the raw ``reasoningContent``
  wire delta in ``provider_specific_fields`` and the thinking_blocks entry
  omits the signature key on text deltas (v1 ``converse_chunk_parser``).

Tool chunks carry ``content: ""`` beside the tool delta, tool indices count
tool blocks (not content blocks), and the finish chunk is an empty delta with
the mapped finish reason. The model field comes from ``message_start`` when
the provider streams one (anthropic) and from the request otherwise
(converse); v1's wrapper stamps its own model either way, so the seam wraps
each body in ``ModelResponseStream`` (ambient id/created).

The ``openai`` dialect folds ``wire_chunk`` events instead: the provider
parser already normalized each chunk to the SDK-dump shape, and this fold
owns only what v1's wrapper adds statefully (first emitted chunk carries
``role: assistant``, later roles are stripped, empty deltas are swallowed,
the wire stream id is pinned to the first non-empty one, and the trailing
``choices: []`` usage chunk passes through for the seam's include_usage
contract). The ``azure`` dialect is the openai fold plus the wrapper's azure
branch (streaming_handler.py:1448-1454): the model is re-read from every
chunk that carries one, choice-level ``content_filter_results`` survives on
content chunks (the ``StreamingChoices(**choice_json)`` rebuild) but not on
the finish flush (default-choice ``Delta(content=None)``), and the
empty-choices ``prompt_filter_results`` chunk is swallowed exactly like
v1's empty-delta handling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from typing_extensions import assert_never

from ...ir import Body, CompositeChunk, PlainJson, StreamEvent

ChunkDialect = Literal["anthropic", "bedrock_converse", "openai", "azure", "gemini"]


@dataclass(frozen=True)
class StreamState:
    model: str
    sent_role: bool
    tool_index: int
    dialect: ChunkDialect
    stream_id: str | None = None
    """openai dialect only: the first non-empty wire chunk id, stamped on
    every emitted chunk exactly like v1's ``set_model_id``."""
    seen_tool_calls: bool = False
    """gemini: tool calls and finishReason arrive in separate wire chunks, so
    a later ``stop`` rewrites to ``tool_calls`` (v1 ``has_seen_tool_calls``)."""


def initial_state(model: str = "", dialect: ChunkDialect = "anthropic") -> StreamState:
    return StreamState(model=model, sent_role=False, tool_index=-1, dialect=dialect)


_StepResult = tuple[StreamState, tuple[Body, ...]]


def step(state: StreamState, event: StreamEvent) -> _StepResult:
    match event.tag:
        case "start":
            return replace(state, model=state.model or event.start.model), ()
        case "text_delta":
            return _emit(state, {"content": event.text_delta.text})
        case "tool_use_start":
            started = replace(state, tool_index=state.tool_index + 1)
            return _emit(
                started,
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": event.tool_use_start.id,
                            "type": "function",
                            "function": {
                                "name": event.tool_use_start.name,
                                "arguments": "",
                            },
                            "index": started.tool_index,
                        }
                    ],
                },
            )
        case "tool_args_delta":
            return _emit(
                state,
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": None,
                            "type": "function",
                            "function": {
                                "name": None,
                                "arguments": event.tool_args_delta.partial_json,
                            },
                            "index": max(state.tool_index, 0),
                        }
                    ],
                },
            )
        case "thinking_delta":
            return _emit(
                state,
                _thinking_delta_body(
                    state.dialect, event.thinking_delta.thinking, signature=None
                ),
            )
        case "signature_delta":
            return _emit(
                state,
                _thinking_delta_body(
                    state.dialect, "", signature=event.signature_delta.signature
                ),
            )
        case "finish":
            return state, (_finish_chunk(state, event.finish.finish),)
        case "stop":
            return state, ()
        case "wire_chunk":
            return _step_openai(state, event.wire_chunk.value)
        case "chunk":
            return _gemini_chunk_step(state, event.chunk)
    assert_never(event.tag)


def _finish_chunk(state: StreamState, finish: str) -> Body:
    return {
        "model": state.model,
        "object": "chat.completion.chunk",
        **_top_level_fields(state.dialect),
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish,
            }
        ],
    }


_OPENAI_WIRE_KEYS = frozenset({"id", "model", "system_fingerprint", "choices", "usage"})


def _step_openai(state: StreamState, chunk: PlainJson) -> _StepResult:
    if state.dialect not in ("openai", "azure") or not isinstance(chunk, dict):
        return state, ()  # cross-family parsers never emit wire chunks
    chunk_id = chunk.get("id")
    pinned = state.stream_id or (
        chunk_id if isinstance(chunk_id, str) and chunk_id.strip() else None
    )
    chunk_model = chunk.get("model")
    model = (
        chunk_model
        if state.dialect == "azure" and isinstance(chunk_model, str)
        else state.model
    )
    next_state = replace(state, stream_id=pinned, model=model)
    base: Body = {
        "id": pinned,
        "model": model,
        "object": "chat.completion.chunk",
        "system_fingerprint": chunk.get("system_fingerprint"),
    }
    # non-envelope wire keys (e.g. service_tier): v1 setattrs them onto every
    # content/finish chunk via preserve_upstream_non_openai_attributes
    extras: Body = {
        key: value for key, value in chunk.items() if key not in _OPENAI_WIRE_KEYS
    }
    usage = chunk.get("usage")
    choices = chunk.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        # The trailing choices=[] usage chunk passes through verbatim: v1's
        # wrapper consumes it into its own synthesized final usage chunk
        # (stream_chunk_builder), which is envelope, not decode; the seam
        # owns reproducing that when stream_options.include_usage was sent.
        if isinstance(usage, dict):
            return next_state, ({**base, "choices": [], "usage": usage},)
        return next_state, ()
    choice = choices[0]
    if not isinstance(choice, dict):
        return next_state, ()
    raw_delta = choice.get("delta")
    delta = raw_delta if isinstance(raw_delta, dict) else {}
    index = choice.get("index")
    finish = choice.get("finish_reason")
    usage_field: dict[str, PlainJson] = (
        {"usage": usage} if isinstance(usage, dict) else {}
    )
    if isinstance(finish, str):
        # v1's flush: an empty delta plus the mapped finish reason (identity
        # for the admitted openai values), index from the default choice.
        body: Body = {
            **base,
            **extras,
            **usage_field,
            "choices": [
                {
                    "index": index if isinstance(index, int) else 0,
                    "delta": {"content": None},
                    "finish_reason": finish,
                }
            ],
        }
        return next_state, (body,)
    if not _openai_chunk_non_empty(state, delta):
        return next_state, ()  # v1 swallows empty deltas (chunks-for-usage only)
    emitted_delta: dict[str, PlainJson]
    if state.sent_role:
        # v1's strip_role_from_delta rebuilds the Delta from a model_dump,
        # which surfaces provider_specific_fields: None on stripped chunks.
        emitted_delta = {
            **{key: value for key, value in delta.items() if key != "role"},
            "provider_specific_fields": None,
        }
    else:
        emitted_delta = {**delta, "role": "assistant"}
    body = {
        **base,
        **extras,
        **usage_field,
        "choices": [
            {
                "index": index if isinstance(index, int) else 0,
                "delta": emitted_delta,
                "logprobs": None,
                "finish_reason": None,
                # azure's choice-level filter annotation survives on content
                # chunks (v1 rebuilds StreamingChoices from the wire choice);
                # the finish flush above uses the default choice and drops it.
                **_content_filter_field(choice),
            }
        ],
    }
    return replace(next_state, sent_role=True), (body,)


def _content_filter_field(choice: dict[str, PlainJson]) -> dict[str, PlainJson]:
    if "content_filter_results" in choice:
        return {"content_filter_results": choice["content_filter_results"]}
    return {}


def _openai_chunk_non_empty(state: StreamState, delta: dict[str, PlainJson]) -> bool:
    """v1 ``is_chunk_non_empty`` for the openai branch: content text, tool
    deltas, or the first chunk's role (refusal-only deltas are swallowed,
    exactly like v1)."""
    content = delta.get("content")
    if isinstance(content, str) and len(content) > 0:
        return True
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        return True
    return not state.sent_role and delta.get("role") is not None


_GEMINI_METADATA_FIELDS: tuple[str, ...] = (
    "vertex_ai_grounding_metadata",
    "vertex_ai_url_context_metadata",
    "vertex_ai_safety_ratings",
    "vertex_ai_safety_results",
    "vertex_ai_citation_metadata",
)


def _gemini_chunk_step(state: StreamState, chunk: CompositeChunk) -> _StepResult:
    """One composite gemini wire event -> at most one content chunk plus, at
    stream end, the finish chunk v1's wrapper synthesizes (the finish-bearing
    wire event is the last one, so emitting both here yields v1's order).
    Usage is withheld exactly like v1 without ``stream_options``."""
    text = chunk.text.default_value(None)
    reasoning = chunk.reasoning.default_value(None)
    has_payload = text is not None or reasoning is not None or len(chunk.tool_calls) > 0
    tool_index = state.tool_index
    bodies: tuple[Body, ...] = ()
    sent_role = state.sent_role
    if has_payload:
        entries: list[PlainJson] = []
        for call in chunk.tool_calls:
            tool_index = tool_index + 1
            entry: dict[str, PlainJson] = {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
                "index": tool_index,
            }
            entries = [*entries, entry]
        signatures: list[PlainJson] = list(chunk.signatures)
        delta: dict[str, PlainJson] = {
            "role": None if sent_role else "assistant",
            "content": text,
            "reasoning_content": reasoning,
            "tool_calls": entries or None,
            "provider_specific_fields": (
                {"thought_signatures": signatures} if signatures else None
            ),
        }
        body: Body = {
            "id": chunk.id,
            "model": state.model,
            "object": "chat.completion.chunk",
            **{field: [] for field in _GEMINI_METADATA_FIELDS},
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        bodies = (body,)
        sent_role = True
    seen = state.seen_tool_calls or len(chunk.tool_calls) > 0
    finish = chunk.finish.default_value(None)
    if finish is not None:
        final: Body = {
            "id": chunk.id,
            "model": state.model,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": (
                        "tool_calls" if seen and finish == "stop" else finish
                    ),
                }
            ],
        }
        bodies = (*bodies, final)
    return (
        StreamState(
            model=state.model,
            sent_role=sent_role,
            tool_index=tool_index,
            dialect=state.dialect,
            seen_tool_calls=seen,
        ),
        bodies,
    )


def _thinking_delta_body(
    dialect: ChunkDialect, thinking: str, signature: str | None
) -> dict[str, PlainJson]:
    match dialect:
        case "gemini":
            # Unreachable: the gemini dialect rides composite chunk events,
            # never per-block thinking deltas; the anthropic shape is a
            # harmless placeholder that keeps the match exhaustive.
            return {"content": "", "reasoning_content": thinking}
        case "anthropic" | "openai" | "azure":
            # openai/azure-dialect streams never produce thinking deltas (the
            # provider parser emits wire_chunk events only); the arm exists
            # for exhaustiveness and mirrors the anthropic shape.
            block: PlainJson = {
                "type": "thinking",
                "thinking": thinking,
                "signature": signature if signature is not None else "",
            }
            return {
                "content": "",
                "reasoning_content": thinking,
                "thinking_blocks": [block],
                "provider_specific_fields": {"thinking_blocks": [block]},
            }
        case "bedrock_converse":
            wire_delta: PlainJson = (
                {"signature": signature}
                if signature is not None
                else {"text": thinking}
            )
            converse_block: dict[str, PlainJson] = {
                "type": "thinking",
                "thinking": thinking,
            }
            if signature is not None:
                converse_block = {**converse_block, "signature": signature}
            return {
                "content": "",
                "reasoning_content": thinking,
                "thinking_blocks": [converse_block],
                "provider_specific_fields": {"reasoningContent": wire_delta},
            }
    assert_never(dialect)


def _top_level_fields(dialect: ChunkDialect) -> dict[str, PlainJson]:
    """v1's converse parser stamps top-level ``provider_specific_fields: {}``
    on every chunk; the anthropic iterator leaves it unset (None)."""
    if dialect == "bedrock_converse":
        return {"provider_specific_fields": {}}
    return {}


def _emit(state: StreamState, delta: dict[str, PlainJson]) -> _StepResult:
    role: str | None = None if state.sent_role else "assistant"
    # v1's wrapper always sets provider_specific_fields (None when absent) on
    # content-bearing deltas; the finish chunk's empty delta never has it.
    body: Body = {
        "model": state.model,
        "object": "chat.completion.chunk",
        **_top_level_fields(state.dialect),
        "choices": [
            {
                "index": 0,
                "delta": {"role": role, "provider_specific_fields": None, **delta},
                "finish_reason": None,
            }
        ],
    }
    return replace(state, sent_role=True), (body,)
