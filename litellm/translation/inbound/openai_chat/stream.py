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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from typing_extensions import assert_never

from ...ir import Body, CompositeChunk, PlainJson, StreamEvent

ChunkDialect = Literal["anthropic", "bedrock_converse", "gemini"]


@dataclass(frozen=True)
class StreamState:
    model: str
    sent_role: bool
    tool_index: int
    dialect: ChunkDialect
    seen_tool_calls: bool = False
    """gemini: tool calls and finishReason arrive in separate wire chunks, so
    a later ``stop`` rewrites to ``tool_calls`` (v1 ``has_seen_tool_calls``)."""


def initial_state(model: str = "", dialect: ChunkDialect = "anthropic") -> StreamState:
    return StreamState(model=model, sent_role=False, tool_index=-1, dialect=dialect)


_StepResult = tuple[StreamState, tuple[Body, ...]]


def step(state: StreamState, event: StreamEvent) -> _StepResult:
    match event.tag:
        case "start":
            return (
                StreamState(
                    model=state.model or event.start.model,
                    sent_role=state.sent_role,
                    tool_index=state.tool_index,
                    dialect=state.dialect,
                ),
                (),
            )
        case "text_delta":
            return _emit(state, {"content": event.text_delta.text})
        case "tool_use_start":
            started = StreamState(
                model=state.model,
                sent_role=state.sent_role,
                tool_index=state.tool_index + 1,
                dialect=state.dialect,
            )
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
            chunk: Body = {
                "model": state.model,
                "object": "chat.completion.chunk",
                **_top_level_fields(state.dialect),
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": event.finish.finish,
                    }
                ],
            }
            return state, (chunk,)
        case "stop":
            return state, ()
        case "chunk":
            return _gemini_chunk_step(state, event.chunk)
    assert_never(event.tag)


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
        case "anthropic":
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
    next_state = StreamState(
        model=state.model,
        sent_role=True,
        tool_index=state.tool_index,
        dialect=state.dialect,
    )
    return next_state, (body,)
