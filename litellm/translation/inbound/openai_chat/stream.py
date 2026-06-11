"""IR stream events -> OpenAI chat-completion chunk bodies.

A pure fold: ``step(state, event)`` returns the next state plus zero or more
chunk bodies. The shapes mirror what v1's CustomStreamWrapper emits for the
anthropic iterator: the first content-bearing chunk carries
``role: "assistant"``, tool chunks carry ``content: ""`` beside the tool
delta, tool indices count tool blocks (not content blocks), thinking deltas
ride ``reasoning_content`` + ``thinking_blocks`` + provider fields, and the
finish chunk is an empty delta with the mapped finish reason. The seam wraps
each body in ``ModelResponseStream`` (ambient id/created).
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import assert_never

from ...ir import Body, PlainJson, StreamEvent


@dataclass(frozen=True)
class StreamState:
    model: str
    sent_role: bool
    tool_index: int


def initial_state() -> StreamState:
    return StreamState(model="", sent_role=False, tool_index=-1)


_StepResult = tuple[StreamState, tuple[Body, ...]]


def step(state: StreamState, event: StreamEvent) -> _StepResult:
    match event.tag:
        case "start":
            return (
                StreamState(
                    model=event.start.model,
                    sent_role=state.sent_role,
                    tool_index=state.tool_index,
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
                _thinking_delta_body(event.thinking_delta.thinking, signature=""),
            )
        case "signature_delta":
            return _emit(
                state,
                _thinking_delta_body("", signature=event.signature_delta.signature),
            )
        case "finish":
            chunk: Body = {
                "model": state.model,
                "object": "chat.completion.chunk",
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
    assert_never(event.tag)


def _thinking_delta_body(thinking: str, signature: str) -> dict[str, PlainJson]:
    block: PlainJson = {
        "type": "thinking",
        "thinking": thinking,
        "signature": signature,
    }
    return {
        "content": "",
        "reasoning_content": thinking,
        "thinking_blocks": [block],
        "provider_specific_fields": {"thinking_blocks": [block]},
    }


def _emit(state: StreamState, delta: dict[str, PlainJson]) -> _StepResult:
    role: str | None = None if state.sent_role else "assistant"
    # v1's wrapper always sets provider_specific_fields (None when absent) on
    # content-bearing deltas; the finish chunk's empty delta never has it.
    body: Body = {
        "model": state.model,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"role": role, "provider_specific_fields": None, **delta},
                "finish_reason": None,
            }
        ],
    }
    next_state = StreamState(
        model=state.model, sent_role=True, tool_index=state.tool_index
    )
    return next_state, (body,)
