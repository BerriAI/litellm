import json
from collections import deque
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from openai._streaming import ServerSentEvent
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.prompt_templates.server_tool_responses import (
    combined_tool_response,
    object_items,
    object_value,
    public_tool_response,
)
from litellm.litellm_core_utils.prompt_templates.server_tools import ServerToolRoute
from litellm.llms.anthropic.experimental_pass_through.messages.agentic_streaming_iterator import (
    AgenticAnthropicStreamingIterator,
)

_OBJECT: Final = TypeAdapter(dict[str, object])
_MAX_ROUND_BYTES: Final = 16 * 1024 * 1024


def sse_bytes(value: Mapping[str, object], event: str | None = None) -> bytes:
    return ((f"event: {event}\n" if event else "") + "data: " + json.dumps(value) + "\n\n").encode()


class ServerToolStream:
    def __init__(
        self, route: ServerToolRoute, server_names: frozenset[str], request: Mapping[str, object] | None = None
    ) -> None:
        self.route: Final[ServerToolRoute] = route
        self.server_names = server_names
        original: Final = request if request is not None else MappingProxyType({})
        self.client_response_fields = (
            MappingProxyType(
                {
                    "instructions": original.get("instructions"),
                    "tools": original.get("tools", ()),
                    "previous_response_id": original.get("previous_response_id"),
                }
            )
            if route == "aresponses"
            else MappingProxyType({})
        )
        self.responses: tuple[Mapping[str, object], ...] = ()
        self.frames: deque[bytes]  # mutable-ok: Bounded SSE buffers require linear-time appends.
        self.frames = deque(  # mutable-ok: Bounded SSE buffers require linear-time appends.
        )
        self.objects: deque[Mapping[str, object]]  # mutable-ok: Bounded SSE buffers require linear-time appends.
        self.objects = deque(  # mutable-ok: Bounded SSE buffers require linear-time appends.
        )
        self.tool_chunks: deque[Mapping[str, object]]  # mutable-ok: Bounded SSE buffers require linear-time appends.
        self.tool_chunks = deque(  # mutable-ok: Bounded SSE buffers require linear-time appends.
        )
        self.chat_names: Mapping[int, str] = MappingProxyType({})
        self.indices: Mapping[int, int | None] = MappingProxyType({})
        self.content_count = 0
        self.sequence = 0
        self.round_size = 0
        self.terminal = False
        self.response_id: str | None = None
        self.complete_response: Mapping[str, object] | None = None
        self.suppress_output = False

    def begin_round(self) -> None:
        self.frames.clear()
        self.objects.clear()
        self.tool_chunks.clear()
        self.chat_names = MappingProxyType({})
        self.indices = MappingProxyType({})
        self.round_size = 0
        self.terminal = False
        self.complete_response = None

    def _emit(self, data: Mapping[str, object], event: str | None = None) -> bytes:
        if self.route == "aresponses":
            result: Final = sse_bytes(
                {  # mutable-ok: Native provider JSON containers.
                    **data,
                    "sequence_number": self.sequence,
                },
                event,
            )
            self.sequence += 1
            return result
        return sse_bytes(
            {  # mutable-ok: Native provider JSON containers.
                **data,
                **(
                    {  # mutable-ok: Native provider JSON containers.
                        "id": self.response_id
                    }
                    if self.route == "acompletion"
                    else {  # mutable-ok: Native provider JSON containers.
                    }
                ),
            },
            event,
        )

    def feed(self, event: ServerSentEvent) -> tuple[bytes, ...]:
        if event.data == "[DONE]":
            return ()
        if not event.data:
            return ()
        data: Final = _OBJECT.validate_json(event.data)
        frame: Final = sse_bytes(data, event.event)
        self.round_size += len(frame)
        if self.round_size > _MAX_ROUND_BYTES:
            raise ValueError("The gateway tool response exceeded its retained-output limit")
        self.frames.append(frame)
        self.objects.append(data)
        if data.get("error") or data.get("type") in ("error", "response.failed"):
            raise ValueError("The model stream failed during gateway tool execution")
        emitted: Final = (
            self._anthropic(data)
            if self.route == "anthropic_messages"
            else self._responses(data)
            if self.route == "aresponses"
            else self._chat(data)
        )
        return () if self.suppress_output else emitted

    def _anthropic(self, data: Mapping[str, object]) -> tuple[bytes, ...]:
        kind: Final = data.get("type")
        if kind == "message_start":
            if self.response_id is not None:
                return ()
            self.response_id = str(object_value(data.get("message")).get("id", ""))
            return (self._emit(data, "message_start"),)
        if kind in ("message_delta", "message_stop"):
            if kind == "message_stop":
                self.terminal = True
            return ()
        index: Final = data.get("index")
        if kind == "content_block_start" and isinstance(index, int):
            block: Final = object_value(data.get("content_block"))
            hidden: Final = block.get("type") == "tool_use" and block.get("name") in self.server_names
            self.indices = MappingProxyType({**self.indices, index: None if hidden else self.content_count})
            if not hidden:
                self.content_count += 1
        if isinstance(index, int):
            mapped: Final = self.indices.get(index)
            if mapped is None:
                return ()
            return (
                self._emit(
                    {  # mutable-ok: Native provider JSON containers.
                        **data,
                        "index": mapped,
                    },
                    str(kind),
                ),
            )
        return (self._emit(data, str(kind)),)

    def _responses(self, data: Mapping[str, object]) -> tuple[bytes, ...]:
        kind: Final = str(data.get("type", ""))
        if kind in ("response.completed", "response.incomplete"):
            self.complete_response = object_value(data.get("response"))
            self.terminal = True
            return ()
        if kind in ("response.created", "response.in_progress"):
            if self.responses:
                return ()
            response: Final = object_value(data.get("response"))
            if self.response_id is None:
                self.response_id = str(response.get("id", ""))
            return (
                self._emit(
                    {  # mutable-ok: Native provider JSON containers.
                        **data,
                        "response": {  # mutable-ok: Native provider JSON containers.
                            **response,
                            **self.client_response_fields,
                            "id": self.response_id,
                        },
                    },
                    kind,
                ),
            )
        index: Final = data.get("output_index")
        if kind == "response.output_item.added" and isinstance(index, int):
            item: Final = object_value(data.get("item"))
            hidden: Final = item.get("type") == "function_call" and item.get("name") in self.server_names
            self.indices = MappingProxyType({**self.indices, index: None if hidden else self.content_count})
            if not hidden:
                self.content_count += 1
        if isinstance(index, int):
            mapped: Final = self.indices.get(index)
            if mapped is None:
                return ()
            return (
                self._emit(
                    {  # mutable-ok: Native provider JSON containers.
                        **data,
                        "output_index": mapped,
                    },
                    kind,
                ),
            )
        return (self._emit(data, kind),)

    def _chat(self, data: Mapping[str, object]) -> tuple[bytes, ...]:
        if self.response_id is None:
            self.response_id = str(data.get("id", ""))
        choices: Final = object_items(data.get("choices"))
        if not choices:
            return ()
        choice: Final = choices[0]
        delta: Final = object_value(choice.get("delta"))
        calls: Final = object_items(delta.get("tool_calls"))
        if calls:
            self.tool_chunks.append(data)
            self.chat_names = MappingProxyType(
                {
                    **self.chat_names,
                    **MappingProxyType(
                        {
                            index: self.chat_names.get(index, "") + name
                            for call in calls
                            if isinstance(index := call.get("index"), int)
                            and isinstance(
                                name := (object_value(call.get("function")) or object_value(call.get("custom"))).get(
                                    "name"
                                ),
                                str,
                            )
                        }
                    ),
                }
            )
        if choice.get("finish_reason") is not None:
            self.terminal = True
        visible: Final = {  # mutable-ok: Native provider JSON containers.
            key: value for key, value in delta.items() if key != "tool_calls"
        }
        if not visible or self.responses and len(visible) == 1 and visible.get("role") == "assistant":
            return ()
        return (
            self._emit(
                {  # mutable-ok: Native provider JSON containers.
                    **data,
                    "usage": None,
                    "choices": [  # mutable-ok: Native provider JSON containers.
                        {  # mutable-ok: Native provider JSON containers.
                            **choice,
                            "delta": visible,
                            "finish_reason": None,
                        }
                    ],
                }
            ),
        )

    def _round_response(self) -> Mapping[str, object] | None:
        if self.route == "aresponses":
            return self.complete_response
        if self.route == "anthropic_messages":
            return _OBJECT.validate_python(
                AgenticAnthropicStreamingIterator._rebuild_anthropic_response_from_sse(  # pyright: ignore[reportPrivateUsage]  # Reuse the native Anthropic stream accumulator.
                    list(  # mutable-ok: Native provider JSON containers.
                        self.frames
                    )
                )
            )
        built: Final = litellm.stream_chunk_builder(  # pyright: ignore[reportUnknownMemberType]  # Legacy builder accepts validated JSON chunks.
            list(  # mutable-ok: Native provider JSON containers.
                self.objects
            )
        )
        return _OBJECT.validate_python(built.model_dump(mode="json")) if built is not None else None  # pyright: ignore[reportUnknownMemberType]  # Validate the legacy response at the wire boundary.

    def finish_round(self) -> tuple[Mapping[str, object], tuple[bytes, ...]]:
        if not self.terminal:
            raise ValueError("The model stream ended before its terminal event")
        response: Final = self._round_response()
        if response is None:
            raise ValueError("The model stream did not contain a complete response")
        self.accept_response(response)
        if self.route != "acompletion":
            return response, ()
        client_indices: Final = MappingProxyType(
            {
                index: position
                for position, index in enumerate(
                    sorted(index for index, name in self.chat_names.items() if name not in self.server_names)
                )
            }
        )
        chunks: Final = tuple(
            self._emit(
                {  # mutable-ok: Native provider JSON containers.
                    **chunk,
                    "usage": None,
                    "choices": [  # mutable-ok: Native provider JSON containers.
                        {  # mutable-ok: Native provider JSON containers.
                            **choice,
                            "delta": {  # mutable-ok: Native provider JSON containers.
                                "tool_calls": [  # mutable-ok: Native provider JSON containers.
                                    {  # mutable-ok: Native provider JSON containers.
                                        **call,
                                        "index": client_indices[index],
                                    }
                                    for call in object_items(object_value(choice.get("delta")).get("tool_calls"))
                                    if isinstance(index := call.get("index"), int) and index in client_indices
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
            for chunk in self.tool_chunks
            for choice in object_items(chunk.get("choices"))
            if any(
                call.get("index") in client_indices
                for call in object_items(object_value(choice.get("delta")).get("tool_calls"))
            )
        )
        return response, chunks

    def accept_response(self, response: Mapping[str, object]) -> None:
        public: Final = {  # mutable-ok: Native provider response JSON.
            **public_tool_response(response, self.route, self.server_names),
            **self.client_response_fields,
        }
        hidden: Final[Mapping[str, object]] = (
            {  # mutable-ok: Native provider JSON containers.
                **public,
                "output": [  # mutable-ok: Native provider JSON containers.
                ],
            }
            if self.route == "aresponses"
            else {  # mutable-ok: Native provider JSON containers.
                **public,
                "content": [  # mutable-ok: Native provider JSON containers.
                ],
                "stop_reason": "end_turn",
            }
            if self.route == "anthropic_messages"
            else {  # mutable-ok: Native provider JSON containers.
                **public,
                "choices": [  # mutable-ok: Native provider JSON containers.
                    {  # mutable-ok: Native provider JSON containers.
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {  # mutable-ok: Native provider JSON containers.
                            "role": "assistant",
                            "content": None,
                        },
                    }
                ],
            }
        )
        self.responses = (*self.responses, hidden if self.suppress_output else public)
        if self.response_id is None:
            self.response_id = str(response.get("id", ""))

    def response(self) -> Mapping[str, object]:
        return {  # mutable-ok: Native provider JSON containers.
            **combined_tool_response(self.responses, self.route),
            "id": self.response_id,
        }

    def finish(self) -> tuple[bytes, ...]:
        response: Final = self.response()
        if self.route == "anthropic_messages":
            return (
                self._emit(
                    {  # mutable-ok: Native provider JSON containers.
                        "type": "message_delta",
                        "delta": {  # mutable-ok: Native provider JSON containers.
                            "stop_reason": response.get("stop_reason"),
                            "stop_sequence": response.get("stop_sequence"),
                        },
                        "usage": response.get(
                            "usage",
                            {  # mutable-ok: Native provider JSON containers.
                            },
                        ),
                    },
                    "message_delta",
                ),
                self._emit(
                    {  # mutable-ok: Native provider JSON containers.
                        "type": "message_stop"
                    },
                    "message_stop",
                ),
            )
        if self.route == "aresponses":
            kind: Final = "response.incomplete" if response.get("status") == "incomplete" else "response.completed"
            return (
                self._emit(
                    {  # mutable-ok: Native provider JSON containers.
                        "type": kind,
                        "response": response,
                    },
                    kind,
                ),
            )
        choices: Final = object_items(response.get("choices"))
        return (
            self._emit(
                {  # mutable-ok: Native provider JSON containers.
                    **{  # mutable-ok: Native provider JSON containers.
                        key: value for key, value in response.items() if key != "choices"
                    },
                    "object": "chat.completion.chunk",
                    "choices": [  # mutable-ok: Native provider JSON containers.
                        {  # mutable-ok: Native provider JSON containers.
                            "index": 0,
                            "delta": dict[str, object](  # mutable-ok: Native provider JSON containers.
                            ),
                            "finish_reason": choices[0].get("finish_reason"),
                        }
                    ],
                }
            ),
            b"data: [DONE]\n\n",
        )

    def error(self, message: str) -> bytes:
        if self.route == "aresponses":
            return self._emit(
                {  # mutable-ok: Native provider JSON containers.
                    "type": "error",
                    "code": "server_error",
                    "message": message,
                    "param": None,
                },
                "error",
            )
        if self.route == "anthropic_messages":
            return self._emit(
                {  # mutable-ok: Native provider JSON containers.
                    "type": "error",
                    "error": {  # mutable-ok: Native provider JSON containers.
                        "type": "api_error",
                        "message": message,
                    },
                },
                "error",
            )
        return sse_bytes(
            {  # mutable-ok: Native provider JSON containers.
                "error": {  # mutable-ok: Native provider JSON containers.
                    "type": "server_error",
                    "message": message,
                    "code": "server_error",
                }
            }
        )
