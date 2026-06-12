"""Parsed Converse stream events -> IR stream events.

Pinned at the parsed-event seam exactly like the characterization corpus: the
binary AWS event-stream framing in front of ``converse_chunk_parser`` is
botocore plumbing, not translation logic. One parsed event maps to at most
one ``StreamEvent``; block-stop bookkeeping and the trailing usage metadata
map to none (v1's wrapper withholds the usage chunk unless stream_options
asks for it). Event shapes the v2 request surface cannot trigger (redacted
reasoning deltas, guardrail traces) are loud error values.
"""

from __future__ import annotations

from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import (
    FinishReason,
    PlainJson,
    SignatureDelta,
    StreamEvent,
    StreamFinish,
    TextDelta,
    ThinkingDelta,
    ToolArgsDelta,
    ToolUseStart,
)
from ...result import Error, Ok, Result
from .params import FINISH_MAP

_EventResult = Result[StreamEvent | None, TranslationError]


def parse_event(event: PlainJson, reverse: dict[str, str]) -> _EventResult:
    if not isinstance(event, dict):
        return Error(_boundary("stream event is not an object"))
    if "trace" in event:
        return Error(
            TranslationError.of_unsupported(
                "guardrail trace stream events; unreachable for v2-sent requests"
            )
        )
    if "start" in event:
        return _start_event(event, reverse)
    if "delta" in event:
        return _delta_event(event)
    if "contentBlockIndex" in event:
        return Ok(None)  # contentBlockStop bookkeeping
    if "stopReason" in event:
        return _finish_event(event)
    if "usage" in event or "metrics" in event:
        return Ok(None)  # metadata: the wrapper withholds the usage chunk
    if event.get("role") == "assistant":
        # messageStart: v1 emits the empty role-stamped chunk here.
        return Ok(StreamEvent.of_text_delta(TextDelta(index=0, text="")))
    return Error(
        TranslationError.of_unsupported(
            f"stream event keys {sorted(event)!r}; unreachable for v2-sent requests"
        )
    )


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def _index_of(event: dict[str, PlainJson]) -> int:
    index = event.get("contentBlockIndex", 0)
    if isinstance(index, bool) or not isinstance(index, int):
        return 0
    return index


def _start_event(event: dict[str, PlainJson], reverse: dict[str, str]) -> _EventResult:
    start = event.get("start")
    if not isinstance(start, dict):
        return Error(_boundary("contentBlockStart is malformed"))
    tool_use = start.get("toolUse")
    if not isinstance(tool_use, dict):
        return Error(
            TranslationError.of_unsupported(
                f"stream block start keys {sorted(start)!r}; unreachable for v2-sent requests"
            )
        )
    identifier = tool_use.get("toolUseId")
    name = tool_use.get("name")
    if not isinstance(identifier, str) or not isinstance(name, str):
        return Error(_boundary("toolUse start is missing 'toolUseId'/'name'"))
    return Ok(
        StreamEvent.of_tool_use_start(
            ToolUseStart(
                index=_index_of(event), id=identifier, name=reverse.get(name, name)
            )
        )
    )


def _delta_event(event: dict[str, PlainJson]) -> _EventResult:
    delta = event.get("delta")
    if not isinstance(delta, dict):
        return Error(_boundary("contentBlockDelta is malformed"))
    index = _index_of(event)
    if "text" in delta:
        text = delta["text"]
        return Ok(
            StreamEvent.of_text_delta(
                TextDelta(index=index, text=text if isinstance(text, str) else "")
            )
        )
    if "toolUse" in delta:
        tool_use = delta["toolUse"]
        partial = tool_use.get("input") if isinstance(tool_use, dict) else None
        return Ok(
            StreamEvent.of_tool_args_delta(
                ToolArgsDelta(
                    index=index,
                    partial_json=partial if isinstance(partial, str) else "",
                )
            )
        )
    if "reasoningContent" in delta:
        return _reasoning_delta(delta["reasoningContent"], index)
    return Error(
        TranslationError.of_unsupported(
            f"stream delta keys {sorted(delta)!r}; unreachable for v2-sent requests"
        )
    )


def _reasoning_delta(payload: PlainJson, index: int) -> _EventResult:
    if not isinstance(payload, dict):
        return Error(_boundary("reasoningContent delta is malformed"))
    if "text" in payload:
        text = payload["text"]
        return Ok(
            StreamEvent.of_thinking_delta(
                ThinkingDelta(
                    index=index, thinking=text if isinstance(text, str) else ""
                )
            )
        )
    if "signature" in payload:
        signature = payload["signature"]
        return Ok(
            StreamEvent.of_signature_delta(
                SignatureDelta(
                    index=index,
                    signature=signature if isinstance(signature, str) else "",
                )
            )
        )
    return Error(
        TranslationError.of_unsupported(
            f"reasoning delta keys {sorted(payload)!r}; unreachable for v2-sent requests"
        )
    )


def _finish_event(event: dict[str, PlainJson]) -> _EventResult:
    stop_reason = event.get("stopReason")
    finish: FinishReason = (
        FINISH_MAP.get(stop_reason, "stop") if isinstance(stop_reason, str) else "stop"
    )
    return Ok(StreamEvent.of_finish(StreamFinish(finish=finish, output_tokens=0)))
