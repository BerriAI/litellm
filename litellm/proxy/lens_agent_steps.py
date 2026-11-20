"""What a gateway trace needs so Lens shows an agent's run the way the agent's own trace would.

The gateway only sees model calls. The agent's name comes from the caller's user agent, and a tool step is
rebuilt from the tool result a model call carries in its prompt, paired by call id with the tool call an
earlier reply made.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

_KNOWN_AGENTS: Final[tuple[tuple[str, str], ...]] = (
    ("claude-cli", "claude-code"),
    ("claude-code", "claude-code"),
    ("codex", "codex"),
)


def agent_name(user_agent: str | None) -> str | None:
    if not user_agent:
        return None
    lowered: Final = user_agent.strip().lower()
    known: Final = next((name for marker, name in _KNOWN_AGENTS if marker in lowered), None)
    return known or lowered.split("/")[0].split(" ")[0] or None


@dataclass(frozen=True, slots=True)
class ToolExchange:
    call_id: str
    name: str
    arguments: str
    result: str


class _Function(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str | None = None
    arguments: object = None


class _ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str | None = None
    function: _Function = _Function()


class _Block(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str | None = None
    id: str | None = None
    name: str | None = None
    input: object = None
    tool_use_id: str | None = None
    content: object = None


class _Item(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: str | None = None
    role: str | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: object = None
    output: object = None
    tool_call_id: str | None = None
    tool_calls: tuple[_ToolCall, ...] | None = None
    content: object = None


_Model = TypeVar("_Model", bound=BaseModel)
_ITEMS: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...])
_BLOCKS: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...])


@dataclass(frozen=True, slots=True)
class _Turn:
    """One prompt item: the tool calls it makes, the tool results it carries, and whether the model wrote it."""

    calls: tuple[tuple[str, str, str], ...] = ()
    results: tuple[tuple[str, str], ...] = ()
    from_model: bool = False


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _blocks(content: object) -> tuple[_Block, ...]:
    if not isinstance(content, list):
        return ()
    try:
        raw: Final = _BLOCKS.validate_python(content)
    except ValidationError:
        return ()
    return tuple(block for block in (_parsed(_Block, item) for item in raw) if block is not None)


def _parsed(model: type[_Model], value: object) -> _Model | None:
    try:
        return model.model_validate(value)
    except ValidationError:
        return None


def _turn(item: _Item) -> _Turn:
    if item.type == "function_call":
        calls: Final = ((item.call_id, item.name or "", _text(item.arguments)),) if item.call_id else ()
        return _Turn(calls=calls, from_model=True)
    if item.type == "function_call_output":
        return _Turn(results=((item.call_id, _text(item.output)),) if item.call_id else ())
    blocks: Final = _blocks(item.content)
    if item.role == "assistant":
        return _Turn(
            calls=(
                *(
                    (call.id, call.function.name or "", _text(call.function.arguments))
                    for call in item.tool_calls or ()
                    if call.id
                ),
                *(
                    (block.id, block.name or "", _text(block.input))
                    for block in blocks
                    if block.type == "tool_use" and block.id
                ),
            ),
            from_model=True,
        )
    if item.role == "tool":
        return _Turn(results=((item.tool_call_id, _text(item.content)),) if item.tool_call_id else ())
    return _Turn(
        results=tuple(
            (block.tool_use_id, _text(block.content))
            for block in blocks
            if block.type == "tool_result" and block.tool_use_id
        )
    )


def tool_exchanges(messages: object) -> tuple[ToolExchange, ...]:
    """The tool results that arrived since the model's last turn, each with the call that asked for it.

    A prompt repeats the whole conversation, so only the results after the last model turn are new to this
    call. Earlier ones were reported by the call that first carried them.
    """
    if not isinstance(messages, list):
        return ()
    try:
        raw: Final = _ITEMS.validate_python(messages)
    except ValidationError:
        return ()
    turns: Final = tuple(_turn(item) for item in (_parsed(_Item, value) for value in raw) if item is not None)
    last_model_turn: Final = max((index for index, turn in enumerate(turns) if turn.from_model), default=-1)
    calls: Final[Mapping[str, tuple[str, str]]] = {
        call_id: (name, arguments) for turn in turns for call_id, name, arguments in turn.calls
    }
    return tuple(
        ToolExchange(call_id=call_id, name=calls[call_id][0] or "tool", arguments=calls[call_id][1], result=result)
        for turn in turns[last_model_turn + 1 :]
        for call_id, result in turn.results
        if call_id in calls
    )


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    user_agent: str | None = None


class _Payload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    metadata: _Metadata = _Metadata()
    messages: object = None


@dataclass(frozen=True, slots=True)
class AgentCall:
    """What one model call says about the agent that made it."""

    agent: str | None = None
    tool_exchanges: tuple[ToolExchange, ...] = ()


def agent_call(standard_logging_payload: object) -> AgentCall:
    payload: Final = _parsed(_Payload, standard_logging_payload)
    if payload is None:
        return AgentCall()
    return AgentCall(agent=agent_name(payload.metadata.user_agent), tool_exchanges=tool_exchanges(payload.messages))
