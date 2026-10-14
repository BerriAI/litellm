"""OpenAI's ``prompt_cache_breakpoint`` on a request bound for Claude.

The marks become Anthropic's ``cache_control`` on the same blocks, and a request that carried at least one is held
to Claude's cap of four breakpoints, which a client writing the OpenAI dialect cannot see. ``prompt_cache_options``
has no Claude equivalent and is rejected or dropped like any other unsupported param."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, groupby
from operator import itemgetter
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.constants import MAX_CACHE_CONTROL_BLOCKS
from litellm.exceptions import BadRequestError, UnsupportedParamsError
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    message_fields,
    message_without_cache_control_marks,
    messages_with_prompt_cache_breakpoints_as_cache_control,
    tools_with_prompt_cache_breakpoints_as_cache_control,
)
from litellm.llms.anthropic.common_utils import tool_call_is_rebuilt_as_server_tool_use
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_TOOL_TYPES
from litellm.types.llms.openai import AllMessageValues

PROMPT_CACHE_BREAKPOINT_ON_TOOLS: Final = "_litellm_prompt_cache_breakpoint_on_tools"
_Site: TypeAlias = tuple[Literal["tool", "message", "part", "tool_call", "tool_message"], int, int]
ClaudeWire: TypeAlias = Literal["anthropic", "bedrock_converse"]
_ANTHROPIC_PART_TYPES_BY_ROLE: Final = MappingProxyType(
    {
        "user": frozenset({"text", "image_url", "document", "file"}),
        "tool": frozenset({"text", "image_url", "file"}),
        "function": frozenset({"text", "image_url", "file"}),
        "assistant": frozenset({"text", "thinking", "server_tool_use"}),
    }
)
_CONVERSE_TOOL_BLOCK_KEYS: Final = frozenset({"systemTool", "toolSpec", "cachePoint"})


@dataclass(frozen=True, slots=True)
class ClaudeCacheMarks:
    messages: tuple[AllMessageValues, ...]
    tools: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class CacheBreakpointCapExceeded:
    found: int


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None  # pyright: ignore[reportUnknownVariableType]  # keys are read one by one


def _as_blocks(value: object) -> Sequence[object] | None:
    return value if isinstance(value, (list, tuple)) else None  # pyright: ignore[reportUnknownVariableType]  # blocks are parsed one by one


def _carries(block: object, key: str) -> bool:
    fields: Final = _as_mapping(block)
    return fields is not None and fields.get(key) is not None


def _tool_carries(tool: object, key: str) -> bool:
    fields: Final = _as_mapping(tool)
    return fields is not None and (fields.get(key) is not None or _carries(fields.get("function"), key))


def tools_carry_prompt_cache_breakpoint(tools: object) -> bool:
    return any(_tool_carries(tool, "prompt_cache_breakpoint") for tool in _as_blocks(tools) or ())


def _text(block: Mapping[str, object]) -> str | None:
    text: Final = block.get("text")
    return text if isinstance(text, str) else None


def _renders_message_mark(message: Mapping[str, object], wire: ClaudeWire) -> bool:
    """Claude's message builders keep a message-level mark on string content and on tool results only, and skip it
    where the string itself is skipped: an empty system prompt on both wires, and on Converse an empty user string
    or a blank assistant string."""
    content: Final = message.get("content")
    role: Final = message.get("role")
    if not isinstance(content, str):
        return role in ("tool", "function")
    if wire == "bedrock_converse" and role == "assistant":
        return bool(content.strip())
    return bool(content) or (wire == "anthropic" and role != "system")


def _renders_part_mark(role: object, part: Mapping[str, object], wire: ClaudeWire) -> bool:
    """Whether the builder for ``wire`` keeps ``part``, and so its mark, in the body it sends."""
    part_type: Final = part.get("type")
    if role == "system" and wire == "bedrock_converse":
        return part_type == "text" and bool(_text(part))
    if role == "system":
        return part_type != "text" or bool(_text(part))
    if wire == "bedrock_converse":
        return part_type != "text" or bool((_text(part) or "").strip())
    if role == "assistant" and isinstance(part_type, str) and part_type.endswith("_tool_result"):
        return True
    kept_types: Final = _ANTHROPIC_PART_TYPES_BY_ROLE.get(role) if isinstance(role, str) else None
    return kept_types is None or part_type in kept_types


def _renders_tool_call_mark(message: Mapping[str, object], call: object, wire: ClaudeWire) -> bool:
    """Anthropic rebuilds a tool call answered by a server tool as ``server_tool_use``, which carries no mark."""
    return wire == "bedrock_converse" or not tool_call_is_rebuilt_as_server_tool_use(
        (_as_mapping(call) or {}).get("id"), message.get("provider_specific_fields")
    )


def _part_sites(index: int, message: Mapping[str, object], key: str, wire: ClaudeWire) -> Iterator[_Site]:
    role: Final = message.get("role")
    for position, part in enumerate(_as_blocks(message.get("content")) or ()):
        if (
            (fields := _as_mapping(part)) is not None
            and _carries(fields, key)
            and _renders_part_mark(role, fields, wire)
        ):
            yield ("part", index, position)


def _message_sites(index: int, message: object, key: str, wire: ClaudeWire) -> Iterator[_Site]:
    fields: Final = _as_mapping(message)
    if fields is None:
        return
    if fields.get(key) is not None and _renders_message_mark(fields, wire):
        yield ("message", index, -1)
    yield from _part_sites(index, fields, key, wire)
    for position, call in enumerate(_as_blocks(fields.get("tool_calls")) or ()):
        if _carries(call, key) and _renders_tool_call_mark(fields, call, wire):
            yield ("tool_call", index, position)


def _converse_message_sites(index: int, message: object, key: str) -> Iterator[_Site]:
    """Converse emits a single ``cachePoint`` per tool message, however many of its blocks carry a mark."""
    fields: Final = _as_mapping(message)
    if fields is None or fields.get("role") != "tool":
        yield from _message_sites(index, message, key, "bedrock_converse")
        return
    if fields.get(key) is not None or any(_carries(part, key) for part in _as_blocks(fields.get("content")) or ()):
        yield ("tool_message", index, -1)


def _messages_sites(messages: Sequence[object], key: str, wire: ClaudeWire) -> tuple[_Site, ...]:
    if wire == "bedrock_converse":
        return tuple(
            chain.from_iterable(_converse_message_sites(index, message, key) for index, message in enumerate(messages))
        )
    return tuple(
        chain.from_iterable(_message_sites(index, message, key, wire) for index, message in enumerate(messages))
    )


def _converse_sends_tool(tool: object) -> bool:
    """Whether Converse keeps ``tool`` in ``toolConfig``. Tool search and built-in tools with no function schema
    are skipped there."""
    fields: Final = _as_mapping(tool)
    if fields is None:
        return False
    if "systemTool" in fields:
        return True
    return fields.get("type") not in ANTHROPIC_TOOL_SEARCH_TOOL_TYPES and any(
        key in fields for key in ("function", "input_schema", *_CONVERSE_TOOL_BLOCK_KEYS)
    )


def _converse_sends_tool_mark(tool: object) -> bool:
    """Converse adds a ``cachePoint`` only after a tool it builds a ``toolSpec`` for, and reads only the top-level
    mark."""
    fields: Final = _as_mapping(tool)
    return (
        fields is not None
        and fields.get("cache_control") is not None
        and _converse_sends_tool(fields)
        and not any(key in fields for key in _CONVERSE_TOOL_BLOCK_KEYS)
    )


def converse_appends_tool_config_cache_point(tools: object, injection_points: object) -> bool:
    """Whether Converse appends a ``cachePoint`` to ``toolConfig``: a ``tool_config`` injection point does once the
    request keeps at least one tool."""
    return any(_converse_sends_tool(tool) for tool in _as_blocks(tools) or ()) and any(
        (_as_mapping(point) or {}).get("location") == "tool_config" for point in _as_blocks(injection_points) or ()
    )


def _tool_sites(tools: Sequence[object], wire: ClaudeWire) -> tuple[_Site, ...]:
    if wire == "bedrock_converse":
        return tuple(("tool", index, -1) for index, tool in enumerate(tools) if _converse_sends_tool_mark(tool))
    return tuple(("tool", index, -1) for index, tool in enumerate(tools) if _tool_carries(tool, "cache_control"))


def _without_cache_control(block: Mapping[str, object]) -> Mapping[str, object]:
    return {key: value for key, value in block.items() if key != "cache_control"}


def _tool_without_cache_control(tool: object) -> object:
    fields: Final = _as_mapping(tool)
    if fields is None:
        return tool
    function: Final = _as_mapping(fields.get("function"))
    return {
        **_without_cache_control(fields),
        **({"function": _without_cache_control(function)} if function is not None else {}),
    }


def _message_without(message: AllMessageValues, dropped: Sequence[_Site]) -> AllMessageValues:
    if any(kind == "tool_message" for kind, _, _ in dropped):
        content: Final = _as_blocks(message_fields(message).get("content")) or ()
        return message_without_cache_control_marks(
            message, message_mark=True, parts=frozenset(range(len(content))), tool_calls=frozenset()
        )
    return message_without_cache_control_marks(
        message,
        message_mark=any(kind == "message" for kind, _, _ in dropped),
        parts=frozenset(position for kind, _, position in dropped if kind == "part"),
        tool_calls=frozenset(position for kind, _, position in dropped if kind == "tool_call"),
    )


def claude_cache_marks(
    messages: Sequence[AllMessageValues],
    tools: object,
    *,
    automatic_breakpoint: bool,
    drop_params: bool,
    tools_carried_breakpoint: bool = False,
    tool_config_breakpoint: bool = False,
    wire: ClaudeWire = "anthropic",
) -> ClaudeCacheMarks | CacheBreakpointCapExceeded:
    """The request's ``prompt_cache_breakpoint`` marks renamed to ``cache_control`` and, when it carried any, held to
    Claude's cap: ``drop_params`` keeps the latest four in wire order, otherwise the excess is reported. A request
    that carried none is forwarded as written so the provider's own answer reaches the client.

    ``automatic_breakpoint`` is the request-level ``cache_control`` (Anthropic's automatic caching) and
    ``tool_config_breakpoint`` the ``cachePoint`` Converse appends to ``toolConfig``. Each takes one slot of its own
    and is never dropped. ``tools_carried_breakpoint`` covers tools already mapped to Anthropic's shape, where the
    mark was renamed before this census ran. ``wire`` is the body the marks are rendered into, so the count covers
    only the breakpoints the provider receives."""
    listed_tools: Final = _as_blocks(tools) or ()
    carried_breakpoint: Final = (
        tools_carried_breakpoint
        or tools_carry_prompt_cache_breakpoint(listed_tools)
        or bool(_messages_sites(messages, "prompt_cache_breakpoint", wire))
    )
    if not carried_breakpoint:
        return ClaudeCacheMarks(tuple(messages), tuple(listed_tools))
    renamed_messages: Final = messages_with_prompt_cache_breakpoints_as_cache_control(messages, drop_params)
    renamed_tools: Final = tools_with_prompt_cache_breakpoints_as_cache_control(listed_tools, drop_params)
    tool_sites: Final = _tool_sites(renamed_tools, wire)
    message_sites: Final = _messages_sites(renamed_messages, "cache_control", wire)
    found: Final = len(tool_sites) + len(message_sites) + int(automatic_breakpoint) + int(tool_config_breakpoint)
    if found <= MAX_CACHE_CONTROL_BLOCKS:
        return ClaudeCacheMarks(renamed_messages, renamed_tools)
    if not drop_params:
        return CacheBreakpointCapExceeded(found)
    excess: Final = found - MAX_CACHE_CONTROL_BLOCKS
    dropped_tools: Final = frozenset(index for _, index, _ in tool_sites[:excess])
    dropped_by_message: Final = MappingProxyType(
        {index: tuple(sites) for index, sites in groupby(message_sites[: excess - len(dropped_tools)], itemgetter(1))}
    )
    verbose_logger.debug(
        "Dropping the earliest %d of %d cache breakpoints: Claude accepts at most %d.",
        excess,
        found,
        MAX_CACHE_CONTROL_BLOCKS,
    )
    return ClaudeCacheMarks(
        messages=tuple(
            _message_without(message, dropped_by_message[index]) if index in dropped_by_message else message
            for index, message in enumerate(renamed_messages)
        ),
        tools=tuple(
            _tool_without_cache_control(tool) if index in dropped_tools else tool
            for index, tool in enumerate(renamed_tools)
        ),
    )


def resolved_claude_cache_marks(
    outcome: ClaudeCacheMarks | CacheBreakpointCapExceeded, model: str, llm_provider: str
) -> ClaudeCacheMarks:
    match outcome:
        case ClaudeCacheMarks():
            return outcome
        case CacheBreakpointCapExceeded(found=found):
            raise BadRequestError(
                message=(
                    f"Claude accepts at most {MAX_CACHE_CONTROL_BLOCKS} cache breakpoints, found {found}. "
                    "Set drop_params=True (proxy: litellm_settings: drop_params: true) to keep the latest four."
                ),
                model=model,
                llm_provider=llm_provider,
            )
        case _:
            assert_never(outcome)


def reject_prompt_cache_options(model: str, options: object, drop_params: bool) -> None:
    """``prompt_cache_options`` has already been popped from the request; this decides whether that was allowed."""
    if options is None:
        return
    if drop_params:
        verbose_logger.debug(
            "Dropping prompt_cache_options=%r for %s: Claude caches through cache_control.", options, model
        )
        return
    raise UnsupportedParamsError(
        message=(
            f"{model} does not support prompt_cache_options={options!r}. Claude caches through cache_control "
            "marks (or prompt_cache_breakpoint on the OpenAI dialect). To drop unsupported params, set "
            "`litellm.drop_params = True`."
        ),
        model=model,
        status_code=400,
    )
