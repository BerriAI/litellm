"""OpenAI's ``prompt_cache_breakpoint`` on a chat request bound for Claude, rewritten as Claude's ``cache_control``."""

from bisect import bisect_left
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import accumulate, chain
from types import MappingProxyType
from typing import Final, Literal, NoReturn, TypeAlias, cast  # noqa: TID251  # isinstance-checked hot-path JSON

from litellm._logging import verbose_logger
from litellm.constants import BEDROCK_COMPUTER_USE_TOOLS, MAX_CACHE_CONTROL_BLOCKS
from litellm.exceptions import UnsupportedParamsError
from litellm.responses.litellm_completion_transformation.tool_search import unanswered_tool_search_call_ids
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_TOOL_TYPES
from litellm.types.llms.openai import ChatCompletionCachedContent

_Path: TypeAlias = tuple[str | int, ...]
_Fields: TypeAlias = Mapping[str, object]
_Mark: TypeAlias = Callable[[_Path, _Fields], _Fields]
_Kind: TypeAlias = Literal["written", "explicit", "free"]
ClaudeRoute: TypeAlias = Literal["anthropic", "bedrock_invoke", "converse"]
_CLAUDE_ROUTES: Final[tuple[ClaudeRoute, ...]] = ("anthropic", "bedrock_invoke", "converse")
_TOOL_TYPES_SENT_UNMARKED: Final = MappingProxyType(
    {
        "anthropic": frozenset({"mcp", "namespace", "url"}),
        "bedrock_invoke": frozenset({"mcp", "namespace", "url", "tool_search_tool_bm25_20251119"}),
    }
)
_MESSAGE_BLOCK_KEYS: Final = ("content", "tool_calls")
_TOOL_RESULT_ROLES: Final = ("tool", "function")
_SYSTEM_ROLES: Final = ("system", "developer")
_SYSTEM_MOVES_WITHIN_ROLES: Final = (*_SYSTEM_ROLES, "user", *_TOOL_RESULT_ROLES)
_REQUEST: Final[_Path] = ("cache_control",)
_TOOL_CONFIG: Final[_Path] = ("tool_config",)
_CLAUDE_TTL_MARKS: Final = MappingProxyType(
    {
        "5m": ChatCompletionCachedContent(type="ephemeral", ttl="5m"),
        "1h": ChatCompletionCachedContent(type="ephemeral", ttl="1h"),
    }
)


@dataclass(frozen=True, slots=True)
class ClaudePromptCache:
    messages: tuple[object, ...]
    tools: tuple[object, ...] | None


@dataclass(frozen=True, slots=True)
class UnsupportedPromptCacheParam:
    param: str
    value: object
    reason: str
    fallback: ChatCompletionCachedContent | None


@dataclass(frozen=True, slots=True)
class _Placed:
    path: _Path
    ttl: Literal["5m", "1h"]
    kind: _Kind
    first_read: tuple[int, ...]
    last_read: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Plan:
    dropped: frozenset[_Path]
    replacements: Mapping[_Path, ChatCompletionCachedContent]


@dataclass(frozen=True, slots=True)
class _Hoisted:
    value: object
    replaced: tuple[object, ...]


def _fields(value: object) -> _Fields | None:
    return cast(_Fields, value) if isinstance(value, dict) else None  # cast-ok: request JSON has string keys


def _blocks(value: object) -> Sequence[object] | None:
    return cast(Sequence[object], value) if isinstance(value, list) else None  # cast-ok: elements checked per use


def _ttl_mark(ttl: object, param: str, value: object) -> ChatCompletionCachedContent | UnsupportedPromptCacheParam:
    if ttl is None:
        return ChatCompletionCachedContent(type="ephemeral")
    known: Final = _CLAUDE_TTL_MARKS.get(ttl) if isinstance(ttl, str) else None
    return (
        ChatCompletionCachedContent(**known)
        if known is not None
        else UnsupportedPromptCacheParam(
            param,
            value,
            "Claude caches for a ttl of 5m or 1h only",
            fallback=ChatCompletionCachedContent(type="ephemeral"),
        )
    )


def _breakpoint_mark(marker: object) -> ChatCompletionCachedContent | UnsupportedPromptCacheParam:
    fields: Final = _fields(marker)
    if fields is None or fields.get("mode") != "explicit":
        return UnsupportedPromptCacheParam(
            "prompt_cache_breakpoint", marker, 'expected {"mode": "explicit"} with an optional ttl', fallback=None
        )
    return _ttl_mark(fields.get("ttl"), "prompt_cache_breakpoint", marker)


def _indexed_fields(values: Sequence[object]) -> Iterator[tuple[int, _Fields]]:
    return ((index, fields) for index, fields in enumerate(map(_fields, values)) if fields is not None)


def _markable_in_message(index: int, message: _Fields) -> Iterator[tuple[_Path, _Fields]]:
    yield ("messages", index), message
    for key in _MESSAGE_BLOCK_KEYS:
        yield from (
            (("messages", index, key, j), block) for j, block in _indexed_fields(_blocks(message.get(key)) or ())
        )


def _markable_in_tool(index: int, tool: _Fields) -> Iterator[tuple[_Path, _Fields]]:
    yield ("tools", index), tool
    function: Final = _fields(tool.get("function"))
    if function is not None:
        yield ("tools", index, "function"), function


def _carries_breakpoint(value: object) -> bool:
    fields: Final = _fields(value)
    return fields is not None and fields.get("prompt_cache_breakpoint") is not None


def _any_block_carries_breakpoint(blocks: object) -> bool:
    return any(map(_carries_breakpoint, _blocks(blocks) or ()))


def _message_carries_breakpoint(message: object) -> bool:
    fields: Final = _fields(message) or {}
    return (
        fields.get("prompt_cache_breakpoint") is not None
        or _any_block_carries_breakpoint(fields.get("content"))
        or _any_block_carries_breakpoint(fields.get("tool_calls"))
    )


def _tool_carries_breakpoint(tool: object) -> bool:
    return _carries_breakpoint(tool) or _carries_breakpoint((_fields(tool) or {}).get("function"))


def _namespace_members(tools: Sequence[object]) -> Iterator[tuple[_Path, _Fields]]:
    namespaces: Final = ((index, tool) for index, tool in _indexed_fields(tools) if tool.get("type") == "namespace")
    for index, namespace in namespaces:
        yield from (
            (("tools", index, "tools", member_index), member)
            for member_index, member in _indexed_fields(_blocks(namespace.get("tools")) or ())
        )


def _function_mark(tool: _Fields) -> _Fields | None:
    return _fields((_fields(tool.get("function")) or {}).get("cache_control"))


def _member_mark(member: _Fields) -> object:
    own: Final = member.get("cache_control")
    return own if own is not None else _function_mark(member)


def _transform_sends_member(member: _Fields) -> bool:
    return "input_schema" in member or "function" in member or "name" in member


def _namespace_member_marks(tools: Sequence[object], route: ClaudeRoute) -> tuple[tuple[_Path, object, _Kind], ...]:
    if route == "converse":
        return ()
    return tuple(
        (path, mark, "written")
        for path, member in _namespace_members(tools)
        if _transform_sends_member(member)
        and tool_mark_reaches_claude(member, route)
        and (mark := _member_mark(member)) is not None
    )


def _markable(messages: Sequence[object], tools: Sequence[object]) -> Iterator[tuple[_Path, _Fields]]:
    """Every block that can carry a cache mark, in the order Claude renders them: tool definitions, then messages."""
    for index, tool in _indexed_fields(tools):
        yield from _markable_in_tool(index, tool)
    for index, message in _indexed_fields(messages):
        yield from _markable_in_message(index, message)


def _marked_block(path: _Path, block: object, mark: _Mark) -> object:
    fields: Final = _fields(block)
    return block if fields is None else mark(path, fields)


def _marked_blocks(message: _Fields, path: _Path, key: str, mark: _Mark) -> _Fields:
    blocks: Final = _blocks(message.get(key))
    if blocks is None:
        return {}
    return {key: [_marked_block((*path, key, j), block, mark) for j, block in enumerate(blocks)]}


def _without_breakpoint(block: object) -> object:
    fields: Final = _fields(block)
    return (
        block if fields is None else {key: value for key, value in fields.items() if key != "prompt_cache_breakpoint"}
    )


def _without_own_breakpoint(message: _Fields) -> _Fields:
    return {key: value for key, value in message.items() if key != "prompt_cache_breakpoint"}


def _block_mark(block: object, key: str) -> object:
    return (_fields(block) or {}).get(key)


def _is_supported(marker: object) -> bool:
    return not isinstance(_breakpoint_mark(marker), UnsupportedPromptCacheParam)


def _asks_for_an_hour(marker: object) -> bool:
    mark: Final = _breakpoint_mark(marker)
    return not isinstance(mark, UnsupportedPromptCacheParam) and mark.get("ttl") == "1h"


def _joined_breakpoint(kept: object, replaced: Sequence[object]) -> object:
    """The breakpoint that stays takes 1h when one it replaces asks for 1h, so joining breakpoints never shortens a
    ttl the caller chose."""
    fields: Final = _fields(kept)
    if fields is None or fields.get("ttl") not in (None, "5m") or not any(map(_asks_for_an_hour, replaced)):
        return kept
    return {**fields, "ttl": "1h"}


def prompt_cache_breakpoint_is_usable(marker: object) -> bool:
    mark: Final = _breakpoint_mark(marker)
    return not isinstance(mark, UnsupportedPromptCacheParam) or mark.fallback is not None


def _is_computer_use_tool(tool: object) -> bool:
    tool_type: Final = (_fields(tool) or {}).get("type")
    return isinstance(tool_type, str) and tool_type.startswith(BEDROCK_COMPUTER_USE_TOOLS)


def _converse_sends_tool_mark(tool: _Fields) -> bool:
    tool_type: Final = tool.get("type")
    if tool_type in ANTHROPIC_TOOL_SEARCH_TOOL_TYPES:
        return False
    if isinstance(tool_type, str) and _is_computer_use_tool(tool):
        return not (tool_type.startswith("computer_") and "function" in tool)
    return "function" in tool or "input_schema" in tool


def tool_mark_reaches_claude(tool: object, route: ClaudeRoute) -> bool:
    fields: Final = _fields(tool) or {}
    match route:
        case "converse":
            return _converse_sends_tool_mark(fields)
        case "anthropic" | "bedrock_invoke":
            return fields.get("type") not in _TOOL_TYPES_SENT_UNMARKED[route]


def tool_mark_may_reach_claude(tool: object) -> bool:
    return any(tool_mark_reaches_claude(tool, route) for route in _CLAUDE_ROUTES)


def _kept_breakpoint(markers: Sequence[object]) -> tuple[object, tuple[object, ...]]:
    """The first breakpoint Claude takes as sent stays, else the first one ``drop_params`` can fix, so a malformed one
    or one whose ttl ``drop_params`` drops never takes the place of a valid one and costs the ttl it chose."""
    supported: Final = (index for index, marker in enumerate(markers) if _is_supported(marker))
    usable: Final = (index for index, marker in enumerate(markers) if prompt_cache_breakpoint_is_usable(marker))
    index: Final = next(chain(supported, usable), 0)
    return markers[index], (*markers[:index], *markers[index + 1 :])


def _tool_result_hoisted(message: _Fields) -> _Hoisted:
    """Claude rejects a tool result marked both on the result and inside its content, or marked twice inside it, so
    a ``cache_control`` the client wrote at either level wins over every breakpoint, and the result's breakpoints
    become one on the message. Every breakpoint that loses is still checked."""
    parts: Final = _blocks(message.get("content"))
    if parts is None:
        return _Hoisted(message, ())
    own: Final = message.get("prompt_cache_breakpoint")
    unmarked_parts: Final = [_without_breakpoint(part) for part in parts]
    part_breakpoints: Final = tuple(
        marker for marker in (_block_mark(part, "prompt_cache_breakpoint") for part in parts) if marker is not None
    )
    if message.get("cache_control") is not None:
        return _Hoisted({**message, "content": unmarked_parts}, part_breakpoints)
    if any(_block_mark(part, "cache_control") is not None for part in parts):
        return _Hoisted(
            {**_without_own_breakpoint(message), "content": unmarked_parts},
            part_breakpoints if own is None else (own, *part_breakpoints),
        )
    if own is None and not part_breakpoints:
        return _Hoisted(message, ())
    kept, replaced = _kept_breakpoint(part_breakpoints if own is None else (own, *part_breakpoints))
    return _Hoisted(
        {**message, "prompt_cache_breakpoint": _joined_breakpoint(kept, replaced), "content": unmarked_parts}, replaced
    )


def _last_block(message: _Fields) -> tuple[str, Sequence[object], _Fields] | None:
    """The block that ends a message's content, or its last tool call when it has no content."""
    keyed: Final = ((key, _blocks(message.get(key))) for key in _MESSAGE_BLOCK_KEYS)
    found: Final = next(((key, blocks) for key, blocks in keyed if blocks), None)
    if found is None:
        return None
    key, blocks = found
    last: Final = _fields(blocks[-1])
    return None if last is None else (key, blocks, last)


def _message_breakpoint_hoisted(message: _Fields) -> _Hoisted:
    """Claude reads a mark on a block, so a message's own breakpoint moves to the block that ends its content, or to
    its last tool call when it has none. String content keeps it, except on a developer message, whose message-level
    keys litellm drops when it maps the role to system."""
    own: Final = message.get("prompt_cache_breakpoint")
    content: Final = message.get("content")
    if own is None or message.get("cache_control") is not None:
        return _Hoisted(message, ())
    if isinstance(content, str):
        return (
            _Hoisted(
                {
                    **_without_own_breakpoint(message),
                    "content": [{"type": "text", "text": content, "prompt_cache_breakpoint": own}],
                },
                (),
            )
            if message.get("role") == "developer"
            else _Hoisted(message, ())
        )
    last: Final = _last_block(message)
    if last is None:
        return _Hoisted(_without_own_breakpoint(message), (own,))
    key, blocks, block = last
    if block.get("cache_control") is not None:
        return _Hoisted(_without_own_breakpoint(message), (own,))
    existing: Final = block.get("prompt_cache_breakpoint")
    kept, replaced = (own, ()) if existing is None else _kept_breakpoint((existing, own))
    return _Hoisted(
        {
            **_without_own_breakpoint(message),
            key: [*blocks[:-1], {**block, "prompt_cache_breakpoint": _joined_breakpoint(kept, replaced)}],
        },
        replaced,
    )


def _hoisted_message(message: object) -> _Hoisted:
    fields: Final = _fields(message)
    if fields is None:
        return _Hoisted(message, ())
    return (
        _tool_result_hoisted(fields)
        if fields.get("role") in _TOOL_RESULT_ROLES
        else _message_breakpoint_hoisted(fields)
    )


def _marked_message(index: int, message: object, mark: _Mark) -> object:
    fields: Final = _fields(message)
    if fields is None:
        return message
    marked: Final = mark(("messages", index), fields)
    return {
        **marked,
        **_marked_blocks(marked, ("messages", index), "content", mark),
        **_marked_blocks(marked, ("messages", index), "tool_calls", mark),
    }


def _hoisted_tool(tool: object) -> _Hoisted:
    """Claude sends one mark per tool and Converse reads only the top-level one, so a breakpoint written under
    ``function`` moves up and joins the tool's own, a ``cache_control`` the client wrote at either level wins over
    every breakpoint, and every breakpoint that loses is still checked."""
    fields: Final = _fields(tool)
    function: Final = None if fields is None else _fields(fields.get("function"))
    if fields is None or function is None:
        return _Hoisted(tool, ())
    function_marker: Final = function.get("prompt_cache_breakpoint")
    tool_marker: Final = fields.get("prompt_cache_breakpoint")
    if function_marker is None and (tool_marker is None or function.get("cache_control") is None):
        return _Hoisted(tool, ())
    markers: Final = tuple(marker for marker in (tool_marker, function_marker) if marker is not None)
    unmarked: Final = {
        **_without_own_breakpoint(fields),
        "function": _without_own_breakpoint(function),
    }
    if function.get("cache_control") is not None or fields.get("cache_control") is not None:
        return _Hoisted(unmarked, markers)
    kept, replaced = _kept_breakpoint(markers)
    return _Hoisted({**unmarked, "prompt_cache_breakpoint": _joined_breakpoint(kept, replaced)}, replaced)


def _marked_tool(index: int, tool: object, mark: _Mark) -> object:
    fields: Final = _fields(tool)
    if fields is None:
        return tool
    marked: Final = mark(("tools", index), fields)
    function: Final = _fields(marked.get("function"))
    return marked if function is None else {**marked, "function": mark(("tools", index, "function"), function)}


def _marker_problem(marker: object) -> UnsupportedPromptCacheParam | None:
    mark: Final = _breakpoint_mark(marker)
    return mark if isinstance(mark, UnsupportedPromptCacheParam) else None


def _breakpoint_problem(block: _Fields) -> UnsupportedPromptCacheParam | None:
    marker: Final = block.get("prompt_cache_breakpoint")
    return None if marker is None else _marker_problem(marker)


def _renamed(_path: _Path, block: _Fields) -> _Fields:
    """A ``cache_control`` the client already wrote on the block wins over the breakpoint."""
    marker: Final = block.get("prompt_cache_breakpoint")
    if marker is None:
        return block
    unmarked: Final = {key: value for key, value in block.items() if key != "prompt_cache_breakpoint"}
    if unmarked.get("cache_control") is not None:
        return unmarked
    mark: Final = _breakpoint_mark(marker)
    cache_control: Final = mark.fallback if isinstance(mark, UnsupportedPromptCacheParam) else mark
    return unmarked if cache_control is None else {**unmarked, "cache_control": cache_control}


def _marked_request(
    messages: Sequence[object], tools: Sequence[object], mark: _Mark
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    return (
        tuple(_marked_message(index, message, mark) for index, message in enumerate(messages)),
        tuple(_marked_tool(index, tool, mark) for index, tool in enumerate(tools)),
    )


def _settled_marks(dropped: frozenset[_Path], replacements: Mapping[_Path, ChatCompletionCachedContent]) -> _Mark:
    def mark(path: _Path, block: _Fields) -> _Fields:
        if path in dropped:
            return {key: value for key, value in block.items() if key != "cache_control"}
        replacement: Final = replacements.get(path)
        return block if replacement is None else {**block, "cache_control": replacement}

    return mark


def _unchanged(_path: _Path, block: _Fields) -> _Fields:
    return block


def _translated(messages: Sequence[object], tools: Sequence[object]) -> frozenset[_Path]:
    """Blocks whose mark comes from a breakpoint rather than a ``cache_control`` the client wrote."""
    hoisted_messages, hoisted_tools = _marked_request(messages, tools, _unchanged)
    return frozenset(
        path
        for path, block in _markable(hoisted_messages, hoisted_tools)
        if block.get("prompt_cache_breakpoint") is not None and block.get("cache_control") is None
    )


def _role(message: object) -> object:
    return (_fields(message) or {}).get("role")


def _leading_system_count(messages: Sequence[object]) -> int:
    return next((index for index, message in enumerate(messages) if _role(message) not in _SYSTEM_ROLES), len(messages))


def _wire_position(path: _Path, leading_system_count: int, late_tools: frozenset[int]) -> tuple[int, ...]:
    """Where Claude reads a mark: tool definitions, then the cachePoint a route appends after them, then the
    ``late_tools`` a route sends apart from the others, then the leading system messages it hoists, then every other
    message in place, with a message-level mark on the message's text, after its content parts and before its tool
    calls, and last the request-level mark."""
    if path == _TOOL_CONFIG:
        return (1,)
    if path == _REQUEST:
        return (4,)
    if path[0] == "tools":
        indices: Final = tuple(index for index in path if isinstance(index, int))
        return (1, 1, *indices) if indices[0] in late_tools else (0, *indices)
    index: Final = int(path[1])
    group: Final = 2 if index < leading_system_count else 3
    block: Final = (1,) if len(path) == 2 else (0 if path[2] == "content" else 2, int(path[3]))
    return (group, index, *block)


def _run_ends(messages: Sequence[object]) -> tuple[int, ...]:
    """For each index, the first message at or after it that a moving system message never passes."""
    return tuple(
        reversed(
            tuple(
                accumulate(
                    reversed(range(len(messages))),
                    lambda end, index: end if _role(messages[index]) in _SYSTEM_MOVES_WITHIN_ROLES else index,
                    initial=len(messages),
                )
            )
        )
    )


def _last_read(
    path: _Path,
    messages: Sequence[object],
    leading_system_count: int,
    run_ends: Sequence[int],
    late_tools: frozenset[int],
) -> tuple[int, ...]:
    """The transforms may move a system message after the conversation start behind the user, tool and function
    turns that follow it, never past the next assistant turn, so Claude may read its marks as late as the end of
    those turns."""
    position: Final = _wire_position(path, leading_system_count, late_tools)
    index: Final = int(path[1]) if path[0] == "messages" else -1
    if index < leading_system_count or _role(messages[index]) not in _SYSTEM_ROLES:
        return position
    return (3, run_ends[index + 1] - 1, 3)


def _tool_call_id(message: _Fields, index: int) -> object:
    return (_fields((_blocks(message.get("tool_calls")) or ())[index]) or {}).get("id")


def _is_tool_history(message: _Fields) -> bool:
    role: Final = message.get("role")
    return role in ("tool", "function") or (role == "assistant" and bool(message.get("tool_calls")))


def _on_wire(
    path: _Path,
    messages: Sequence[object],
    tools: Sequence[object],
    unsent_call_ids: frozenset[str],
    tool_history_as_text: bool,
    route: ClaudeRoute,
) -> bool:
    """Claude sends a message-level mark on string content, or on list content only for a tool result, and none on
    a developer message, which reaches it as a system message without one; a tool's mark only when the route sends
    it, and one under ``function`` only off Converse, as an object, and when the tool carries none of its own; no
    tool_search call nobody answered; and no mark on tool history the route sends as text."""
    if path[0] == "tools":
        tool: Final = _fields(tools[int(path[1])]) or {}
        return tool_mark_reaches_claude(tool, route) and (
            len(path) == 2
            or (route != "converse" and tool.get("cache_control") is None and _function_mark(tool) is not None)
        )
    message: Final = _fields(messages[int(path[1])]) or {}
    if tool_history_as_text and _is_tool_history(message):
        return False
    if len(path) == 4:
        return path[2] != "tool_calls" or _tool_call_id(message, int(path[3])) not in unsent_call_ids
    role: Final = message.get("role")
    return role in _TOOL_RESULT_ROLES or (role != "developer" and isinstance(message.get("content"), str))


def _ttl(cache_control: object) -> Literal["5m", "1h"]:
    return "1h" if (_fields(cache_control) or {}).get("ttl") == "1h" else "5m"


def _kind(path: _Path, cache_control: object, translated: frozenset[_Path]) -> _Kind:
    if path not in translated:
        return "written"
    return "free" if (_fields(cache_control) or {}).get("ttl") is None else "explicit"


def _chosen(mark: _Placed, ttl: Literal["5m", "1h"]) -> bool:
    return mark.kind != "free" and mark.ttl == ttl


def _first_five(marks: Sequence[_Placed]) -> tuple[int, ...]:
    return min((mark.first_read for mark in marks if _chosen(mark, "5m")), default=(5,))


def _hour_reach(
    hour: _Placed, free_starts: Sequence[tuple[int, ...]], free_latest: Sequence[tuple[int, ...]]
) -> tuple[int, ...]:
    read_before: Final = bisect_left(free_starts, hour.last_read)
    return hour.last_read if read_before == 0 else max(hour.last_read, free_latest[read_before - 1])


def _hour_reaches(marks: Sequence[_Placed]) -> Mapping[_Path, tuple[int, ...]]:
    """A 1h mark makes every breakpoint without a ttl that Claude may read before it take 1h too, so its order with a
    5m mark holds as late as the last of them may be read."""
    free: Final = sorted((mark for mark in marks if mark.kind == "free"), key=lambda mark: mark.first_read)
    starts: Final = tuple(mark.first_read for mark in free)
    latest: Final = tuple(accumulate((mark.last_read for mark in free), max))
    return MappingProxyType({hour.path: _hour_reach(hour, starts, latest) for hour in marks if _chosen(hour, "1h")})


def _loosened(marks: Sequence[_Placed], drop_params: bool) -> tuple[_Placed, ...]:
    """Claude rejects a 1h mark read after a 5m one."""
    if not drop_params:
        return tuple(marks)
    first_five: Final = _first_five(marks)
    reaches: Final = _hour_reaches(marks)
    last_hour: Final = max(reaches.values(), default=(-1,))
    return tuple(
        replace(mark, ttl="5m", kind="free")
        if mark.kind == "explicit"
        and (reaches[mark.path] > first_five if mark.ttl == "1h" else mark.first_read < last_hour)
        else mark
        for mark in marks
    )


def _settled(marks: Sequence[_Placed], drop_params: bool) -> tuple[_Placed, ...]:
    """A breakpoint without a ttl takes 1h when Claude may read a 1h mark after it and no 5m mark before it, so the
    request stays valid for Claude; under ``drop_params`` a ttl the client set on a breakpoint that breaks that order
    is dropped first."""
    loosened: Final = _loosened(marks, drop_params)
    first_five: Final = _first_five(loosened)
    last_hour: Final = max((mark.last_read for mark in loosened if _chosen(mark, "1h")), default=(-1,))
    return tuple(
        replace(mark, ttl="1h")
        if mark.kind == "free" and mark.first_read < last_hour and mark.last_read < first_five
        else mark
        for mark in loosened
    )


def _route_marks(
    tool_config_cache_control: ChatCompletionCachedContent | None, request_mark: object
) -> tuple[tuple[_Path, object, _Kind], ...]:
    """The cachePoint a route appends after the tools takes 1h like a breakpoint when it sets no ttl. Neither it nor
    the request-level mark is ever dropped."""
    tool_config_kind: Final[_Kind] = (
        "free" if tool_config_cache_control is None or tool_config_cache_control.get("ttl") is None else "written"
    )
    return (
        *(() if tool_config_cache_control is None else ((_TOOL_CONFIG, tool_config_cache_control, tool_config_kind),)),
        *(() if request_mark is None else ((_REQUEST, request_mark, "written"),)),
    )


def _droppable(mark: _Placed) -> bool:
    return mark.kind != "written" and mark.path != _TOOL_CONFIG


def _planned(marks: Sequence[_Placed], drop_params: bool) -> _Plan:
    excess: Final = len(marks) - MAX_CACHE_CONTROL_BLOCKS
    dropped: Final[frozenset[_Path]] = (
        frozenset(tuple(mark.path for mark in marks if _droppable(mark))[:excess])
        if excess > 0 and drop_params
        else frozenset()
    )
    kept: Final = tuple(mark for mark in marks if mark.path not in dropped)
    return _Plan(
        dropped,
        MappingProxyType(
            {
                settled.path: (
                    ChatCompletionCachedContent(type="ephemeral", ttl="1h")
                    if settled.ttl == "1h"
                    else ChatCompletionCachedContent(type="ephemeral")
                )
                for mark, settled in zip(kept, _settled(kept, drop_params))
                if settled.ttl != mark.ttl
            }
        ),
    )


def _in_converse_tool_config(tool: _Fields) -> bool:
    return "toolSpec" in tool or "cachePoint" in tool or _converse_sends_tool_mark(tool)


def _last_tool_in_tool_config(tools: Sequence[object], late_tools: frozenset[int]) -> _Path | None:
    sent: Final = tuple(
        index
        for index, tool in enumerate(tools)
        if index not in late_tools and _in_converse_tool_config(_fields(tool) or {})
    )
    return ("tools", sent[-1]) if sent else None


def _sets_a_ttl(cache_control: object) -> bool:
    return (_fields(cache_control) or {}).get("ttl") is not None


def _planned_cache_control(plan: _Plan, path: _Path, cache_control: object) -> object:
    return None if path in plan.dropped else plan.replacements.get(path, cache_control)


def _route_plan(
    marks: Sequence[_Placed],
    tools: Sequence[object],
    last_tool: _Path | None,
    tool_config_cache_control: ChatCompletionCachedContent | None,
    drop_params: bool,
) -> _Plan:
    """Converse leaves out a ``tool_config`` cachePoint with a ttl that would follow the last tool's mark with a ttl,
    so that request is planned again without it, keeping the last tool's mark as first planned."""
    plan: Final = _planned(marks, drop_params)
    if last_tool is None:
        return plan
    last_tool_mark: Final = _planned_cache_control(
        plan, last_tool, (_fields(tools[int(last_tool[1])]) or {}).get("cache_control")
    )
    if not (
        _sets_a_ttl(last_tool_mark)
        and _sets_a_ttl(_planned_cache_control(plan, _TOOL_CONFIG, tool_config_cache_control))
    ):
        return plan
    replanned: Final = _planned(
        tuple(
            replace(mark, ttl=_ttl(last_tool_mark), kind="written") if mark.path == last_tool else mark
            for mark in marks
            if mark.path != _TOOL_CONFIG
        ),
        drop_params,
    )
    last_tool_replacement: Final = plan.replacements.get(last_tool)
    return _Plan(
        replanned.dropped,
        MappingProxyType(
            {
                **replanned.replacements,
                **({} if last_tool_replacement is None else {last_tool: last_tool_replacement}),
            }
        ),
    )


def carries_prompt_cache_breakpoint(messages: Sequence[object], tools: Sequence[object] | None) -> bool:
    return any(map(_message_carries_breakpoint, messages)) or any(map(_tool_carries_breakpoint, tools or ()))


def claude_prompt_cache(
    messages: Sequence[object],
    tools: Sequence[object] | None,
    params: _Fields,
    *,
    drop_params: bool,
    tool_history_as_text: bool,
    tool_config_cache_control: ChatCompletionCachedContent | None,
    route: ClaudeRoute,
    pairs_unanswered_tool_calls: bool,
) -> ClaudePromptCache | UnsupportedPromptCacheParam:
    """``prompt_cache_breakpoint`` marks renamed to ``cache_control``. A mark Claude cannot honor is reported unless
    ``drop_params``, which keeps what Claude can use and, when the request ends up with more than Claude's four
    breakpoints, drops the earliest translated ones. A request-level ``cache_control`` in ``params`` and the
    ``tool_config_cache_control`` cachePoint a route appends after the tools each take one of the four, unless Converse
    leaves that cachePoint out, and marks on tool history a ``tool_history_as_text`` route flattens, marks on a tool the
    ``route`` sends without one or under a tool's ``function`` on Converse, and marks on a tool_search call nobody
    answered, unless the route ``pairs_unanswered_tool_calls`` with a placeholder result, take none. A ``cache_control``
    on a ``namespace`` tool's member takes one off Converse and is never changed. A request carrying no breakpoint is
    returned as written."""
    if not carries_prompt_cache_breakpoint(messages, tools):
        return ClaudePromptCache(tuple(messages), None if tools is None else tuple(tools))
    listed_tools: Final = tools or ()
    hoists: Final = (*map(_hoisted_message, messages), *map(_hoisted_tool, listed_tools))
    hoisted_messages: Final = tuple(hoist.value for hoist in hoists[: len(messages)])
    hoisted_tools: Final = tuple(hoist.value for hoist in hoists[len(messages) :])
    blocks: Final = tuple(
        block for _, block in _markable(*_marked_request(hoisted_messages, hoisted_tools, _unchanged))
    )
    problem: Final = next(
        filter(
            None,
            chain(
                map(_breakpoint_problem, blocks),
                map(_marker_problem, chain.from_iterable(hoist.replaced for hoist in hoists)),
            ),
        ),
        None,
    )
    if problem is not None and not drop_params:
        return problem
    if problem is not None:
        verbose_logger.debug("Claude: adjusting %s=%r because %s", problem.param, problem.value, problem.reason)
    renamed_messages, renamed_tools = _marked_request(hoisted_messages, hoisted_tools, _renamed)
    translated: Final = _translated(hoisted_messages, hoisted_tools)
    leading_system_count: Final = _leading_system_count(messages)
    run_ends: Final = _run_ends(messages)
    late_tools: Final = frozenset(
        index for index, tool in enumerate(renamed_tools) if route == "converse" and _is_computer_use_tool(tool)
    )
    unsent_call_ids: Final = frozenset() if pairs_unanswered_tool_calls else unanswered_tool_search_call_ids(messages)
    request_mark: Final = params.get("cache_control")
    placements: Final[tuple[tuple[_Path, object, _Kind], ...]] = (
        *(
            (path, block.get("cache_control"), _kind(path, block.get("cache_control"), translated))
            for path, block in _markable(renamed_messages, renamed_tools)
            if block.get("cache_control") is not None
            and _on_wire(path, renamed_messages, renamed_tools, unsent_call_ids, tool_history_as_text, route)
        ),
        *_namespace_member_marks(renamed_tools, route),
        *_route_marks(tool_config_cache_control, request_mark),
    )
    marks: Final = sorted(
        (
            _Placed(
                path,
                _ttl(cache_control),
                kind,
                _wire_position(path, leading_system_count, late_tools),
                _last_read(path, messages, leading_system_count, run_ends, late_tools),
            )
            for path, cache_control, kind in placements
        ),
        key=lambda mark: mark.first_read,
    )
    plan: Final = _route_plan(
        marks,
        renamed_tools,
        _last_tool_in_tool_config(renamed_tools, late_tools),
        tool_config_cache_control,
        drop_params,
    )
    if plan.dropped:
        verbose_logger.debug("Claude: dropping the earliest %d prompt_cache_breakpoint marks", len(plan.dropped))
    final_messages, final_tools = _marked_request(
        renamed_messages, renamed_tools, _settled_marks(plan.dropped, plan.replacements)
    )
    return ClaudePromptCache(final_messages, None if tools is None else final_tools)


def resolved_claude_prompt_cache(
    outcome: ClaudePromptCache | UnsupportedPromptCacheParam, model: str, llm_provider: str
) -> ClaudePromptCache:
    if isinstance(outcome, ClaudePromptCache):
        return outcome
    raise_unsupported_prompt_cache_param(outcome, model=model, llm_provider=llm_provider)


def raise_unsupported_prompt_cache_param(
    problem: UnsupportedPromptCacheParam, model: str, llm_provider: str
) -> NoReturn:
    raise UnsupportedParamsError(
        message=(
            f"{model} does not support {problem.param}={problem.value!r}: {problem.reason}. "
            "To drop unsupported params, set `litellm.drop_params = True`."
        ),
        model=model,
        llm_provider=llm_provider,
    )
