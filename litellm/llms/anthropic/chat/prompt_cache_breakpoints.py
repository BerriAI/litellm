"""OpenAI's ``prompt_cache_breakpoint`` on a chat request bound for Claude, rewritten as Claude's ``cache_control``."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, NoReturn, TypeAlias, cast  # noqa: TID251  # isinstance-checked hot-path JSON

from litellm._logging import verbose_logger
from litellm.constants import MAX_CACHE_CONTROL_BLOCKS
from litellm.exceptions import UnsupportedParamsError
from litellm.responses.litellm_completion_transformation.tool_search import unanswered_tool_search_call_ids
from litellm.types.llms.openai import ChatCompletionCachedContent

_Path: TypeAlias = tuple[str | int, ...]
_Fields: TypeAlias = Mapping[str, object]
_Mark: TypeAlias = Callable[[_Path, _Fields], _Fields]
_Kind: TypeAlias = Literal["written", "explicit", "free"]
_MESSAGE_BLOCK_KEYS: Final = ("content", "tool_calls")
_REQUEST: Final[_Path] = ("cache_control",)
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


def _with_one_tool_breakpoint(tool: _Fields) -> _Fields:
    """Claude sends one mark per tool and Converse reads only the top-level one, so a breakpoint written under
    ``function`` moves up, and a ``cache_control`` the client wrote at either level wins over every breakpoint."""
    function: Final = _fields(tool.get("function"))
    if function is None:
        return tool
    function_marker: Final = function.get("prompt_cache_breakpoint")
    tool_marker: Final = tool.get("prompt_cache_breakpoint")
    if function_marker is None and (tool_marker is None or function.get("cache_control") is None):
        return tool
    client_marked: Final = function.get("cache_control") is not None or tool.get("cache_control") is not None
    return {
        **{key: value for key, value in tool.items() if key != "prompt_cache_breakpoint"},
        "function": {key: value for key, value in function.items() if key != "prompt_cache_breakpoint"},
        **(
            {}
            if client_marked
            else {"prompt_cache_breakpoint": function_marker if tool_marker is None else tool_marker}
        ),
    }


def _marked_tool(index: int, tool: object, mark: _Mark) -> object:
    fields: Final = _fields(tool)
    if fields is None:
        return tool
    marked: Final = mark(("tools", index), _with_one_tool_breakpoint(fields))
    function: Final = _fields(marked.get("function"))
    return marked if function is None else {**marked, "function": mark(("tools", index, "function"), function)}


def _breakpoint_problem(block: _Fields) -> UnsupportedPromptCacheParam | None:
    marker: Final = block.get("prompt_cache_breakpoint")
    if marker is None or block.get("cache_control") is not None:
        return None
    mark: Final = _breakpoint_mark(marker)
    return mark if isinstance(mark, UnsupportedPromptCacheParam) else None


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


def _leading_system_count(messages: Sequence[object]) -> int:
    return next(
        (
            index
            for index, message in enumerate(messages)
            if (_fields(message) or {}).get("role") not in ("system", "developer")
        ),
        len(messages),
    )


def _wire_position(path: _Path, leading_system_count: int) -> tuple[int, ...]:
    """Where Claude reads a mark: tool definitions, then the leading system messages it hoists, then every other
    message in place, with a message-level mark on the message's last block."""
    if path[0] == "tools":
        return (0, *(index for index in path if isinstance(index, int)))
    index: Final = int(path[1])
    group: Final = 1 if index < leading_system_count else 2
    block: Final = (
        (_MESSAGE_BLOCK_KEYS.index(str(path[2])), int(path[3])) if len(path) == 4 else (len(_MESSAGE_BLOCK_KEYS),)
    )
    return (group, index, *block)


def _tool_call_id(message: _Fields, index: int) -> object:
    return (_fields((_blocks(message.get("tool_calls")) or ())[index]) or {}).get("id")


def _on_wire(path: _Path, messages: Sequence[object], tools: Sequence[object], unsent_call_ids: frozenset[str]) -> bool:
    """Claude sends a message-level mark on string content, or on list content only for a tool result, a mark
    under ``function`` only when the tool carries none of its own, and no tool_search call nobody answered."""
    if path[0] == "tools":
        return len(path) == 2 or (_fields(tools[int(path[1])]) or {}).get("cache_control") is None
    message: Final = _fields(messages[int(path[1])]) or {}
    if len(path) == 4:
        return path[2] != "tool_calls" or _tool_call_id(message, int(path[3])) not in unsent_call_ids
    return isinstance(message.get("content"), str) or message.get("role") == "tool"


def _ttl(cache_control: object) -> Literal["5m", "1h"]:
    return "1h" if (_fields(cache_control) or {}).get("ttl") == "1h" else "5m"


def _kind(path: _Path, cache_control: object, translated: frozenset[_Path]) -> _Kind:
    if path not in translated:
        return "written"
    return "free" if (_fields(cache_control) or {}).get("ttl") is None else "explicit"


def _chosen(mark: _Placed, ttl: Literal["5m", "1h"]) -> bool:
    return mark.kind != "free" and mark.ttl == ttl


def _last_chosen_hour(marks: Sequence[_Placed]) -> int:
    return max((index for index, mark in enumerate(marks) if _chosen(mark, "1h")), default=-1)


def _loosened(marks: Sequence[_Placed], drop_params: bool) -> tuple[_Placed, ...]:
    """Claude rejects a 1h mark read after a 5m one."""
    first_five: Final = next((index for index, mark in enumerate(marks) if _chosen(mark, "5m")), len(marks))
    last_hour: Final = _last_chosen_hour(marks)
    return tuple(
        _Placed(mark.path, "5m", "free")
        if drop_params and mark.kind == "explicit" and (index > first_five if mark.ttl == "1h" else index < last_hour)
        else mark
        for index, mark in enumerate(marks)
    )


def _settled(marks: Sequence[_Placed], drop_params: bool) -> tuple[_Placed, ...]:
    """A breakpoint without a ttl takes 1h when a 1h mark is read after it, so the request stays valid for Claude;
    under ``drop_params`` a ttl the client set on a breakpoint that breaks that order is dropped first."""
    loosened: Final = _loosened(marks, drop_params)
    last_hour: Final = _last_chosen_hour(loosened)
    return tuple(
        _Placed(mark.path, "1h", "free") if mark.kind == "free" and index < last_hour else mark
        for index, mark in enumerate(loosened)
    )


def claude_prompt_cache(
    messages: Sequence[object],
    tools: Sequence[object] | None,
    params: _Fields,
    *,
    drop_params: bool,
) -> ClaudePromptCache | UnsupportedPromptCacheParam:
    """``prompt_cache_breakpoint`` marks renamed to ``cache_control``. A mark Claude cannot honor is reported unless
    ``drop_params``, which keeps what Claude can use and, when the request ends up with more than Claude's four
    breakpoints, drops the earliest translated ones. A request-level ``cache_control`` in ``params`` takes one of the
    four. A request carrying no breakpoint is returned as written."""
    listed_tools: Final = tools or ()
    if not any(map(_message_carries_breakpoint, messages)) and not any(map(_tool_carries_breakpoint, listed_tools)):
        return ClaudePromptCache(tuple(messages), None if tools is None else tuple(tools))
    blocks: Final = tuple(block for _, block in _markable(*_marked_request(messages, listed_tools, _unchanged)))
    problem: Final = next(filter(None, map(_breakpoint_problem, blocks)), None)
    if problem is not None and not drop_params:
        return problem
    if problem is not None:
        verbose_logger.debug("Claude: adjusting %s=%r because %s", problem.param, problem.value, problem.reason)
    renamed_messages, renamed_tools = _marked_request(messages, listed_tools, _renamed)
    translated: Final = _translated(messages, listed_tools)
    leading_system_count: Final = _leading_system_count(messages)
    unsent_call_ids: Final = unanswered_tool_search_call_ids(messages)
    block_marks: Final = sorted(
        (
            _Placed(path, _ttl(block.get("cache_control")), _kind(path, block.get("cache_control"), translated))
            for path, block in _markable(renamed_messages, renamed_tools)
            if block.get("cache_control") is not None
            and _on_wire(path, renamed_messages, renamed_tools, unsent_call_ids)
        ),
        key=lambda mark: _wire_position(mark.path, leading_system_count),
    )
    request_mark: Final = params.get("cache_control")
    marks: Final = (
        *block_marks,
        *(() if request_mark is None else (_Placed(_REQUEST, _ttl(request_mark), "written"),)),
    )
    excess: Final = len(marks) - MAX_CACHE_CONTROL_BLOCKS
    dropped: Final[frozenset[_Path]] = (
        frozenset(tuple(mark.path for mark in marks if mark.kind != "written")[:excess])
        if excess > 0 and drop_params
        else frozenset()
    )
    if dropped:
        verbose_logger.debug("Claude: dropping the earliest %d prompt_cache_breakpoint marks", len(dropped))
    kept: Final = tuple(mark for mark in marks if mark.path not in dropped)
    replacements: Final = MappingProxyType(
        {
            settled.path: (
                ChatCompletionCachedContent(type="ephemeral", ttl="1h")
                if settled.ttl == "1h"
                else ChatCompletionCachedContent(type="ephemeral")
            )
            for mark, settled in zip(kept, _settled(kept, drop_params))
            if settled.ttl != mark.ttl
        }
    )
    final_messages, final_tools = _marked_request(
        renamed_messages, renamed_tools, _settled_marks(dropped, replacements)
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
