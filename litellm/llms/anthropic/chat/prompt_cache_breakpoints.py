"""OpenAI's ``prompt_cache_breakpoint`` on a chat request bound for Claude, rewritten as Claude's ``cache_control``."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, NoReturn, TypeAlias

from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.constants import MAX_CACHE_CONTROL_BLOCKS
from litellm.exceptions import UnsupportedParamsError
from litellm.types.llms.openai import ChatCompletionCachedContent

_Path: TypeAlias = tuple[str | int, ...]
_Fields: TypeAlias = Mapping[str, object]
_Mark: TypeAlias = Callable[[_Path, _Fields], _Fields]
_MESSAGE_BLOCK_KEYS: Final = ("content", "tool_calls")
_FIELDS: Final = TypeAdapter(Mapping[str, object])
_BLOCKS: Final = TypeAdapter(Sequence[object])
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


def _fields(value: object) -> _Fields | None:
    return _FIELDS.validate_python(value) if isinstance(value, Mapping) else None


def _blocks(value: object) -> Sequence[object] | None:
    return _BLOCKS.validate_python(value) if isinstance(value, list) else None


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


def _with_function_breakpoint_hoisted(tool: _Fields) -> _Fields:
    """Converse reads only a tool's top-level mark, so a breakpoint written under ``function`` moves up."""
    function: Final = _fields(tool.get("function"))
    marker: Final = None if function is None else function.get("prompt_cache_breakpoint")
    if function is None or marker is None:
        return tool
    keeps_client_mark: Final = function.get("cache_control") is not None or "prompt_cache_breakpoint" in tool
    return {
        **tool,
        "function": {key: value for key, value in function.items() if key != "prompt_cache_breakpoint"},
        **({} if keeps_client_mark else {"prompt_cache_breakpoint": marker}),
    }


def _marked_tool(index: int, tool: object, mark: _Mark) -> object:
    fields: Final = _fields(tool)
    if fields is None:
        return tool
    marked: Final = mark(("tools", index), _with_function_breakpoint_hoisted(fields))
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


def _without_cache_control(dropped: frozenset[_Path]) -> _Mark:
    def mark(path: _Path, block: _Fields) -> _Fields:
        return {key: value for key, value in block.items() if key != "cache_control"} if path in dropped else block

    return mark


def claude_prompt_cache(
    messages: Sequence[object],
    tools: Sequence[object] | None,
    params: _Fields,
    *,
    drop_params: bool,
) -> ClaudePromptCache | UnsupportedPromptCacheParam:
    """``prompt_cache_breakpoint`` marks renamed to ``cache_control``. A mark Claude cannot honor is reported unless
    ``drop_params``, which keeps what Claude can use and, when the request ends up with more than Claude's four
    breakpoints, the latest four. A request-level ``cache_control`` in ``params`` takes one of the four. A request
    carrying no breakpoint is returned as written."""
    listed_tools: Final = tools or ()
    blocks: Final = tuple(block for _, block in _markable(messages, listed_tools))
    if all(block.get("prompt_cache_breakpoint") is None for block in blocks):
        return ClaudePromptCache(tuple(messages), None if tools is None else tuple(tools))
    problem: Final = next(filter(None, map(_breakpoint_problem, blocks)), None)
    if problem is not None and not drop_params:
        return problem
    if problem is not None:
        verbose_logger.debug("Claude: adjusting %s=%r because %s", problem.param, problem.value, problem.reason)
    renamed_messages, renamed_tools = _marked_request(messages, listed_tools, _renamed)
    sites: Final = tuple(
        path for path, block in _markable(renamed_messages, renamed_tools) if block.get("cache_control") is not None
    )
    excess: Final = len(sites) + int(params.get("cache_control") is not None) - MAX_CACHE_CONTROL_BLOCKS
    if excess > 0 and drop_params:
        verbose_logger.debug("Claude: dropping the earliest %d cache breakpoints", excess)
    trimmed_messages, trimmed_tools = (
        _marked_request(renamed_messages, renamed_tools, _without_cache_control(frozenset(sites[:excess])))
        if excess > 0 and drop_params
        else (renamed_messages, renamed_tools)
    )
    return ClaudePromptCache(trimmed_messages, None if tools is None else trimmed_tools)


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
