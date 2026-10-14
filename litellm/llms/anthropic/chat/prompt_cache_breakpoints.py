"""OpenAI's ``prompt_cache_breakpoint`` on a request bound for Claude.

The marks become Anthropic's ``cache_control`` on the same blocks, and a request that carried at least one is held
to Claude's cap of four breakpoints, which a client writing the OpenAI dialect cannot see. ``prompt_cache_options``
has no Claude equivalent and is rejected or dropped like any other unsupported param."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final, Literal, NoReturn, TypeAlias

from typing_extensions import assert_never

import litellm
from litellm._logging import verbose_logger
from litellm.constants import MAX_CACHE_CONTROL_BLOCKS
from litellm.exceptions import BadRequestError, UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    message_fields,
    message_without_cache_control_marks,
    messages_with_prompt_cache_breakpoints_as_cache_control,
    tools_with_prompt_cache_breakpoints_as_cache_control,
)
from litellm.llms.anthropic.common_utils import tool_call_is_rebuilt_as_server_tool_use
from litellm.types.llms.openai import AllMessageValues

PROMPT_CACHE_BREAKPOINT_ON_TOOLS: Final = "_litellm_prompt_cache_breakpoint_on_tools"
_Site: TypeAlias = tuple[Literal["tool", "message", "part", "tool_call", "tool_message"], int, int]
ClaudeWire: TypeAlias = Literal["anthropic", "bedrock_converse"]


@dataclass(frozen=True, slots=True)
class ClaudeCacheMarks:
    messages: tuple[AllMessageValues, ...]
    tools: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class CacheBreakpointCapExceeded:
    found: int


@dataclass(frozen=True, slots=True)
class PromptCacheOptionsUnsupported:
    options: object


ClaudeCacheRefusal: TypeAlias = CacheBreakpointCapExceeded | PromptCacheOptionsUnsupported


def drop_params_enabled(litellm_params: Mapping[str, object]) -> bool:
    """``drop_params`` from the global setting, or from the request or deployment (both arrive in ``litellm_params``)."""
    return bool(litellm.drop_params) or normalize_drop_params(litellm_params.get("drop_params")) is True


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


def _renders_message_mark(message: Mapping[str, object], wire: ClaudeWire) -> bool:
    """Claude's message builder keeps a message-level mark on string content and on tool results only. Anthropic's
    builder skips an empty assistant string, mark included, where Converse still emits its ``cachePoint``."""
    content: Final = message.get("content")
    if isinstance(content, str):
        return bool(content) or message.get("role") != "assistant" or wire == "bedrock_converse"
    return message.get("role") in ("tool", "function")


def _renders_tool_call_mark(message: Mapping[str, object], call: object, wire: ClaudeWire) -> bool:
    """Anthropic rebuilds a tool call answered by a server tool as ``server_tool_use``, which carries no mark."""
    return wire == "bedrock_converse" or not tool_call_is_rebuilt_as_server_tool_use(
        (_as_mapping(call) or {}).get("id"), message.get("provider_specific_fields")
    )


def _message_sites(index: int, message: object, key: str, wire: ClaudeWire) -> Iterator[_Site]:
    fields: Final = _as_mapping(message)
    if fields is None:
        return
    if fields.get(key) is not None and _renders_message_mark(fields, wire):
        yield ("message", index, -1)
    for position, part in enumerate(_as_blocks(fields.get("content")) or ()):
        if _carries(part, key):
            yield ("part", index, position)
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


def _sites(messages: Sequence[object], tools: Sequence[object], key: str, wire: ClaudeWire) -> tuple[_Site, ...]:
    """Every mark of ``key`` the ``wire`` renders, in wire order: tool definitions first, then the messages as
    listed. Converse reads only a tool definition's top-level mark."""
    if wire == "bedrock_converse":
        return (
            *(("tool", index, -1) for index, tool in enumerate(tools) if _carries(tool, key)),
            *chain.from_iterable(
                _converse_message_sites(index, message, key) for index, message in enumerate(messages)
            ),
        )
    return (
        *(("tool", index, -1) for index, tool in enumerate(tools) if _tool_carries(tool, key)),
        *chain.from_iterable(_message_sites(index, message, key, wire) for index, message in enumerate(messages)),
    )


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


def _dropped_positions(dropped: frozenset[_Site], kind: str, index: int) -> frozenset[int]:
    return frozenset(
        position for site_kind, site_index, position in dropped if (site_kind, site_index) == (kind, index)
    )


def _message_without(index: int, message: AllMessageValues, dropped: frozenset[_Site]) -> AllMessageValues:
    if ("tool_message", index, -1) in dropped:
        content: Final = _as_blocks(message_fields(message).get("content")) or ()
        return message_without_cache_control_marks(
            message, message_mark=True, parts=frozenset(range(len(content))), tool_calls=frozenset()
        )
    return message_without_cache_control_marks(
        message,
        message_mark=("message", index, -1) in dropped,
        parts=_dropped_positions(dropped, "part", index),
        tool_calls=_dropped_positions(dropped, "tool_call", index),
    )


def _prompt_cache_options_refusal(options: object, drop_params: bool) -> PromptCacheOptionsUnsupported | None:
    if options is None:
        return None
    if not drop_params:
        return PromptCacheOptionsUnsupported(options)
    verbose_logger.debug("Dropping prompt_cache_options=%r: Claude caches through cache_control.", options)
    return None


def claude_cache_marks(
    messages: Sequence[AllMessageValues],
    tools: object,
    *,
    automatic_breakpoint: bool,
    drop_params: bool,
    prompt_cache_options: object = None,
    tools_carried_breakpoint: bool = False,
    wire: ClaudeWire = "anthropic",
) -> ClaudeCacheMarks | ClaudeCacheRefusal:
    """The request's ``prompt_cache_breakpoint`` marks renamed to ``cache_control`` and, when it carried any, held to
    Claude's cap: ``drop_params`` keeps the latest four in wire order, otherwise the excess is reported. A request
    that carried none is forwarded as written so the provider's own answer reaches the client.

    ``prompt_cache_options`` has no Claude equivalent: the caller has already taken it out of the request, and it is
    reported unless ``drop_params``. ``automatic_breakpoint`` is the request-level ``cache_control`` (Anthropic's
    automatic caching), which takes one slot of its own and is never dropped. ``tools_carried_breakpoint`` covers
    tools already mapped to Anthropic's shape, where the mark was renamed before this census ran. ``wire`` is the
    body the marks are rendered into, so the count matches the breakpoints the provider receives."""
    options_refusal: Final = _prompt_cache_options_refusal(prompt_cache_options, drop_params)
    if options_refusal is not None:
        return options_refusal
    listed_tools: Final = _as_blocks(tools) or ()
    if not tools_carried_breakpoint and not _sites(messages, listed_tools, "prompt_cache_breakpoint", wire):
        return ClaudeCacheMarks(tuple(messages), tuple(listed_tools))
    renamed_messages: Final = messages_with_prompt_cache_breakpoints_as_cache_control(messages)
    renamed_tools: Final = tools_with_prompt_cache_breakpoints_as_cache_control(listed_tools)
    sites: Final = _sites(renamed_messages, renamed_tools, "cache_control", wire)
    found: Final = len(sites) + int(automatic_breakpoint)
    if found <= MAX_CACHE_CONTROL_BLOCKS:
        return ClaudeCacheMarks(renamed_messages, renamed_tools)
    if not drop_params:
        return CacheBreakpointCapExceeded(found)
    dropped: Final = frozenset(sites[: found - MAX_CACHE_CONTROL_BLOCKS])
    verbose_logger.debug(
        "Dropping the earliest %d of %d cache breakpoints: Claude accepts at most %d.",
        len(dropped),
        found,
        MAX_CACHE_CONTROL_BLOCKS,
    )
    return ClaudeCacheMarks(
        messages=tuple(_message_without(index, message, dropped) for index, message in enumerate(renamed_messages)),
        tools=tuple(
            _tool_without_cache_control(tool) if ("tool", index, -1) in dropped else tool
            for index, tool in enumerate(renamed_tools)
        ),
    )


def claude_extra_body(
    extra_body: Mapping[str, object], drop_params: bool
) -> Mapping[str, object] | PromptCacheOptionsUnsupported:
    """``extra_body`` merged into a Claude body, held to the same ``prompt_cache_options`` rule as the request."""
    refusal: Final = _prompt_cache_options_refusal(extra_body.get("prompt_cache_options"), drop_params)
    if refusal is not None:
        return refusal
    if "prompt_cache_options" not in extra_body:
        return extra_body
    return {key: value for key, value in extra_body.items() if key != "prompt_cache_options"}


def raise_claude_cache_refusal(refusal: ClaudeCacheRefusal, model: str, llm_provider: str) -> NoReturn:
    match refusal:
        case CacheBreakpointCapExceeded(found=found):
            raise BadRequestError(
                message=(
                    f"Claude accepts at most {MAX_CACHE_CONTROL_BLOCKS} cache breakpoints, found {found}. "
                    "Set drop_params=True (proxy: litellm_settings: drop_params: true) to keep the latest four."
                ),
                model=model,
                llm_provider=llm_provider,
            )
        case PromptCacheOptionsUnsupported(options=options):
            raise UnsupportedParamsError(
                message=(
                    f"{model} does not support prompt_cache_options={options!r}. It caches through cache_control "
                    "marks (or prompt_cache_breakpoint on the OpenAI dialect). To drop unsupported params, set "
                    "`litellm.drop_params = True`."
                ),
                model=model,
                status_code=400,
            )
        case _:
            assert_never(refusal)


def resolved_claude_cache_marks(
    outcome: ClaudeCacheMarks | ClaudeCacheRefusal, model: str, llm_provider: str
) -> ClaudeCacheMarks:
    if isinstance(outcome, ClaudeCacheMarks):
        return outcome
    raise_claude_cache_refusal(outcome, model=model, llm_provider=llm_provider)


def resolved_claude_extra_body(
    outcome: Mapping[str, object] | PromptCacheOptionsUnsupported, model: str, llm_provider: str
) -> Mapping[str, object]:
    if isinstance(outcome, PromptCacheOptionsUnsupported):
        raise_claude_cache_refusal(outcome, model=model, llm_provider=llm_provider)
    return outcome
