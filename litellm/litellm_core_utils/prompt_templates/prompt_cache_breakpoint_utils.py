from collections.abc import Mapping
from typing import Final, TypeVar, cast  # noqa: TID251  # Existing casts retained unchanged in this pure move

from pydantic import TypeAdapter, ValidationError

from litellm import verbose_logger
from litellm.exceptions import UnsupportedParamsError
from litellm.types.llms.openai import (
    ChatCompletionCachedContent,
    ChatCompletionPromptCacheBreakpoint,
    PromptCacheBreakpoint,
)

_MarkedT: Final = TypeVar("_MarkedT", bound=Mapping[str, object])


PROMPT_CACHE_BREAKPOINT_DROP_PARAMS_HINT: Final = (
    "set `drop_params: true` on the request or in litellm_settings (`litellm.drop_params = True` in the SDK)"
)


def with_prompt_cache_breakpoint(target: _MarkedT, marker: object) -> _MarkedT:
    """A breakpoint the client already wrote on the block wins, TTL included."""
    if marker is None or target.get("prompt_cache_breakpoint") is not None:
        return target
    marked: Final = {**target, "prompt_cache_breakpoint": marker}
    return cast(_MarkedT, marked)  # cast-ok: same block shape as the input plus the marker key


_PROMPT_CACHE_BREAKPOINT: Final = TypeAdapter(PromptCacheBreakpoint)


_CHAT_COMPLETION_PROMPT_CACHE_BREAKPOINT: Final = TypeAdapter(ChatCompletionPromptCacheBreakpoint)


_AcceptedBreakpointT: Final = TypeVar("_AcceptedBreakpointT", bound=Mapping[str, object])


def _accepted_prompt_cache_breakpoint(
    accepted: TypeAdapter[_AcceptedBreakpointT], marker: object
) -> _AcceptedBreakpointT | None:
    try:
        return accepted.validate_python(marker)
    except ValidationError:
        return None


def prompt_cache_breakpoint_within(
    accepted: TypeAdapter[_AcceptedBreakpointT], marker: object
) -> _AcceptedBreakpointT | None:
    """The client's marker cut down to the ``accepted`` shape: an unsupported ttl is dropped and the mode kept,
    and a marker whose mode the target cannot read is dropped whole."""
    whole: Final = _accepted_prompt_cache_breakpoint(accepted, marker)
    if whole is not None:
        return whole
    fields: Final = _breakpoint_fields_or_none(marker)
    without_ttl: Final = (
        None
        if fields is None
        else _accepted_prompt_cache_breakpoint(accepted, {key: item for key, item in fields.items() if key != "ttl"})
    )
    verbose_logger.debug(
        "Dropping malformed prompt_cache_breakpoint %r under drop_params"
        if without_ttl is None
        else "Dropping the unsupported ttl of prompt_cache_breakpoint %r under drop_params",
        marker,
    )
    return without_ttl


def prompt_cache_breakpoint_for_wire(marker: object, drop_params: bool) -> object:
    if marker is None or not drop_params:
        return marker
    return prompt_cache_breakpoint_within(_PROMPT_CACHE_BREAKPOINT, marker)


def prompt_cache_breakpoint_with_readable_mode(marker: object, drop_params: bool) -> object:
    """The client's marker as written, or None under drop_params when no target could read its mode, so the
    target's own config decides which ttl and keys it keeps."""
    return None if prompt_cache_breakpoint_for_wire(marker, drop_params) is None else marker


def whole_prompt_cache_breakpoint_or_none(marker: object) -> PromptCacheBreakpoint | None:
    return _accepted_prompt_cache_breakpoint(_PROMPT_CACHE_BREAKPOINT, marker)


def chat_completion_prompt_cache_breakpoint_for_wire(marker: object, drop_params: bool) -> object:
    if marker is None or not drop_params:
        return marker
    return prompt_cache_breakpoint_within(_CHAT_COMPLETION_PROMPT_CACHE_BREAKPOINT, marker)


def cache_control_as_prompt_cache_breakpoint(block: _MarkedT) -> _MarkedT:
    """Anthropic's ``cache_control`` mark rewritten as OpenAI's ``prompt_cache_breakpoint`` on the same block."""
    if block.get("cache_control") is None:
        return block
    unmarked: Final = {key: value for key, value in block.items() if key != "cache_control"}
    return cast(  # cast-ok: same block shape as the input with the marker key renamed
        _MarkedT, with_prompt_cache_breakpoint(unmarked, PromptCacheBreakpoint(mode="explicit"))
    )


def prompt_cache_breakpoint_as_cache_control(block: _MarkedT, drop_params: bool) -> _MarkedT:
    """OpenAI's ``prompt_cache_breakpoint`` mark rewritten as Anthropic's ``cache_control`` on the same block.
    A ``cache_control`` the client already wrote on the block wins, TTL included."""
    marker: Final = block.get("prompt_cache_breakpoint")
    if marker is None:
        return block
    unmarked: Final = {key: value for key, value in block.items() if key != "prompt_cache_breakpoint"}
    control: Final = (
        None
        if block.get("cache_control") is not None
        else _cache_control_for_prompt_cache_breakpoint(marker, drop_params)
    )
    marked: Final = unmarked if control is None else {**unmarked, "cache_control": control}
    return cast(_MarkedT, marked)  # cast-ok: same block shape as the input with the marker key renamed


def _cache_control_for_prompt_cache_breakpoint(marker: object, drop_params: bool) -> ChatCompletionCachedContent | None:
    try:
        breakpoint_mark: Final = _PROMPT_CACHE_BREAKPOINT.validate_python(marker)
    except ValidationError:
        if not drop_params:
            raise UnsupportedParamsError(
                message=(
                    f"prompt_cache_breakpoint must be {{'mode': 'explicit'}} with an optional ttl of '30m' or '12h', got "
                    f"{marker!r}. Fix the marker, or "
                    f"{PROMPT_CACHE_BREAKPOINT_DROP_PARAMS_HINT} to drop it."
                ),
                status_code=400,
            )
        verbose_logger.debug("Dropping malformed prompt_cache_breakpoint %r under drop_params", marker)
        return None
    ttl: Final = breakpoint_mark.get("ttl")
    if ttl is not None and not drop_params:
        raise UnsupportedParamsError(
            message=(
                f"This model's prompt cache lasts 5m, or 1h where it offers one, so prompt_cache_breakpoint "
                f"ttl={ttl!r} has no equivalent. Remove the ttl, {PROMPT_CACHE_BREAKPOINT_DROP_PARAMS_HINT} to send "
                "a 5m breakpoint instead, or send cache_control {'type': 'ephemeral', 'ttl': '1h'} on the block "
                "where the model offers a 1h cache."
            ),
            status_code=400,
        )
    return ChatCompletionCachedContent(type="ephemeral")


_BREAKPOINT_BLOCK_LIST_KEYS: Final = frozenset({"content", "tool_calls"})


_BREAKPOINT_FIELDS: Final = TypeAdapter(Mapping[str, object])


_BREAKPOINT_BLOCKS: Final = TypeAdapter(list[object])


def _breakpoint_fields_or_none(value: object) -> Mapping[str, object] | None:
    try:
        return _BREAKPOINT_FIELDS.validate_python(value)
    except ValidationError:
        return None


def _without_prompt_cache_breakpoint(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    fields: Final = _BREAKPOINT_FIELDS.validate_python(value)
    return {key: item for key, item in fields.items() if key != "prompt_cache_breakpoint"}


def _breakpoints_stripped_from(key: str, value: object) -> object:
    if key in _BREAKPOINT_BLOCK_LIST_KEYS and isinstance(value, list):
        return [_without_prompt_cache_breakpoint(block) for block in _BREAKPOINT_BLOCKS.validate_python(value)]
    return _without_prompt_cache_breakpoint(value) if key == "function" else value


def without_prompt_cache_breakpoints(block: _MarkedT) -> _MarkedT:
    """A message or tool definition with every ``prompt_cache_breakpoint`` removed: its own, its content parts',
    its tool calls' and its ``function``'s."""
    stripped: Final = {
        key: _breakpoints_stripped_from(key, value) for key, value in block.items() if key != "prompt_cache_breakpoint"
    }
    return cast(_MarkedT, stripped)  # cast-ok: same block shape as the input minus the marker keys


def _part_with_chat_completion_prompt_cache_breakpoint(part: object) -> object:
    if not isinstance(part, Mapping):
        return part
    fields: Final = _BREAKPOINT_FIELDS.validate_python(part)
    marker: Final = fields.get("prompt_cache_breakpoint")
    if marker is None:
        return fields
    unmarked: Final = {key: item for key, item in fields.items() if key != "prompt_cache_breakpoint"}
    return with_prompt_cache_breakpoint(unmarked, chat_completion_prompt_cache_breakpoint_for_wire(marker, True))


def chat_completion_prompt_cache_breakpoints_for_wire(message: _MarkedT, drop_params: bool) -> _MarkedT:
    content: Final = message.get("content")
    if not drop_params or not isinstance(content, list):
        return message
    parts: Final = [
        _part_with_chat_completion_prompt_cache_breakpoint(part) for part in _BREAKPOINT_BLOCKS.validate_python(content)
    ]
    return cast(_MarkedT, {**message, "content": parts})  # cast-ok: same message shape with its parts' markers trimmed


LITELLM_RESPONSE_ONLY_MESSAGE_FIELDS: Final = frozenset(
    {"thinking_blocks", "reasoning_items", "provider_specific_fields"}
)


LITELLM_INTERNAL_MESSAGE_FIELDS: Final = LITELLM_RESPONSE_ONLY_MESSAGE_FIELDS | {"reasoning_content"}
