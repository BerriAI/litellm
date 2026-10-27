"""Body fields that an ``anthropic-beta`` value gates.

``anthropic_beta_headers_config.json`` says which beta each provider leg
carries. When a leg drops a beta, the request fields that beta unlocks
would make the provider 400, so the same request degrades on that leg by
dropping those fields too. Every helper here is copy-on-write: the caller's
containers are returned untouched when nothing is stripped, and a shallow
copy is returned otherwise, so a fallback leg still sends the full request.
"""

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.types.llms.anthropic import (
    ANTHROPIC_BETA_HEADER_VALUES,
    ANTHROPIC_MID_CONVERSATION_OUTPUT_CONFIG_BETA_HEADER,
    ANTHROPIC_MID_CONVERSATION_TOOL_CHANGES_BETA_HEADER,
    ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER,
)

ANTHROPIC_THINKING_BINDING_CONTROLS_BETA_HEADER: Final = "thinking-binding-controls-2026-08-01"
ANTHROPIC_MID_CONVERSATION_SYSTEM_CLEAR_AT_BETA_HEADER: Final = "mid-conversation-system-clear-at-2026-08-21"

THINKING_BLOCK_BINDING: Final = "thinking.block_binding"
THINKING_DISPLAY_UPDATES: Final = "thinking.display=updates"
SYSTEM_CLEAR_AT: Final = "messages[].system.clear_at"
SYSTEM_OUTPUT_CONFIG: Final = "messages[].system.output_config"
SYSTEM_TOOL_CHANGES: Final = "messages[].system.content[].tool_addition|tool_removal"
SYSTEM_MESSAGE_EMPTIED: Final = "messages[].system (emptied by the strip)"

_TOOL_CHANGE_BLOCK_TYPES: Final = frozenset({"tool_addition", "tool_removal"})

BETA_GATED_FIELDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        ANTHROPIC_THINKING_BINDING_CONTROLS_BETA_HEADER: (THINKING_BLOCK_BINDING,),
        ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER: (THINKING_DISPLAY_UPDATES,),
        ANTHROPIC_MID_CONVERSATION_SYSTEM_CLEAR_AT_BETA_HEADER: (SYSTEM_CLEAR_AT,),
        ANTHROPIC_MID_CONVERSATION_OUTPUT_CONFIG_BETA_HEADER: (SYSTEM_OUTPUT_CONFIG,),
        ANTHROPIC_BETA_HEADER_VALUES.PER_TURN_CONTROL_2026_07_01.value: (SYSTEM_OUTPUT_CONFIG,),
        ANTHROPIC_MID_CONVERSATION_TOOL_CHANGES_BETA_HEADER: (SYSTEM_TOOL_CHANGES,),
    }
)


def _betas_unlocking(field: str) -> frozenset[str]:
    return frozenset(beta for beta, fields in BETA_GATED_FIELDS.items() if field in fields)


_FIELD_BETAS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        field: _betas_unlocking(field) for fields in BETA_GATED_FIELDS.values() for field in fields
    }  # comprehension-ok: flattening the beta-to-fields map into a field-to-betas index
)


def field_is_gated(field: str, active_betas: Set[str]) -> bool:
    """True when no active beta unlocks ``field``, so the field must be stripped."""
    return _FIELD_BETAS[field].isdisjoint(active_betas)


_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_SEQUENCE_ADAPTER: Final = TypeAdapter(tuple[object, ...])
_HEADERS_ADAPTER: Final = TypeAdapter(Mapping[str, str])


def _as_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return None


def _as_sequence(value: object) -> tuple[object, ...] | None:
    try:
        return _SEQUENCE_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _is_tool_change_block(block: object) -> bool:
    mapping: Final = _as_mapping(block)
    return mapping is not None and mapping.get("type") in _TOOL_CHANGE_BLOCK_TYPES


def _header_beta_values(headers: Mapping[str, str]) -> frozenset[str]:
    joined: Final = ",".join(value for key, value in headers.items() if key.lower() == "anthropic-beta")
    return frozenset(piece.strip() for piece in joined.split(",") if piece.strip())


def _validated_headers(headers: object) -> Mapping[str, str]:
    try:
        return _HEADERS_ADAPTER.validate_python(headers)
    except ValidationError:
        return MappingProxyType({})


def active_beta_values(headers: object, body_betas: object = ()) -> frozenset[str]:
    """The ``anthropic-beta`` values a request will send: the comma-joined header (key matched
    case-insensitively) plus the body's ``anthropic_beta`` list on legs that carry betas in the body."""
    body_values: Final = _as_sequence(body_betas) or ()
    body_betas_as_strings: Final = frozenset(beta for beta in body_values if isinstance(beta, str))
    return _header_beta_values(_validated_headers(headers)) | body_betas_as_strings


@dataclass(frozen=True, slots=True)
class StrippedThinking:
    thinking: object
    stripped: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StrippedMessages:
    messages: Sequence[object]
    stripped: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StrippedRequest:
    request: Mapping[str, object]
    stripped: tuple[str, ...]


def strip_beta_gated_thinking(thinking: object, active_betas: Set[str]) -> StrippedThinking:
    """Drop ``thinking`` sub-fields whose beta is absent. Returns the same object when nothing is gated."""
    mapping: Final = _as_mapping(thinking)
    if mapping is None:
        return StrippedThinking(thinking=thinking, stripped=())
    binding_gated: Final = "block_binding" in mapping and field_is_gated(THINKING_BLOCK_BINDING, active_betas)
    display_gated: Final = mapping.get("display") == "updates" and field_is_gated(
        THINKING_DISPLAY_UPDATES, active_betas
    )
    if not binding_gated and not display_gated:
        return StrippedThinking(thinking=thinking, stripped=())
    gated_keys: Final = frozenset(
        {*(("block_binding",) if binding_gated else ()), *(("display",) if display_gated else ())}
    )
    stripped: Final = (
        *((THINKING_BLOCK_BINDING,) if binding_gated else ()),
        *((THINKING_DISPLAY_UPDATES,) if display_gated else ()),
    )
    return StrippedThinking(
        thinking={key: value for key, value in mapping.items() if key not in gated_keys},
        stripped=stripped,
    )


def _strip_beta_gated_system_message(message: Mapping[str, object], active_betas: Set[str]) -> StrippedMessages:
    """One mid-conversation ``role: system`` message. ``messages`` holds the rewritten message, or nothing
    when the strip left it with no content, since an empty system turn is itself a 400."""
    clear_at_gated: Final = "clear_at" in message and field_is_gated(SYSTEM_CLEAR_AT, active_betas)
    output_config_gated: Final = "output_config" in message and field_is_gated(SYSTEM_OUTPUT_CONFIG, active_betas)
    content: Final = message.get("content")
    blocks: Final = _as_sequence(content) if not isinstance(content, str) else None
    tool_changes_gated: Final = (
        blocks is not None
        and any(_is_tool_change_block(block) for block in blocks)
        and field_is_gated(SYSTEM_TOOL_CHANGES, active_betas)
    )
    if not clear_at_gated and not output_config_gated and not tool_changes_gated:
        return StrippedMessages(messages=(message,), stripped=())
    kept_content: Final[object] = (
        [block for block in blocks if not _is_tool_change_block(block)]
        if tool_changes_gated and blocks is not None
        else content
    )
    gated_keys: Final = frozenset(
        {*(("clear_at",) if clear_at_gated else ()), *(("output_config",) if output_config_gated else ())}
    )
    rewritten: Final = {
        key: (kept_content if key == "content" else value) for key, value in message.items() if key not in gated_keys
    }
    stripped: Final = (
        *((SYSTEM_CLEAR_AT,) if clear_at_gated else ()),
        *((SYSTEM_OUTPUT_CONFIG,) if output_config_gated else ()),
        *((SYSTEM_TOOL_CHANGES,) if tool_changes_gated else ()),
    )
    if kept_content in (None, "", []):
        return StrippedMessages(messages=(), stripped=(*stripped, SYSTEM_MESSAGE_EMPTIED))
    return StrippedMessages(messages=(rewritten,), stripped=stripped)


def _strip_beta_gated_message(message: object, active_betas: Set[str]) -> StrippedMessages:
    mapping: Final = _as_mapping(message)
    if mapping is None or mapping.get("role") != "system":
        return StrippedMessages(messages=(message,), stripped=())
    return _strip_beta_gated_system_message(mapping, active_betas)


def strip_beta_gated_system_messages(messages: Sequence[object], active_betas: Set[str]) -> StrippedMessages:
    """Drop the gated fields of every mid-conversation ``role: system`` message. Returns the same
    sequence when nothing is gated."""
    per_message: Final = tuple(_strip_beta_gated_message(message, active_betas) for message in messages)
    stripped: Final = tuple(
        name for result in per_message for name in result.stripped
    )  # comprehension-ok: flattening per-message strip reports
    if not stripped:
        return StrippedMessages(messages=messages, stripped=())
    return StrippedMessages(
        messages=[
            message for result in per_message for message in result.messages
        ],  # comprehension-ok: flattening per-message results
        stripped=stripped,
    )


def strip_beta_gated_request_fields(request: Mapping[str, object], active_betas: Set[str]) -> StrippedRequest:
    """Apply both strips to an Anthropic ``/v1/messages`` shaped body. Returns the same mapping when
    nothing is gated, otherwise a shallow copy with ``thinking`` and ``messages`` replaced."""
    thinking: Final = strip_beta_gated_thinking(request.get("thinking"), active_betas)
    messages: Final = request.get("messages")
    message_sequence: Final = _as_sequence(messages) if not isinstance(messages, str) else None
    system_messages: Final = (
        strip_beta_gated_system_messages(message_sequence, active_betas)
        if message_sequence is not None
        else StrippedMessages(messages=(), stripped=())
    )
    stripped: Final = (*thinking.stripped, *system_messages.stripped)
    if not stripped:
        return StrippedRequest(request=request, stripped=())
    rewritten: Final = {
        **request,
        **({"thinking": thinking.thinking} if thinking.stripped else {}),
        **({"messages": system_messages.messages} if system_messages.stripped else {}),
    }
    return StrippedRequest(request=rewritten, stripped=stripped)


@dataclass(frozen=True, slots=True)
class StrippedNativeRequest:
    thinking: object
    thinking_stripped: bool
    messages: Sequence[object]
    messages_stripped: bool
    stripped: tuple[str, ...]


def strip_beta_gated_native_request(
    headers: object, optional_params: object, messages: object
) -> StrippedNativeRequest:
    """The ``/v1/messages`` seam, fed the final (already filtered) headers plus the body parts it may rewrite."""
    active_betas: Final = active_beta_values(headers)
    params: Final = _as_mapping(optional_params) or MappingProxyType({})
    thinking: Final = strip_beta_gated_thinking(params.get("thinking"), active_betas)
    message_sequence: Final = _as_sequence(messages) if not isinstance(messages, str) else None
    system_messages: Final = (
        strip_beta_gated_system_messages(message_sequence, active_betas)
        if message_sequence is not None
        else StrippedMessages(messages=(), stripped=())
    )
    return StrippedNativeRequest(
        thinking=thinking.thinking,
        thinking_stripped=bool(thinking.stripped),
        messages=system_messages.messages,
        messages_stripped=bool(system_messages.stripped),
        stripped=(*thinking.stripped, *system_messages.stripped),
    )


def log_beta_gated_strips(stripped: Sequence[str], model: str | None, provider: str) -> None:
    if not stripped:
        return
    verbose_logger.warning(
        "Stripped %s from the request for model=%s provider=%s because this leg does not carry the "
        "anthropic-beta that gates them (see anthropic_beta_headers_config.json). The request is sent "
        "without those fields instead of failing with a provider 400.",
        sorted(set(stripped)),
        model,
        provider,
    )
