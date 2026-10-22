from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Final, overload

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field, parts_of

if TYPE_CHECKING:
    from litellm.types.llms.anthropic import AnthropicMessagesRequest


@dataclass(frozen=True, slots=True)
class _ToolChange:
    name: str
    remove: bool
    definition: Mapping[str, object] | None


def _tool_change(block: object) -> _ToolChange | None:
    action: Final = message_field(block, "type")
    if action not in ("tool_addition", "tool_removal"):
        return None
    tool: Final = message_field(block, "tool")
    tool_type: Final = message_field(tool, "type")
    definition: Final = (
        as_str_mapping(message_field(tool, "definition"))
        if action == "tool_addition" and tool_type == "tool_definition"
        else None
    )
    if definition is None and tool_type != "tool_reference":
        return None
    name: Final = message_field(definition if definition is not None else tool, "name")
    return _ToolChange(name, action == "tool_removal", definition) if isinstance(name, str) and name else None


def _message_tool_changes(message: object) -> tuple[_ToolChange, ...]:
    if message_field(message, "role") != "system":
        return ()
    return tuple(
        change for block in parts_of(message_field(message, "content")) if (change := _tool_change(block)) is not None
    )


def _without_tool_changes(message: object) -> object | None:
    mapping: Final = as_str_mapping(message)
    if mapping is None or not _message_tool_changes(mapping):
        return message
    content: Final = [block for block in parts_of(mapping.get("content")) if _tool_change(block) is None]
    return {**mapping, "content": content} if content else None


def _tool_name(tool: object) -> str | None:
    definition: Final = message_field(tool, "function") if message_field(tool, "type") == "function" else tool
    name: Final = message_field(definition, "name")
    return name if isinstance(name, str) else None


@overload
def resolve_midturn_tool_changes(request: AnthropicMessagesRequest) -> AnthropicMessagesRequest: ...


@overload
def resolve_midturn_tool_changes(request: Mapping[str, object]) -> Mapping[str, object]: ...


def resolve_midturn_tool_changes(request: Mapping[str, object]) -> Mapping[str, object]:
    """Resolve tool availability once so authorization and translation agree."""
    messages: Final = parts_of(request.get("messages"))
    changes: Final = tuple(chain.from_iterable(_message_tool_changes(message) for message in messages))
    if not changes:
        return request
    tools: Final = parts_of(request.get("tools"))
    existing_names: Final = frozenset(_tool_name(tool) for tool in tools)
    last_change: Final = {change.name: change for change in changes}
    definitions: Final = {
        change.name: change.definition for change in reversed(changes) if change.definition is not None
    }
    added: Final = tuple(
        definitions[name] for name in last_change if name in definitions and name not in existing_names
    )
    removed: Final = frozenset(change.name for change in last_change.values() if change.remove)
    return {
        **request,
        "tools": [tool for tool in (*tools, *added) if _tool_name(tool) not in removed],
        "messages": [cleaned for message in messages if (cleaned := _without_tool_changes(message)) is not None],
    }
