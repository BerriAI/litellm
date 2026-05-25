"""Dispatch ``context_management`` edits to registered polyfill editors."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_logger
from litellm.types.llms.anthropic import AppliedEdit

from .constants import CLEAR_TOOL_USES_EDIT_TYPE
from .editors import apply_clear_tool_uses_20250919

EditorFn = Callable[..., Tuple[List[Dict[str, Any]], Optional[AppliedEdit]]]

_EDITOR_REGISTRY: Dict[str, EditorFn] = {
    CLEAR_TOOL_USES_EDIT_TYPE: apply_clear_tool_uses_20250919,
}


def apply_context_management(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    system: Any,
    context_management_spec: Union[Dict[str, Any], List[Dict[str, Any]], None],
) -> Tuple[List[Dict[str, Any]], List[AppliedEdit]]:
    """Run edits in order; return (messages, applied_edits that fired)."""
    # Accept both Anthropic-native dict form and OpenAI list form. The other
    # provider paths normalize via ``map_openai_context_management_to_anthropic``
    # before dispatching; do the same here so the polyfill path doesn't silently
    # no-op on list input.
    if isinstance(context_management_spec, list):
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        context_management_spec = (
            AnthropicConfig.map_openai_context_management_to_anthropic(
                context_management_spec
            )
        )

    edits = (
        context_management_spec.get("edits")
        if isinstance(context_management_spec, dict)
        else None
    )
    if not edits or not isinstance(edits, list):
        return messages, []

    applied_edits: List[AppliedEdit] = []
    current_messages = messages

    for edit_spec in edits:
        if not isinstance(edit_spec, dict):
            continue
        edit_type = edit_spec.get("type")
        editor = _EDITOR_REGISTRY.get(edit_type) if isinstance(edit_type, str) else None
        if editor is None:
            verbose_logger.debug(
                "context_management polyfill: unknown edit type '%s' — skipping",
                edit_type,
            )
            continue

        current_messages, applied = editor(
            model=model,
            messages=current_messages,
            tools=tools,
            system=system,
            edit_spec=edit_spec,
        )
        if applied is not None:
            applied_edits.append(applied)

    return current_messages, applied_edits
