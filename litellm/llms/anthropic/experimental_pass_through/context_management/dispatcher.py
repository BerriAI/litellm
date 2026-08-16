"""Dispatch ``context_management`` edits to registered polyfill editors."""

import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union, cast

from litellm._logging import verbose_logger
from litellm.types.llms.anthropic import AppliedEdit

from .constants import CLEAR_TOOL_USES_EDIT_TYPE, COMPACT_EDIT_TYPE
from .editors import apply_clear_tool_uses_20250919, apply_compact_20260112
from .result import PolyfillResult

EditorFn = Callable[..., Any]

_EDITOR_REGISTRY: Dict[str, EditorFn] = {
    CLEAR_TOOL_USES_EDIT_TYPE: apply_clear_tool_uses_20250919,
    COMPACT_EDIT_TYPE: apply_compact_20260112,
}


def _normalize_spec(
    spec: Union[Dict[str, Any], List[Dict[str, Any]], None],
) -> Optional[List[Dict[str, Any]]]:
    """Accept Anthropic-native dict form or OpenAI list form; return edits list."""
    if isinstance(spec, list):
        # Local import to avoid an import cycle at module load.
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        spec = AnthropicConfig.map_openai_context_management_to_anthropic(spec)

    edits = spec.get("edits") if isinstance(spec, dict) else None
    if not edits or not isinstance(edits, list):
        return None
    return [edit for edit in edits if isinstance(edit, dict)]


def _wrap_editor_return(raw: Any, *, fallback_system: Any) -> PolyfillResult:
    """Coerce an editor's native return shape into a ``PolyfillResult``.

    v0 sync editors (e.g. ``clear_tool_uses_20250919``) return a 2-tuple
    ``(messages, Optional[AppliedEdit])``. The new async ``compact_20260112``
    editor returns a ``PolyfillResult`` directly.
    """
    if isinstance(raw, PolyfillResult):
        return raw
    # Legacy 2-tuple return — sync editors don't mutate ``system``, so
    # carry the caller's value forward.
    messages, applied = cast(Tuple[List[Dict[str, Any]], Any], raw)
    return PolyfillResult(
        messages=messages,
        system=fallback_system,
        applied_edits=[applied] if applied is not None else [],
    )


async def apply_context_management(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    system: Any,
    context_management_spec: Union[Dict[str, Any], List[Dict[str, Any]], None],
    litellm_metadata: Optional[Dict[str, Any]] = None,
    llm_router: Any = None,
    user_api_key_auth: Any = None,
) -> PolyfillResult:
    """Run edits in order; return a single ``PolyfillResult``.

    The dispatcher is async so async editors (``compact_20260112``) can
    ``await`` the configured summarization model. Sync editors are called
    inline — ``inspect.iscoroutinefunction`` decides how each editor is
    invoked.
    """
    edits = _normalize_spec(context_management_spec)
    if not edits:
        return PolyfillResult(messages=messages, system=system, applied_edits=[])

    current_messages = messages
    current_system = system
    aggregated_applied: List[AppliedEdit] = []
    aggregated_compaction_block = None
    aggregated_iterations_usage = None

    for edit_spec in edits:
        edit_type = edit_spec.get("type")
        editor = _EDITOR_REGISTRY.get(edit_type) if isinstance(edit_type, str) else None
        if editor is None:
            verbose_logger.debug(
                "context_management polyfill: unknown edit type '%s' — skipping",
                edit_type,
            )
            continue

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": current_messages,
            "tools": tools,
            "system": current_system,
            "edit_spec": edit_spec,
        }
        # Only async editors accept these — passing them to sync v0 editors
        # would break their signature.
        if inspect.iscoroutinefunction(editor):
            kwargs["litellm_metadata"] = litellm_metadata
            kwargs["llm_router"] = llm_router
            kwargs["user_api_key_auth"] = user_api_key_auth
            raw_result = await cast(Callable[..., Awaitable[Any]], editor)(**kwargs)
        else:
            raw_result = editor(**kwargs)

        result = _wrap_editor_return(raw_result, fallback_system=current_system)

        current_messages = result.messages
        current_system = result.system
        aggregated_applied.extend(result.applied_edits)
        if result.compaction_block is not None:
            aggregated_compaction_block = result.compaction_block
        if result.iterations_usage is not None:
            aggregated_iterations_usage = result.iterations_usage

    return PolyfillResult(
        messages=current_messages,
        system=current_system,
        applied_edits=aggregated_applied,
        compaction_block=aggregated_compaction_block,
        iterations_usage=aggregated_iterations_usage,
    )
