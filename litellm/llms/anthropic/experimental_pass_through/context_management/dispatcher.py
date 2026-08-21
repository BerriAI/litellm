"""Dispatch ``context_management`` edits to registered polyfill editors."""

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Final, TypeAlias, TypedDict, cast

from typing_extensions import ReadOnly

from litellm._logging import verbose_logger
from litellm.types.llms.anthropic import AppliedEdit

from .constants import CLEAR_TOOL_USES_EDIT_TYPE, COMPACT_EDIT_TYPE
from .editors import apply_clear_tool_uses_20250919, apply_compact_20260112
from .result import PolyfillResult

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.router import Router

AnthropicMessages: TypeAlias = list[dict[str, object]]
AnthropicSystem: TypeAlias = str | list[dict[str, object]] | None
AnthropicTools: TypeAlias = list[dict[str, object]] | None
EditSpec: TypeAlias = dict[str, object]
ContextManagementSpec: TypeAlias = EditSpec | list[EditSpec] | None
SyncEditorReturn: TypeAlias = tuple[AnthropicMessages, AppliedEdit | None]
EditorFn: TypeAlias = Callable[..., object]


class EditorKwargs(TypedDict):
    """The keyword payload every registered editor accepts."""

    model: ReadOnly[str]
    messages: ReadOnly[AnthropicMessages]
    tools: ReadOnly[AnthropicTools]
    system: ReadOnly[AnthropicSystem]
    edit_spec: ReadOnly[EditSpec]


_EDITOR_REGISTRY: Final[Mapping[str, EditorFn]] = {
    CLEAR_TOOL_USES_EDIT_TYPE: apply_clear_tool_uses_20250919,
    COMPACT_EDIT_TYPE: apply_compact_20260112,
}


def _map_openai_spec(spec: list[EditSpec]) -> EditSpec | None:
    """Translate the OpenAI list form into the Anthropic-native dict form."""
    # Local import to avoid an import cycle at module load.
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    return AnthropicConfig.map_openai_context_management_to_anthropic(spec)


def _normalize_spec(spec: ContextManagementSpec) -> list[EditSpec] | None:
    """Accept Anthropic-native dict form or OpenAI list form; return edits list."""
    normalized: Final = _map_openai_spec(spec) if isinstance(spec, list) else spec

    edits: Final = normalized.get("edits") if isinstance(normalized, dict) else None
    if not edits or not isinstance(edits, list):
        return None
    return [edit for edit in edits if isinstance(edit, dict)]


def _wrap_editor_return(raw: object, *, fallback_system: AnthropicSystem) -> PolyfillResult:
    """Coerce an editor's native return shape into a ``PolyfillResult``.

    v0 sync editors (e.g. ``clear_tool_uses_20250919``) return a 2-tuple
    ``(messages, Optional[AppliedEdit])``. The new async ``compact_20260112``
    editor returns a ``PolyfillResult`` directly.
    """
    if isinstance(raw, PolyfillResult):
        return raw
    # Legacy 2-tuple return — sync editors don't mutate ``system``, so
    # carry the caller's value forward.
    messages, applied = cast(SyncEditorReturn, raw)
    return PolyfillResult(
        messages=messages,
        system=fallback_system,
        applied_edits=[applied] if applied is not None else [],
    )


async def apply_context_management(
    *,
    model: str,
    messages: AnthropicMessages,
    tools: AnthropicTools,
    system: AnthropicSystem,
    context_management_spec: ContextManagementSpec,
    litellm_metadata: Mapping[str, object] | None = None,
    llm_router: "Router | None" = None,
    user_api_key_auth: "UserAPIKeyAuth | None" = None,
) -> PolyfillResult:
    """Run edits in order; return a single ``PolyfillResult``.

    The dispatcher is async so async editors (``compact_20260112``) can
    ``await`` the configured summarization model. Sync editors are called
    inline — ``inspect.iscoroutinefunction`` decides how each editor is
    invoked.
    """
    edits: Final = _normalize_spec(context_management_spec)
    if not edits:
        return PolyfillResult(messages=messages, system=system, applied_edits=[])

    current_messages = messages
    current_system = system
    aggregated_applied: Final[list[AppliedEdit]] = []
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

        kwargs: EditorKwargs = {
            "model": model,
            "messages": current_messages,
            "tools": tools,
            "system": current_system,
            "edit_spec": edit_spec,
        }
        # Only async editors accept these — passing them to sync v0 editors
        # would break their signature.
        if inspect.iscoroutinefunction(editor):
            raw_result = await cast(Callable[..., Awaitable[PolyfillResult]], editor)(
                **kwargs,
                litellm_metadata=litellm_metadata,
                llm_router=llm_router,
                user_api_key_auth=user_api_key_auth,
            )
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
