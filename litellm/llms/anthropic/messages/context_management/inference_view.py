from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.types.llms.anthropic import AnthropicMessagesTextParam
from litellm.types.utils import CountContextManagement, TokenCountMessagesFormat


@dataclass(frozen=True, slots=True)
class InferenceRequestView:
    messages: Sequence[Mapping[str, object]]
    system: object | None


@dataclass(frozen=True, slots=True)
class InferenceRequestViewError:
    error_message: str
    status_code: int


@dataclass(frozen=True, slots=True)
class CountContextEdits:
    spec: JsonValue
    model: str
    tools: tuple[Mapping[str, object], ...] | None
    drop_invalid_edits: bool = False


_CONTEXT_MANAGEMENT_SPEC: Final[TypeAdapter[Mapping[str, object] | tuple[Mapping[str, object], ...]]] = TypeAdapter(
    Mapping[str, object] | tuple[Mapping[str, object], ...]
)
_SYSTEM_BLOCKS: Final = TypeAdapter(tuple[AnthropicMessagesTextParam, ...])


def count_context_edits(
    context_management: CountContextManagement | None,
    tools: Sequence[Mapping[str, object]] | None,
    litellm_params: Mapping[str, object],
    fallback_model: str,
) -> CountContextEdits:
    model: Final = litellm_params.get("model")
    return CountContextEdits(
        spec=None if context_management is None else context_management.spec,
        model=model if isinstance(model, str) else fallback_model,
        tools=None if tools is None else tuple(tools),
        drop_invalid_edits=context_management is not None and context_management.drop_invalid_edits,
    )


def _typed_spec(spec: JsonValue) -> Mapping[str, object] | tuple[Mapping[str, object], ...] | None:
    try:
        return _CONTEXT_MANAGEMENT_SPEC.validate_python(spec)
    except ValidationError:
        return None


def _typed_system(system: object | None) -> str | tuple[AnthropicMessagesTextParam, ...] | None | ValidationError:
    if system is None or isinstance(system, str):
        return system
    try:
        return _SYSTEM_BLOCKS.validate_python(system)
    except ValidationError as error:
        return error


def _view_without_edits(
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    typed_system: str | Sequence[Mapping[str, object]] | None,
) -> InferenceRequestView:
    from litellm.llms.anthropic.messages.context_management.editors.compact import (
        apply_client_compaction_block_history,
    )

    history: Final = apply_client_compaction_block_history(
        messages=[dict(message) for message in messages],
        system=typed_system
        if typed_system is None or isinstance(typed_system, str)
        else [dict(block) for block in typed_system],
    )
    if history is None:
        return InferenceRequestView(messages=messages, system=system)
    return InferenceRequestView(messages=history.messages, system=history.system)


async def _anthropic_inference_view(
    messages: Sequence[Mapping[str, object]], system: object | None, edits: CountContextEdits
) -> InferenceRequestView | InferenceRequestViewError:
    from litellm.llms.anthropic.messages.adapters.handler import prepare_context_managed_request
    from litellm.llms.anthropic.messages.context_management import AnthropicContextManagementError
    from litellm.llms.anthropic.messages.native.utils import normalize_native_messages

    typed_system: Final = _typed_system(system)
    if isinstance(typed_system, ValidationError):
        return InferenceRequestView(messages=messages, system=system)
    normalized: Final = normalize_native_messages(messages)
    spec: Final = _typed_spec(edits.spec)
    try:
        prepared: Final = await prepare_context_managed_request(
            model=edits.model,
            messages=normalized,
            tools=None if edits.tools is None else [dict(tool) for tool in edits.tools],
            system=[dict(block) for block in typed_system] if isinstance(typed_system, tuple) else typed_system,
            context_management_spec=(
                None if spec is None else dict(spec) if isinstance(spec, Mapping) else [dict(edit) for edit in spec]
            ),
            litellm_metadata=None,
            additional_drop_params=None,
            llm_router=None,
            count_only=True,
            drop_invalid_edits=edits.drop_invalid_edits,
        )
    except AnthropicContextManagementError as error:
        if error.status_code >= 500:
            return _view_without_edits(normalized, system, typed_system)
        return InferenceRequestViewError(error_message=error.message, status_code=error.status_code)
    if prepared is None:
        return InferenceRequestView(messages=normalized, system=system)
    return InferenceRequestView(messages=prepared.messages, system=prepared.system)


async def inference_request_view(
    messages: Sequence[Mapping[str, object]],
    system: object | None,
    messages_format: TokenCountMessagesFormat,
    edits: CountContextEdits,
) -> InferenceRequestView | InferenceRequestViewError:
    """Apply the history rewrites and context edits /v1/messages runs before it converts a body for another provider."""
    match messages_format:
        case "openai":
            return InferenceRequestView(messages=messages, system=system)
        case "anthropic":
            return await _anthropic_inference_view(messages, system, edits)
        case _:
            assert_never(messages_format)
