from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, cast  # noqa: TID251  # input items are forwarded as sent, as inference does

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    update_responses_input_with_model_file_ids,
    update_responses_tools_with_model_file_ids,
)
from litellm.litellm_core_utils.tool_fields import DropPolicy
from litellm.llms.base_llm.files.batch_records import responses_batch_body_to_chat_body
from litellm.responses.additional_tools import hoist_additional_tools
from litellm.responses.file_search.emulated_handler import replace_file_search_tools
from litellm.responses.litellm_completion_transformation.compaction_items import holds_compaction_summary
from litellm.responses.litellm_completion_transformation.transformation import (
    ChatCompletionSession,
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.main import previous_response_client_ids
from litellm.types.llms.openai import ResponseInputParam
from litellm.utils import ProviderConfigManager

_COUNTED_CHAT_PARAMS: Final = (
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "web_search_options",
    "reasoning_effort",
)
_ITEMS: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...])
_MAPPINGS: Final[TypeAdapter[tuple[Mapping[str, object], ...]]] = TypeAdapter(tuple[Mapping[str, object], ...])
_JSON_PARAMS: Final[TypeAdapter[Mapping[str, JsonValue]]] = TypeAdapter(Mapping[str, JsonValue])
_ENTRY: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_MARKED_ENTRY_LISTS: Final = frozenset({"content", "tool_calls"})
_FILE_ID_MAPPING: Final[TypeAdapter[Mapping[str, Mapping[str, str]] | None]] = TypeAdapter(
    Mapping[str, Mapping[str, str]] | None
)
MODEL_FILE_ID_MAPPING_FIELD: Final = "model_file_id_mapping"


@dataclass(frozen=True, slots=True)
class ChatCountRequest:
    messages: tuple[Mapping[str, object], ...]
    tools: tuple[Mapping[str, object], ...] | None
    params: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ChatCountBridgeError:
    message: str


def _plain(item: object) -> object:
    return item.model_dump(exclude_none=True) if isinstance(item, BaseModel) else item


def _mappings(value: object) -> tuple[Mapping[str, object], ...]:
    return _MAPPINGS.validate_python(tuple(_plain(item) for item in _ITEMS.validate_python(value or ())))


def _unmarked(entry: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: _unmarked_entries(value) if key in _MARKED_ENTRY_LISTS else value
        for key, value in entry.items()
        if key != "cache_control"
    }


def _unmarked_entries(value: object) -> object:
    return (
        [
            _unmarked(_ENTRY.validate_python(item)) if isinstance(item, Mapping) else item
            for item in _ITEMS.validate_python(value)
        ]
        if isinstance(value, list)
        else value
    )


def _routed_model_id(deployment: Mapping[str, object] | None) -> str | None:
    model_info: Final = as_str_mapping(None if deployment is None else deployment.get("model_info")) or {}
    model_id: Final = model_info.get("id")
    return model_id if isinstance(model_id, str) else None


def with_provider_file_ids(
    responses_request: Mapping[str, object], deployment: Mapping[str, object] | None
) -> Mapping[str, object]:
    mapping: Final = _FILE_ID_MAPPING.validate_python(responses_request.get(MODEL_FILE_ID_MAPPING_FIELD))
    model_id: Final = _routed_model_id(deployment)
    fields: Final = {name: value for name, value in responses_request.items() if name != MODEL_FILE_ID_MAPPING_FIELD}
    tools: Final = fields.get("tools")
    return {
        **fields,
        **(
            {"input": update_responses_input_with_model_file_ids(fields["input"], model_id, mapping)}
            if "input" in fields
            else {}
        ),
        **(
            {
                "tools": update_responses_tools_with_model_file_ids(
                    cast(list[dict[str, object]], tools),  # cast-ok: entries that are not tools pass through as sent
                    model_id,
                    mapping,
                )
            }
            if isinstance(tools, list)
            else {}
        ),
    }


def _with_hoisted_additional_tools(responses_request: Mapping[str, object]) -> Mapping[str, object]:
    responses_input: Final = responses_request.get("input")
    if not isinstance(responses_input, list):
        return responses_request
    input_items: Final = cast(ResponseInputParam, responses_input)  # cast-ok: forwarded as sent, as inference does
    hoisted: Final = hoist_additional_tools(input_items, None)
    if not hoisted.hoisted:
        return responses_request
    return {
        **responses_request,
        "input": hoisted.input,
        "tools": [*_ITEMS.validate_python(responses_request.get("tools") or ()), *hoisted.hoisted],
    }


def _with_emulated_file_search(responses_request: Mapping[str, object]) -> Mapping[str, object]:
    tools: Final = responses_request.get("tools")
    if tools is None:
        return responses_request
    return {**responses_request, "tools": replace_file_search_tools(_ITEMS.validate_python(tools))[0]}


def _with_compaction_as_summary_text(responses_request: Mapping[str, object]) -> Mapping[str, object]:
    return {name: value for name, value in responses_request.items() if name != "context_management"}


async def _with_previous_turns(
    chat_body: Mapping[str, object],
    model: str,
    custom_llm_provider: str,
    responses_request: Mapping[str, object],
    policy: DropPolicy,
    session_lookup: Callable[[str], Awaitable[ChatCompletionSession]] | None,
) -> Mapping[str, object]:
    previous_response_id: Final = responses_request.get("previous_response_id")
    if not isinstance(previous_response_id, str) or not previous_response_id:
        return chat_body
    instructions: Final = responses_request.get("instructions")
    return await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(  # pyright: ignore[reportUnknownMemberType]  # the session handler takes a bare dict request
        previous_response_id=previous_response_id,
        litellm_completion_request={**chat_body, "model": model, "custom_llm_provider": custom_llm_provider},
        instructions=instructions if isinstance(instructions, str) else None,
        session_lookup=session_lookup,
        request_kwargs={"drop_params": policy.drop_params},
        previous_response_client_ids=previous_response_client_ids(previous_response_id, responses_request),
        input_replaces_history=holds_compaction_summary(responses_request.get("input")),
    )


def _counted_params(
    chat_body: Mapping[str, object], tools: tuple[Mapping[str, object], ...], policy: DropPolicy
) -> Mapping[str, JsonValue]:
    sent: Final = {
        name: chat_body[name]
        for name in _COUNTED_CHAT_PARAMS
        if chat_body.get(name) is not None and name not in policy.additional_drop_params
    }
    return _JSON_PARAMS.validate_python(
        {**sent, **({"drop_params": True} if policy.drop_params and (sent or tools) else {})}
    )


async def chat_count_request(
    model: str,
    custom_llm_provider: str | None,
    api_base: str | None,
    responses_request: Mapping[str, object],
    policy: DropPolicy,
    session_lookup: Callable[[str], Awaitable[ChatCompletionSession]] | None = None,
) -> ChatCountRequest | ChatCountBridgeError | None:
    """The chat request Responses inference sends a provider with no native Responses API, or None
    when inference sends this provider the Responses body itself."""
    if custom_llm_provider is None or (
        ProviderConfigManager.get_provider_responses_api_config(
            provider=custom_llm_provider, model=model, api_base=api_base
        )
        is not None
    ):
        return None
    try:
        chat_body: Final = await _with_previous_turns(
            responses_batch_body_to_chat_body(
                _with_compaction_as_summary_text(
                    _with_emulated_file_search(_with_hoisted_additional_tools(responses_request))
                ),
                custom_llm_provider,
                policy.drop_params,
            ),
            model,
            custom_llm_provider,
            responses_request,
            policy,
            session_lookup,
        )
        tools: Final = (
            () if "tools" in policy.additional_drop_params else tuple(map(_unmarked, _mappings(chat_body.get("tools"))))
        )
        return ChatCountRequest(
            messages=tuple(map(_unmarked, _mappings(chat_body.get("messages")))),
            tools=tools or None,
            params=_counted_params(chat_body, tools, policy),
        )
    except BadRequestError as e:
        return ChatCountBridgeError(message=str(e.message))
    except (ValidationError, ValueError, KeyError) as e:
        return ChatCountBridgeError(message=f"Invalid request for token counting: {e}")
