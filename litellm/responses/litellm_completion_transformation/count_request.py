from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from litellm.llms.base_llm.files.batch_records import responses_batch_body_to_chat_body
from litellm.responses.file_search.emulated_handler import replace_file_search_tools
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


def _with_emulated_file_search(responses_request: Mapping[str, object]) -> Mapping[str, object]:
    tools: Final = responses_request.get("tools")
    if tools is None:
        return responses_request
    return {**responses_request, "tools": replace_file_search_tools(_ITEMS.validate_python(tools))[0]}


def chat_count_request(
    model: str,
    custom_llm_provider: str | None,
    api_base: str | None,
    responses_request: Mapping[str, object],
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
        chat_body: Final = responses_batch_body_to_chat_body(
            _with_emulated_file_search(responses_request), custom_llm_provider
        )
        tools: Final = _mappings(chat_body.get("tools"))
        return ChatCountRequest(
            messages=_mappings(chat_body.get("messages")),
            tools=tools or None,
            params=_JSON_PARAMS.validate_python(
                {name: chat_body[name] for name in _COUNTED_CHAT_PARAMS if chat_body.get(name) is not None}
            ),
        )
    except (ValidationError, ValueError, KeyError) as e:
        return ChatCountBridgeError(message=f"Invalid request for token counting: {e}")
