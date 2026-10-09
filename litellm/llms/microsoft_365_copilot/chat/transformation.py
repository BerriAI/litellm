from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    cast,  # noqa: TID251  # AllMessageValues variants are read-only TypedDict mappings
)

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm.utils as litellm_utils
from litellm.constants import MICROSOFT_365_COPILOT_DEFAULT_TIME_ZONE
from litellm.litellm_core_utils.oauth_token_exchange import redact_sensitive_values
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.microsoft_365_copilot.common_utils import Microsoft365CopilotError
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage


class _TokenCounter(Protocol):
    def __call__(
        self,
        model: str = "",
        *,
        messages: Sequence[AllMessageValues | Message] | None = None,
        text: str | None = None,
        count_response_tokens: bool | None = False,
    ) -> int: ...


_TOKEN_COUNTER: Final = cast(  # cast-ok: only the typed token-counter arguments used here are narrowed locally
    _TokenCounter, litellm_utils.token_counter
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer


class _TextPart(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    type: Literal["text"]
    text: str


class _GraphConversation(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    id: str


class _GraphMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    text: str | None = None


class _GraphConversationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    messages: Sequence[_GraphMessage] | None = None


class _GraphError(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    message: str | None = None


class _GraphErrorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    error: _GraphError | None = None


class _GraphRequestMessage(TypedDict):
    text: ReadOnly[str]


class _GraphAdditionalContext(TypedDict):
    text: ReadOnly[str]
    description: ReadOnly[str]


class _GraphLocationHint(TypedDict):
    timeZone: ReadOnly[str]


class GraphChatRequest(TypedDict):
    message: ReadOnly[_GraphRequestMessage]
    locationHint: ReadOnly[_GraphLocationHint]
    additionalContext: NotRequired[ReadOnly[Sequence[_GraphAdditionalContext]]]


_TEXT_PART_ADAPTER: Final = TypeAdapter(_TextPart)
_GRAPH_CONVERSATION_ADAPTER: Final = TypeAdapter(_GraphConversation)
_GRAPH_RESPONSE_ADAPTER: Final = TypeAdapter(_GraphConversationResponse)
_GRAPH_ERROR_ADAPTER: Final = TypeAdapter(_GraphErrorResponse)
_GRAPH_CHAT_REQUEST_ADAPTER: Final = TypeAdapter(GraphChatRequest)
_CONTENT_PARTS_ADAPTER: Final = TypeAdapter(list[object])
_JSON_VALUE_ADAPTER: Final = TypeAdapter(object)


class Microsoft365CopilotChatConfig(BaseConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseConfig requires a list return
        return ["stream", "max_tokens", "max_completion_tokens"]

    def transform_request(
        self,
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: BaseConfig requires a mutable JSON dictionary
        return {**build_chat_request(messages=messages, optional_params=optional_params)}

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "Logging",
        request_data: Mapping[str, object],
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        return map_graph_response(
            graph_response=_response_json(raw_response.content),
            model=model,
            messages=messages,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, object] | httpx.Headers,
    ) -> BaseLLMException:
        return Microsoft365CopilotError(status_code=status_code, message=error_message)

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseConfig requires a mutable parameter dictionary
        return dict(optional_params)

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: BaseConfig requires a mutable header dictionary
        return dict(headers)


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise Microsoft365CopilotError(status_code=400, message="only text content is supported")
    try:
        content_parts: Final = _CONTENT_PARTS_ADAPTER.validate_python(content)
        text_parts: Final = tuple(_TEXT_PART_ADAPTER.validate_python(part) for part in content_parts)
    except ValidationError:
        raise Microsoft365CopilotError(status_code=400, message="only text content is supported") from None
    return "\n".join(part.text for part in text_parts)


def _message_content(message: AllMessageValues) -> object:
    message_fields: Final = cast(  # cast-ok: AllMessageValues variants are read-only TypedDict mappings
        Mapping[str, object],
        message,
    )
    return message_fields.get("content")


def _response_json(response_content: bytes) -> object:
    try:
        return _JSON_VALUE_ADAPTER.validate_json(response_content)
    except ValidationError:
        return {}


def build_chat_request(
    messages: Sequence[AllMessageValues],
    optional_params: Mapping[str, object],
) -> GraphChatRequest:
    last_user_index: Final = next(
        (index for index in reversed(range(len(messages))) if messages[index]["role"] == "user"),
        None,
    )
    if last_user_index is None:
        raise Microsoft365CopilotError(status_code=400, message="at least one message must have role 'user'")
    last_user_message: Final = messages[last_user_index]
    history: Final = tuple(
        {
            "text": _content_to_text(_message_content(message)),
            "description": f"{message['role']} message",
        }
        for index, message in enumerate(messages)
        if index != last_user_index
    )
    requested_time_zone: Final = optional_params.get("time_zone")
    time_zone: Final = (
        requested_time_zone
        if isinstance(requested_time_zone, str) and requested_time_zone
        else MICROSOFT_365_COPILOT_DEFAULT_TIME_ZONE
    )
    request_data: Final = {
        "message": {"text": _content_to_text(_message_content(last_user_message))},
        "locationHint": {"timeZone": time_zone},
        **({"additionalContext": list(history)} if history else {}),
    }
    try:
        return _GRAPH_CHAT_REQUEST_ADAPTER.validate_python(request_data)
    except ValidationError:
        raise Microsoft365CopilotError(status_code=400, message="only text content is supported") from None


def _collapse_doubled_reply(text: str) -> str:
    if not text or len(text) % 2 != 0:
        return text
    half: Final = len(text) // 2
    return text[:half] if text[:half] == text[half:] else text


def map_graph_response(
    graph_response: object,
    model: str,
    messages: Sequence[AllMessageValues],
) -> ModelResponse:
    try:
        parsed_response: Final = _GRAPH_RESPONSE_ADAPTER.validate_python(graph_response)
    except ValidationError:
        raise Microsoft365CopilotError(
            status_code=502,
            message="Microsoft Graph returned an invalid Copilot conversation response",
        ) from None
    if not parsed_response.messages:
        raise Microsoft365CopilotError(
            status_code=502,
            message="Microsoft Graph returned a Copilot response without messages",
        )
    reply: Final = parsed_response.messages[-1].text
    if reply is None:
        raise Microsoft365CopilotError(
            status_code=502,
            message="Microsoft Graph returned a Copilot response without reply text",
        )
    normalized_reply: Final = _collapse_doubled_reply(reply)
    prompt_tokens: Final = _TOKEN_COUNTER(model=model, messages=messages)
    completion_tokens: Final = _TOKEN_COUNTER(
        model=model,
        text=normalized_reply,
        count_response_tokens=True,
    )
    return ModelResponse(
        model=model,
        choices=[
            Choices(
                index=0,
                message=Message(content=normalized_reply, role="assistant"),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def parse_graph_conversation_id(graph_conversation: object) -> str:
    try:
        conversation: Final = _GRAPH_CONVERSATION_ADAPTER.validate_python(graph_conversation)
    except ValidationError:
        raise Microsoft365CopilotError(
            status_code=502,
            message="Microsoft Graph returned an invalid Copilot conversation response",
        ) from None
    if not conversation.id:
        raise Microsoft365CopilotError(
            status_code=502,
            message="Microsoft Graph returned a Copilot conversation without an id",
        )
    return conversation.id


def _last_message_text(graph_response: object) -> str | None:
    try:
        parsed_response: Final = _GRAPH_RESPONSE_ADAPTER.validate_python(graph_response)
    except ValidationError:
        return None
    if not parsed_response.messages:
        return None
    return parsed_response.messages[-1].text


def extract_graph_error_message(
    response_body: object,
    sensitive_values: tuple[str, ...] = (),
) -> str:
    try:
        parsed_error: Final = _GRAPH_ERROR_ADAPTER.validate_python(response_body)
    except ValidationError:
        return "Microsoft Graph request failed"
    message: Final = parsed_error.error.message if parsed_error.error is not None else None
    if message is None:
        return "Microsoft Graph request failed"
    try:
        nested_response: Final = _JSON_VALUE_ADAPTER.validate_json(message)
    except ValidationError:
        return redact_sensitive_values(message, sensitive_values)
    nested_reply: Final = _last_message_text(nested_response)
    return redact_sensitive_values(nested_reply if nested_reply is not None else message, sensitive_values)
