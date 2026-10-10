from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from httpx import Response
from pydantic import ConfigDict, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.passthrough.transformation import RelayShape, logged_relay_shape
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesTerminalEvent
from litellm.types.utils import CallTypes, EmbeddingResponse, ImageResponse, ModelResponse

if TYPE_CHECKING:
    from litellm.llms.base_llm.passthrough.transformation import LoggedRelayResponse


class RelayedChatRequest(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    messages: Sequence[Mapping[str, object]] | None = None


class RelayedCallDetails(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    request_data: RelayedChatRequest | None = None


def relayed_messages(litellm_logging_obj: Logging) -> Sequence[Mapping[str, object]] | None:
    try:
        details: Final = RelayedCallDetails.model_validate(litellm_logging_obj.model_call_details)
    except ValidationError:
        return None
    return details.request_data.messages if details.request_data else None


RESPONSES_RELAY_SHAPE: Final = RelayShape("/responses", CallTypes.aresponses, ResponsesAPIResponse.model_validate)

OPENAI_RELAY_SHAPES: Final = (
    RelayShape("/embeddings", CallTypes.aembedding, EmbeddingResponse.model_validate),
    RESPONSES_RELAY_SHAPE,
    RelayShape("/images/generations", CallTypes.aimage_generation, ImageResponse.model_validate),
)


def logged_responses_stream(all_chunks: Sequence[str], logging_obj: Logging) -> ResponsesTerminalEvent | None:
    """A streaming logging object assembles the logged response from the terminal event, not from its body."""
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    terminal_event: Final = OpenAIResponsesAPIConfig.parse_terminal_event_from_stream_chunks(all_chunks=all_chunks)
    if terminal_event is None:
        return None
    logging_obj.call_type = RESPONSES_RELAY_SHAPE.call_type.value
    return terminal_event


def logged_openai_like_response(
    model: str, httpx_response: Response, request_data: Mapping[str, object], logging_obj: Logging, endpoint: str
) -> "LoggedRelayResponse | None":
    from litellm import encoding
    from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

    if "chat/completions" not in endpoint:
        return logged_relay_shape(OPENAI_RELAY_SHAPES, httpx_response, logging_obj, endpoint)

    return OpenAIGPTConfig().transform_response(
        model=model,
        messages=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
        raw_response=httpx_response,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        optional_params={},
        litellm_params={},
        api_key="",
        request_data=dict(request_data),
        encoding=encoding,
    )


def logged_openai_like_stream(
    all_chunks: Sequence[str], logging_obj: Logging, model: str, endpoint: str
) -> "LoggedRelayResponse | None":
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
        OpenAIPassthroughLoggingHandler,
    )

    if f"/{endpoint.strip('/')}".endswith(RESPONSES_RELAY_SHAPE.path_suffix):
        return logged_responses_stream(all_chunks, logging_obj)
    if "chat/completions" not in endpoint:
        return None

    return OpenAIPassthroughLoggingHandler()._build_complete_streaming_response(  # pyright: ignore[reportPrivateUsage]  # the only OpenAI SSE-to-ModelResponse assembler; reimplementing it would fork the parser
        all_chunks=all_chunks,
        litellm_logging_obj=logging_obj,
        model=model,
        messages=relayed_messages(logging_obj),
    )
