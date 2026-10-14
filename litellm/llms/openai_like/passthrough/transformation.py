import itertools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, TypeAlias

from httpx import Response
from pydantic import ConfigDict, ValidationError

import litellm.litellm_core_utils.token_counter as token_counting
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.passthrough.transformation import RelayShape, logged_relay_shape
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesTerminalEvent
from litellm.types.utils import (
    CallTypes,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    TextChoices,
    TextCompletionResponse,
    Usage,
)

if TYPE_CHECKING:
    from litellm.llms.base_llm.passthrough.transformation import LoggedRelayResponse


RelayedPrompt: TypeAlias = str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]]


def _count_text_tokens(model: str, text: str, count_response_tokens: bool = False) -> int:
    return token_counting.token_counter(  # pyright: ignore[reportUnknownMemberType]  # token_counter's custom_tokenizer param is an untyped dict
        model=model, text=text, count_response_tokens=count_response_tokens
    )


class RelayedRequest(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    messages: Sequence[Mapping[str, object]] | None = None
    prompt: RelayedPrompt | None = None


class RelayedCallDetails(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    request_data: RelayedRequest | None = None


def _relayed_request(litellm_logging_obj: Logging) -> RelayedRequest | None:
    try:
        return RelayedCallDetails.model_validate(litellm_logging_obj.model_call_details).request_data
    except ValidationError:
        return None


def relayed_messages(litellm_logging_obj: Logging) -> Sequence[Mapping[str, object]] | None:
    request: Final = _relayed_request(litellm_logging_obj)
    return request.messages if request else None


class RelayedTextChoice(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = 0
    text: str = ""
    finish_reason: str | None = None


class RelayedTextCompletionChunk(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    created: int
    model: str
    choices: Sequence[RelayedTextChoice] = ()
    usage: Usage | None = None


RESPONSES_RELAY_SHAPE: Final = RelayShape("/responses", CallTypes.aresponses, ResponsesAPIResponse.model_validate)
TEXT_COMPLETION_RELAY_SHAPE: Final = RelayShape(
    "/completions", CallTypes.atext_completion, TextCompletionResponse.model_validate
)

OPENAI_RELAY_SHAPES: Final = (
    RelayShape("/embeddings", CallTypes.aembedding, EmbeddingResponse.model_validate),
    RESPONSES_RELAY_SHAPE,
    RelayShape("/images/generations", CallTypes.aimage_generation, ImageResponse.model_validate),
)


def relays_text_completion(endpoint: str) -> bool:
    relayed_path: Final = f"/{endpoint.strip('/')}"
    return relayed_path.endswith(TEXT_COMPLETION_RELAY_SHAPE.path_suffix) and not relayed_path.endswith(
        "/chat/completions"
    )


def logged_responses_stream(all_chunks: Sequence[str], logging_obj: Logging) -> ResponsesTerminalEvent | None:
    """A streaming logging object assembles the logged response from the terminal event, not from its body."""
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    terminal_event: Final = OpenAIResponsesAPIConfig.parse_terminal_event_from_stream_chunks(all_chunks=all_chunks)
    if terminal_event is None:
        return None
    logging_obj.call_type = RESPONSES_RELAY_SHAPE.call_type.value
    return terminal_event


def _relayed_text_chunk(line: str) -> RelayedTextCompletionChunk | None:
    payload: Final = BaseModelResponseIterator.string_to_dict_parser(str_line=line)
    if not payload:
        return None
    try:
        return RelayedTextCompletionChunk.model_validate(payload)
    except ValidationError:
        return None


def _assembled_text_choice(index: int, own: Sequence[RelayedTextChoice]) -> RelayedTextChoice:
    return RelayedTextChoice(
        index=index,
        text="".join(delta.text for delta in own),
        finish_reason=next((delta.finish_reason for delta in reversed(own) if delta.finish_reason), None),
    )


def _delta_index(delta: RelayedTextChoice) -> int:
    return delta.index


def _prompt_part_tokens(model: str, part: str | int | Sequence[int]) -> int:
    match part:
        case str():
            return _count_text_tokens(model=model, text=part)
        case int():
            return 1
        case _:
            return len(part)


def _prompt_tokens(model: str, prompt: RelayedPrompt | None) -> int:
    match prompt:
        case None:
            return 0
        case str():
            return _count_text_tokens(model=model, text=prompt)
        case _:
            return sum(_prompt_part_tokens(model, part) for part in prompt)


def _counted_text_usage(model: str, prompt: RelayedPrompt | None, choices: Sequence[RelayedTextChoice]) -> Usage:
    prompt_tokens: Final = _prompt_tokens(model, prompt)
    completion_tokens: Final = sum(
        _count_text_tokens(model=model, text=choice.text, count_response_tokens=True) for choice in choices
    )
    return Usage(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens
    )


def logged_text_completion_response(
    httpx_response: Response, logging_obj: Logging, endpoint: str
) -> "LoggedRelayResponse | None":
    return logged_relay_shape((TEXT_COMPLETION_RELAY_SHAPE,), httpx_response, logging_obj, endpoint)


def logged_text_completion_stream(
    all_chunks: Sequence[str], logging_obj: Logging, model: str
) -> TextCompletionResponse | None:
    chunks: Final = tuple(chunk for chunk in map(_relayed_text_chunk, all_chunks) if chunk is not None)
    head: Final = next((chunk for chunk in chunks if chunk.choices), None)
    if head is None:
        return None
    deltas: Final = tuple(itertools.chain.from_iterable(chunk.choices for chunk in chunks))
    choices: Final = tuple(
        _assembled_text_choice(index, tuple(own))
        for index, own in itertools.groupby(sorted(deltas, key=_delta_index), key=_delta_index)
    )
    reported_usage: Final = next((chunk.usage for chunk in reversed(chunks) if chunk.usage is not None), None)
    request: Final = _relayed_request(logging_obj)
    logging_obj.call_type = TEXT_COMPLETION_RELAY_SHAPE.call_type.value  # rebind-ok: the relay's logging object starts as the generic passthrough call and learns its real call type from the relayed shape
    return TextCompletionResponse(
        id=head.id,
        created=head.created,
        model=head.model,
        choices=[
            TextChoices(index=choice.index, text=choice.text, finish_reason=choice.finish_reason) for choice in choices
        ],
        usage=(
            reported_usage
            if reported_usage is not None
            else _counted_text_usage(model, request.prompt if request else None, choices)
        ),
    )


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

    relayed_path: Final = f"/{endpoint.strip('/')}"
    if relayed_path.endswith(RESPONSES_RELAY_SHAPE.path_suffix):
        return logged_responses_stream(all_chunks, logging_obj)
    if "chat/completions" in endpoint:
        return OpenAIPassthroughLoggingHandler()._build_complete_streaming_response(  # pyright: ignore[reportPrivateUsage]  # the only OpenAI SSE-to-ModelResponse assembler; reimplementing it would fork the parser
            all_chunks=all_chunks,
            litellm_logging_obj=logging_obj,
            model=model,
            messages=relayed_messages(logging_obj),
        )
    return None
