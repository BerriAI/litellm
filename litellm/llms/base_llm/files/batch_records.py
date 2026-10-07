from collections.abc import Iterable, Mapping
from functools import cache
from types import MappingProxyType
from typing import Final, cast, get_type_hints

from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams


def _frozen_mapping(items: Iterable[tuple[str, object]]) -> Mapping[str, object]:
    return MappingProxyType(dict(items))


@cache
def _responses_request_keys() -> frozenset[str]:
    return frozenset(get_type_hints(ResponsesAPIOptionalRequestParams))


def responses_batch_body_to_chat_body(
    openai_request_body: Mapping[str, object],
    custom_llm_provider: str | None = None,
) -> dict[str, object]:  # mutable-ok: provider transforms take the bridged chat body as a plain dict
    """
    Rewrite the body of an OpenAI `/v1/responses` batch record as a Chat Completions body.

    Batch providers translate chat bodies into their own request shape, so a Responses
    record goes through the same Responses-to-Chat bridge the real-time path uses for
    providers without a native Responses API: `input`, `instructions`, `max_output_tokens`
    and the tool params translate identically in batch and real time. Like real time, the
    record's fields are forwarded as sent instead of validated against the SDK TypedDicts,
    whose required keys (a function tool's `strict`, an image part's `detail`) clients omit.
    """
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    responses_input: Final = openai_request_body.get("input")
    if responses_input is None:
        raise ValueError(
            "Batch record for /v1/responses is missing required `input` field: "
            f"model={openai_request_body.get('model', '')}"
        )
    model: Final = openai_request_body.get("model")
    chat_input: Final = cast(str | ResponseInputParam, responses_input)  # cast-ok: forwarded as sent
    responses_request: Final = cast(  # cast-ok: client-supplied fields forwarded verbatim, as real time does
        ResponsesAPIOptionalRequestParams,
        _frozen_mapping((key, value) for key, value in openai_request_body.items() if key in _responses_request_keys()),
    )
    return LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # transformer declares a bare dict return
        model=model if isinstance(model, str) else "",
        input=chat_input,
        responses_api_request=responses_request,
        custom_llm_provider=custom_llm_provider,
        metadata=openai_request_body.get("metadata"),
    )
