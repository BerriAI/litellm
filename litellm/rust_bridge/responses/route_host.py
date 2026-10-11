from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Final, Protocol, cast  # noqa: TID251  # bounds the untyped shared helpers this route host borrows

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils import core_helpers
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import optional_str
from litellm.types.llms.openai import (
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

_RESPONSE_HEADERS: Final = TypeAdapter(dict[str, str])


def connection_defaults(_provider: str) -> tuple[str | None, str | None]:
    return litellm.api_key or litellm.openai_key, litellm.api_base


def response(value: Mapping[str, object]) -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_validate(value)


class HeaderProcessor(Protocol):
    def __call__(self, response_headers: httpx.Headers) -> Mapping[str, object]: ...


class StreamEventTransform(Protocol):
    def __call__(
        self, *, model: str, parsed_chunk: Mapping[str, object], logging_obj: Logging
    ) -> ResponsesAPIStreamingResponse: ...


def stream_hidden_params(headers: Sequence[tuple[str, str]]) -> Mapping[str, object]:
    process: Final = cast(  # cast-ok: the shared header projection is untyped
        HeaderProcessor, core_helpers.process_response_headers
    )
    return {"additional_headers": _RESPONSE_HEADERS.validate_python(process(httpx.Headers(list(headers))))}


def _provider_and_model(request: Mapping[str, object]) -> tuple[str, str]:
    provider: Final = optional_str(request.get("custom_llm_provider")) or "openai"
    return provider, str(request["model"]).removeprefix(f"{provider}/")


def stream_decoder(request: Mapping[str, object]) -> Callable[[Mapping[str, object]], ResponsesAPIStreamingResponse]:
    """The provider config's own event transformation for this call, so a native stream yields
    the same typed events as the Python iterator."""
    provider, model = _provider_and_model(request)
    config: Final = ProviderConfigManager.get_provider_responses_api_config(
        LlmProviders(provider), model=model, api_base=optional_str(request.get("api_base"))
    )
    if config is None:
        raise ValueError(f"no Responses provider config for {provider}")
    logger: Final = request["litellm_logging_obj"]
    if not isinstance(logger, Logging):
        raise TypeError("the argument stage did not inject the Logging object")
    transform: Final = cast(  # cast-ok: the provider config's event transformation takes an untyped dict
        StreamEventTransform, config.transform_streaming_response
    )

    def decode(parsed: Mapping[str, object]) -> ResponsesAPIStreamingResponse:
        return transform(model=model, parsed_chunk=dict(parsed), logging_obj=logger)

    return decode


def map_failure(error: Exception, request: Mapping[str, object]) -> Exception:
    provider: Final = optional_str(request.get("custom_llm_provider")) or "openai"
    return failures.map_native_failure(
        error,
        str(request["model"]),
        provider,
        request,
        optional_str(request.get("api_base")) or optional_str(request.get("base_url")),
    )
