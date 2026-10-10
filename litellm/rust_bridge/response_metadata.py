from collections.abc import Mapping, Sequence
from typing import Final, TypedDict, TypeVar

import httpx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm.litellm_core_utils.core_helpers import (
    process_response_headers,  # pyright: ignore[reportUnknownVariableType]  # legacy signature, output validated by TypeAdapter
)
from litellm.litellm_core_utils.hidden_params import set_hidden_param
from litellm.router_utils.add_retry_fallback_headers import (
    _add_headers_to_response,  # pyright: ignore[reportPrivateUsage]  # reuse the proxy's identity-preserving response metadata writer
    get_hidden_params_dict,
)

ResultT: Final = TypeVar("ResultT")


def mark_rust_response(response: ResultT) -> ResultT:
    cache_key: Final = get_hidden_params_dict(response).get("cache_key")
    _add_headers_to_response(
        response,
        {"x-litellm-rust": "true", **({"x-litellm-cache-key": cache_key} if isinstance(cache_key, str) else {})},
    )
    return response


_RESPONSE_HEADERS: Final = TypeAdapter(Mapping[str, str])
_HEADER_PAIRS: Final = TypeAdapter(tuple[tuple[str, str], ...])


class ProviderHeaderMetadata(TypedDict):
    provider_response_headers: ReadOnly[tuple[tuple[str, str], ...]]
    additional_headers: ReadOnly[Mapping[str, object]]
    headers: ReadOnly[Mapping[str, str]]


def provider_header_metadata(
    headers: Sequence[tuple[str, str]], forwarded: Sequence[tuple[str, str]]
) -> ProviderHeaderMetadata:
    safe: Final = httpx.Headers(forwarded)
    compatible: Final = _RESPONSE_HEADERS.validate_python(process_response_headers(safe), strict=True)
    return {
        "provider_response_headers": _HEADER_PAIRS.validate_python(headers),
        "additional_headers": {**dict(safe.items()), **compatible},
        "headers": dict(safe.items()),
    }


def with_provider_headers(
    response: ResultT, headers: Sequence[tuple[str, str]], forwarded: Sequence[tuple[str, str]]
) -> ResultT:
    if not headers:
        return response
    metadata: Final = provider_header_metadata(headers, forwarded)
    set_hidden_param(response, "provider_response_headers", metadata["provider_response_headers"])
    set_hidden_param(response, "headers", metadata["headers"])
    _add_headers_to_response(response, dict(metadata["additional_headers"]))
    return response


def restore_provider_headers(response: object) -> None:
    hidden: Final = get_hidden_params_dict(response)
    if "provider_response_headers" not in hidden:
        return
    forwarded: Final = _RESPONSE_HEADERS.validate_python(hidden.get("headers", {}), strict=True)
    _add_headers_to_response(response, dict(forwarded))
