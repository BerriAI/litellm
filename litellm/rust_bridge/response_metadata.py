from collections.abc import Mapping, Sequence
from typing import Final, TypeVar

import httpx
from pydantic import TypeAdapter

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


def with_provider_headers(response: ResultT, headers: Sequence[tuple[str, str]]) -> ResultT:
    """Record the provider's forwardable headers the way the Python handlers do.

    `_hidden_params["headers"]` keeps the raw names, `additional_headers` carries the
    `llm_provider-*` names the proxy forwards. Repeated values are joined, as `httpx` does.
    """
    if not headers:
        return response
    raw: Final = httpx.Headers(list(headers))
    set_hidden_param(response, "headers", dict(raw.items()))
    _add_headers_to_response(
        response, dict(_RESPONSE_HEADERS.validate_python(process_response_headers(raw), strict=True))
    )
    return response
