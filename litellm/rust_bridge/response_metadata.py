from typing import Final, TypeVar

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
