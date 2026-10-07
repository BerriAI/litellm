from typing import TypeVar

from litellm.router_utils.add_retry_fallback_headers import (
    _add_headers_to_response,  # pyright: ignore[reportPrivateUsage]  # reuse the proxy's identity-preserving response metadata writer
)

ResultT = TypeVar("ResultT")


def mark_rust_response(response: ResultT) -> ResultT:
    _add_headers_to_response(response, {"x-litellm-rust": "true"})
    return response
