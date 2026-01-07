"""
Utility functions for JSON encoding with surrogate character support.

This module provides functions to encode JSON data while preserving surrogate
characters that would otherwise cause UnicodeEncodeError with standard UTF-8 encoding.
"""

import json
from typing import Any, Optional, Tuple


def encode_json_with_surrogates(data: Any) -> bytes:
    """
    Encode a Python object to JSON bytes, preserving surrogate characters.

    Unlike standard json.dumps().encode('utf-8'), this function uses the
    'surrogatepass' error handler which allows lone surrogate characters
    (U+D800-U+DFFF) to be encoded. This is necessary because some LLM
    providers accept these characters even though they're technically
    invalid in strict UTF-8.

    Args:
        data: Python object to serialize to JSON

    Returns:
        JSON-encoded bytes with surrogates preserved
    """
    json_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return json_str.encode("utf-8", errors="surrogatepass")


def prepare_json_request(
    json_data: Optional[dict],
    headers: Optional[dict],
    content: Any,
) -> Tuple[Optional[dict], Any]:
    """
    Prepare JSON data for HTTP request, handling surrogate characters.

    If json_data is provided, converts it to bytes using surrogate-safe
    encoding and ensures Content-Type header is set.

    Args:
        json_data: Optional JSON dict to send
        headers: Optional existing headers dict
        content: Existing content parameter

    Returns:
        Tuple of (updated_headers, request_content)
    """
    if json_data is None:
        return headers, content

    # Encode JSON with surrogate support
    encoded_content = encode_json_with_surrogates(json_data)

    # Ensure Content-Type header is set
    if headers is None:
        headers = {}
    if "content-type" not in {k.lower() for k in headers}:
        headers["Content-Type"] = "application/json"

    return headers, encoded_content

