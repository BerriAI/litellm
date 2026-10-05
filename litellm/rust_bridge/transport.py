from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import process_response_headers

_HEADERS: Final = TypeAdapter(dict[str, str])


def hidden_params(headers: httpx.Headers, status: int) -> dict[str, object]:
    return {
        "additional_headers": _HEADERS.validate_python(process_response_headers(headers)),
        "response_headers": headers,
        "status_code": status,
    }
