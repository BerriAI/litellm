"""An upstream whose first byte takes longer than the request allows, the way a slow model does.

httpx hands every transport the request's timeout in ``request.extensions["timeout"]``, so this one honours it
in process the way a socket would: a read timeout shorter than the first byte's latency times out, a longer one
gets the answer.
"""

from __future__ import annotations

from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

STREAM_TIMEOUT_SECONDS: Final = 0.5
UPSTREAM_FIRST_BYTE_SECONDS: Final = 2.0

_TIMEOUT_EXTENSION: Final = TypeAdapter(dict[str, float | None])


def _answer_once_the_first_byte_is_due(request: httpx.Request) -> httpx.Response:
    read_timeout: Final = _TIMEOUT_EXTENSION.validate_python(request.extensions["timeout"])["read"]
    if read_timeout is not None and read_timeout < UPSTREAM_FIRST_BYTE_SECONDS:
        raise httpx.ReadTimeout(f"no byte arrived within {read_timeout}s", request=request)
    return httpx.Response(200, content=b"", request=request)


def slow_upstream_async_client() -> AsyncHTTPHandler:
    return AsyncHTTPHandler(transport=httpx.MockTransport(_answer_once_the_first_byte_is_due))


def slow_upstream_sync_client() -> HTTPHandler:
    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_answer_once_the_first_byte_is_due)))
