"""
Helpers for the LIT-3193 OTEL HTTP-attribute matrix.

Module split from ``conftest.py`` because pytest auto-discovers fixtures but
forbids ``from .conftest import …`` (no parent package). Fixtures stay in
``conftest.py``; pure helpers (assertions, exception factories, attribute
constants) live here so test modules can ``from ._helpers import …``.
"""

from typing import Any, Optional

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from litellm.integrations.opentelemetry import (
    HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE,
    HTTP_ROUTE_ATTRIBUTE,
    LITELLM_PROXY_REQUEST_SPAN_NAME,
    URL_PATH_ATTRIBUTE,
)


def get_server_span(exporter: InMemorySpanExporter):
    """Return the (single) finished SERVER span, or None if it never ended."""
    for s in exporter.get_finished_spans():
        if s.name == LITELLM_PROXY_REQUEST_SPAN_NAME:
            return s
    return None


def assert_server_span_attrs(
    exporter: InMemorySpanExporter,
    *,
    expected_status: int,
    expected_url_path: str,
    expected_http_route: Optional[str] = None,
    where: str = "",
) -> None:
    """The four required attributes on the SERVER span must all be set."""
    span = get_server_span(exporter)
    assert span is not None, (
        f"{where}: SERVER span never finished — exporter saw "
        f"{[s.name for s in exporter.get_finished_spans()]}"
    )

    actual_status = span.attributes.get(HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE)
    assert actual_status == expected_status, (
        f"{where}: {HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE} = "
        f"{actual_status!r}, expected {expected_status}"
    )
    assert isinstance(
        actual_status, int
    ), f"{where}: status code must be int (semconv), got {type(actual_status)}"

    actual_url = span.attributes.get(URL_PATH_ATTRIBUTE)
    assert actual_url == expected_url_path, (
        f"{where}: {URL_PATH_ATTRIBUTE} = {actual_url!r}, "
        f"expected {expected_url_path!r}"
    )

    expected_route = expected_http_route or expected_url_path
    actual_route = span.attributes.get(HTTP_ROUTE_ATTRIBUTE)
    assert actual_route == expected_route, (
        f"{where}: {HTTP_ROUTE_ATTRIBUTE} = {actual_route!r}, "
        f"expected {expected_route!r}"
    )

    duration_ns = (span.end_time or 0) - (span.start_time or 0)
    assert duration_ns > 0, f"{where}: duration must be > 0, got {duration_ns}ns"

    expected_span_status = StatusCode.ERROR if expected_status >= 400 else StatusCode.OK
    actual_span_status = span.status.status_code
    assert actual_span_status == expected_span_status, (
        f"{where}: span.status = {actual_span_status!r}, "
        f"expected {expected_span_status!r}"
    )


# ---------------------------------------------------------------------------
# Synthetic exceptions covering the matrix triggers
# ---------------------------------------------------------------------------
class HttpStatusException(Exception):
    """Generic exception with .status_code; mirrors what proxy code reads."""

    def __init__(self, status_code: int, message: str = "boom"):
        super().__init__(message)
        self.status_code = status_code
        self.code = status_code


def make_httpx_status_error(status_code: int, body: str = "upstream error"):
    """Real httpx.HTTPStatusError — what providers emit on 4xx/5xx upstream."""
    import httpx

    request = httpx.Request("POST", "https://upstream.example/v1/x")
    response = httpx.Response(
        status_code=status_code, content=body.encode("utf-8"), request=request
    )
    return httpx.HTTPStatusError(
        f"HTTP {status_code}", request=request, response=response
    )


def make_fastapi_http_exception(status_code: int, detail: Any = "boom"):
    from fastapi import HTTPException

    return HTTPException(status_code=status_code, detail=detail)
