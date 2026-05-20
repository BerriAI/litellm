"""
LIT-3193 — Class 3: admin / management endpoints.

Admin endpoints don't go through ``_handle_llm_api_exception``. They use
``management_endpoint_wrapper`` (litellm/proxy/management_helpers/utils.py),
which catches every exception, calls
``open_telemetry_logger.async_management_endpoint_failure_hook``, and
re-raises. That hook is the OTEL integration point — both for failures
(4xx admin gap from the ticket) and successes.

Each cell asserts the four SERVER-span attributes
(``http.response.status_code``, ``url.path``, ``http.route``, duration > 0)
on the parent span the auth layer opened, just like the unified and
passthrough classes.
"""

import asyncio
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import (
    ManagementEndpointLoggingPayload,
    ProxyException,
    UserAPIKeyAuth,
)

from ._helpers import (
    HttpStatusException,
    assert_server_span_attrs,
    make_fastapi_http_exception,
)


def _real_user_api_key_dict(parent_span):
    return UserAPIKeyAuth(
        api_key="sk-test-admin",
        team_id="team-lit-3193",
        team_alias="lit-3193-team",
        parent_otel_span=parent_span,
    )


async def _drive_admin_failure(*, otel, exception, parent_span, route):
    """Mirror ``management_endpoint_wrapper``'s ``except Exception`` block."""
    payload = ManagementEndpointLoggingPayload(
        route=route,
        request_data={},
        response=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
        exception=exception,
    )
    await otel.async_management_endpoint_failure_hook(
        logging_payload=payload,
        parent_otel_span=parent_span,
    )


async def _drive_admin_success(*, otel, parent_span, route, response):
    """Mirror ``management_endpoint_wrapper``'s success block."""
    payload = ManagementEndpointLoggingPayload(
        route=route,
        request_data={},
        response=response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    await otel.async_management_endpoint_success_hook(
        logging_payload=payload,
        parent_otel_span=parent_span,
    )


# ---------------------------------------------------------------------------
# /key/generate — deep matrix
# ---------------------------------------------------------------------------
KEY_GENERATE_PATH = "/key/generate"


@pytest.mark.parametrize(
    "exception, expected_status",
    [
        (make_fastapi_http_exception(400, "negative max_budget"), 400),
        (make_fastapi_http_exception(401, "missing master key"), 401),
        (make_fastapi_http_exception(403, "non-admin"), 403),
        (make_fastapi_http_exception(422, "validation"), 422),
        (HttpStatusException(500, "DB unreachable"), 500),
    ],
    ids=["400", "401", "403", "422", "500"],
)
def test_key_generate_failure_stamps_server_span(
    exception,
    expected_status,
    server_span_factory,
    otel_with_exporter,
):
    otel, exporter = otel_with_exporter
    server_span = server_span_factory(KEY_GENERATE_PATH)

    asyncio.run(
        _drive_admin_failure(
            otel=otel,
            exception=exception,
            parent_span=server_span,
            route=KEY_GENERATE_PATH,
        )
    )

    assert_server_span_attrs(
        exporter,
        expected_status=expected_status,
        expected_url_path=KEY_GENERATE_PATH,
        where=f"key/generate {expected_status}",
    )


def test_key_generate_success_stamps_server_span(
    server_span_factory, otel_with_exporter
):
    otel, exporter = otel_with_exporter
    server_span = server_span_factory(KEY_GENERATE_PATH)

    asyncio.run(
        _drive_admin_success(
            otel=otel,
            parent_span=server_span,
            route=KEY_GENERATE_PATH,
            response={"key": "sk-1", "key_name": "k"},
        )
    )

    assert_server_span_attrs(
        exporter,
        expected_status=200,
        expected_url_path=KEY_GENERATE_PATH,
        where="key/generate 200",
    )


# ---------------------------------------------------------------------------
# Smoke matrix across other admin endpoints — one 4xx + one 5xx each
# ---------------------------------------------------------------------------
SMOKE_ADMIN_ENDPOINTS = [
    "/key/info",
    "/key/update",
    "/key/delete",
    "/team/new",
    "/team/member_add",
    "/user/new",
    "/user/info",
    "/model/new",
    "/model/delete",
    "/customer/new",
    "/customer/info",
    "/organization/new",
    "/organization/member_add",
    "/budget/new",
    "/budget/info",
    "/credentials/new",
    "/mcp/server/add",
    "/tag/new",
]


@pytest.mark.parametrize("path", SMOKE_ADMIN_ENDPOINTS)
@pytest.mark.parametrize(
    "exception, expected_status",
    [
        (make_fastapi_http_exception(404, "not found"), 404),
        (HttpStatusException(500, "DB unreachable"), 500),
    ],
    ids=["404", "500"],
)
def test_admin_endpoint_failure_stamps_server_span(
    path,
    exception,
    expected_status,
    server_span_factory,
    otel_with_exporter,
):
    """Confirm SERVER-span stamping works for every admin resource family —
    same wrapper, just different routes."""
    otel, exporter = otel_with_exporter
    server_span = server_span_factory(path)

    asyncio.run(
        _drive_admin_failure(
            otel=otel,
            exception=exception,
            parent_span=server_span,
            route=path,
        )
    )

    assert_server_span_attrs(
        exporter,
        expected_status=expected_status,
        expected_url_path=path,
        where=f"{path} {expected_status}",
    )
