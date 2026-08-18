"""LIT-3193 — admin / management endpoints. Drives the
async_management_endpoint_{success,failure}_hook integration points."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from litellm.proxy._types import (
    ManagementEndpointLoggingPayload,
    UserAPIKeyAuth,
)

from ._helpers import (
    HttpStatusException,
    assert_server_span_attrs,
    get_server_span,
    make_fastapi_http_exception,
    make_httpx_status_error,
)


def _real_user_api_key_dict(parent_span):
    return UserAPIKeyAuth(
        api_key="sk-test-admin",
        team_id="team-lit-3193",
        team_alias="lit-3193-team",
        parent_otel_span=parent_span,
    )


async def _noop_alert(*args, **kwargs):
    return None


async def _drive_admin_failure(*, otel, exception, parent_span, route):
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


KEY_GENERATE_PATH = "/key/generate"


@pytest.mark.parametrize(
    "exception, expected_status",
    [
        (make_fastapi_http_exception(400, "negative max_budget"), 400),
        (make_fastapi_http_exception(401, "missing master key"), 401),
        (make_fastapi_http_exception(403, "non-admin"), 403),
        (make_fastapi_http_exception(422, "validation"), 422),
        (HttpStatusException(500, "DB unreachable"), 500),
        # Pins .response.status_code fallback through the admin path.
        (make_httpx_status_error(500, "upstream blew up"), 500),
    ],
    ids=["400", "401", "403", "422", "500", "500-httpx"],
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


def test_management_wrapper_success_ends_server_span_without_http_request(
    server_span_factory, otel_with_exporter, monkeypatch
):
    """Regression: management endpoints whose handler does not declare an
    ``http_request`` parameter (``/key/generate``, ``/user/new``, ``/mcp/*``,
    ...) must still get their parent SERVER span stamped + ended on success.

    The success hook itself stamps 200 and ``end()``s the parent, but the
    wrapper only invoked it when ``http_request`` was present — so on success
    the span (created in auth) was never ended and never exported. This drives
    the real wrapper around an ``http_request``-less handler and asserts the
    SERVER span reaches the exporter with status 200.
    """
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.management_helpers import utils as mgmt_utils

    otel, exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", otel, raising=False)
    monkeypatch.setattr(mgmt_utils, "send_management_endpoint_alert", _noop_alert)

    server_span = server_span_factory(KEY_GENERATE_PATH)

    @mgmt_utils.management_endpoint_wrapper
    async def fake_generate_key_fn(data=None, user_api_key_dict=None):
        # No ``http_request`` parameter — mirrors generate_key_fn et al.
        return {"key": "sk-xyz", "key_name": "k"}

    asyncio.run(
        fake_generate_key_fn(
            data={},
            user_api_key_dict=_real_user_api_key_dict(server_span),
        )
    )

    assert_server_span_attrs(
        exporter,
        expected_status=200,
        expected_url_path=KEY_GENERATE_PATH,
        where="management wrapper success without http_request",
    )


def test_management_wrapper_failure_ends_server_span(
    server_span_factory, otel_with_exporter, monkeypatch
):
    """When the handler raises, the wrapper must route through the failure hook
    and stamp + end the parent SERVER span with the error status — even for an
    ``http_request``-less handler (route falls back to ``func.__name__``)."""
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.management_helpers import utils as mgmt_utils

    otel, exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", otel, raising=False)

    server_span = server_span_factory(KEY_GENERATE_PATH)

    @mgmt_utils.management_endpoint_wrapper
    async def failing_fn(data=None, user_api_key_dict=None):
        raise HttpStatusException(500, "boom")

    with pytest.raises(HttpStatusException):
        asyncio.run(
            failing_fn(data={}, user_api_key_dict=_real_user_api_key_dict(server_span))
        )

    assert_server_span_attrs(
        exporter,
        expected_status=500,
        expected_url_path=KEY_GENERATE_PATH,
        where="management wrapper failure",
    )


def test_management_wrapper_success_with_http_request(
    server_span_factory, otel_with_exporter, monkeypatch
):
    """Cover the branch where the handler DOES declare ``http_request``: the
    route comes from ``http_request.url.path`` and the body is read from it."""
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.management_helpers import utils as mgmt_utils

    otel, exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", otel, raising=False)
    monkeypatch.setattr(mgmt_utils, "send_management_endpoint_alert", _noop_alert)

    async def _fake_body(request=None):
        return {"team_alias": "t"}

    monkeypatch.setattr(mgmt_utils, "_read_request_body", _fake_body)

    server_span = server_span_factory("/team/new")
    http_request = MagicMock()
    http_request.url.path = "/team/new"

    @mgmt_utils.management_endpoint_wrapper
    async def fake_new_team(data=None, http_request=None, user_api_key_dict=None):
        return {"team_id": "t-1"}

    asyncio.run(
        fake_new_team(
            data={},
            http_request=http_request,
            user_api_key_dict=_real_user_api_key_dict(server_span),
        )
    )

    assert_server_span_attrs(
        exporter,
        expected_status=200,
        expected_url_path="/team/new",
        where="management wrapper success with http_request",
    )


def test_management_wrapper_noop_when_otel_logger_absent(
    server_span_factory, otel_with_exporter, monkeypatch
):
    """When no OTEL logger is registered, the helper early-returns and no SERVER
    span is exported — and the handler result is still returned unchanged."""
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.management_helpers import utils as mgmt_utils

    _otel, exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", None, raising=False)
    monkeypatch.setattr(mgmt_utils, "send_management_endpoint_alert", _noop_alert)

    server_span = server_span_factory(KEY_GENERATE_PATH)

    @mgmt_utils.management_endpoint_wrapper
    async def fake_fn(data=None, user_api_key_dict=None):
        return {"ok": True}

    result = asyncio.run(
        fake_fn(data={}, user_api_key_dict=_real_user_api_key_dict(server_span))
    )

    assert result == {"ok": True}
    assert get_server_span(exporter) is None


def test_management_wrapper_swallows_post_success_errors(
    server_span_factory, otel_with_exporter, monkeypatch
):
    """A failure in post-success bookkeeping (cache invalidation, alerting) must
    not propagate — the handler result is returned regardless (non-blocking)."""
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.management_helpers import utils as mgmt_utils

    otel, _exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", otel, raising=False)
    monkeypatch.setattr(mgmt_utils, "send_management_endpoint_alert", _noop_alert)

    def _boom(*args, **kwargs):
        raise RuntimeError("cache backend down")

    monkeypatch.setattr(mgmt_utils, "_delete_api_key_from_cache", _boom)

    server_span = server_span_factory(KEY_GENERATE_PATH)

    @mgmt_utils.management_endpoint_wrapper
    async def fake_fn(data=None, user_api_key_dict=None):
        return {"ok": True}

    result = asyncio.run(
        fake_fn(data={}, user_api_key_dict=_real_user_api_key_dict(server_span))
    )

    assert result == {"ok": True}
