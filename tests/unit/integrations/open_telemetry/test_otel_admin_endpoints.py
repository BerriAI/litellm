"""LIT-3193 — admin / management endpoints. Drives the
async_management_endpoint_{success,failure}_hook integration points."""

import asyncio
from datetime import datetime
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import Request

from litellm import Router
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


def test_key_generate_success_stamps_server_span(server_span_factory, otel_with_exporter):
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


def test_management_wrapper_failure_ends_server_span(server_span_factory, otel_with_exporter, monkeypatch):
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
        asyncio.run(failing_fn(data={}, user_api_key_dict=_real_user_api_key_dict(server_span)))

    assert_server_span_attrs(
        exporter,
        expected_status=500,
        expected_url_path=KEY_GENERATE_PATH,
        where="management wrapper failure",
    )


def test_management_wrapper_success_with_http_request(server_span_factory, otel_with_exporter, monkeypatch):
    """Cover the branch where the handler DOES declare ``http_request``: the
    route comes from ``http_request.url.path`` and the body is read from it."""
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.management_helpers import utils as mgmt_utils

    otel, exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", otel, raising=False)
    monkeypatch.setattr(mgmt_utils, "send_management_endpoint_alert", _noop_alert)

    async def _fake_body(request=None):
        return {"team_alias": "t"}

    monkeypatch.setattr(mgmt_utils, "read_request_body", _fake_body)

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


def test_management_wrapper_noop_when_otel_logger_absent(server_span_factory, otel_with_exporter, monkeypatch):
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

    result = asyncio.run(fake_fn(data={}, user_api_key_dict=_real_user_api_key_dict(server_span)))

    assert result == {"ok": True}
    assert get_server_span(exporter) is None


def test_management_wrapper_swallows_post_success_errors(server_span_factory, otel_with_exporter, monkeypatch):
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

    result = asyncio.run(fake_fn(data={}, user_api_key_dict=_real_user_api_key_dict(server_span)))

    assert result == {"ok": True}


TEAM_NEW_PATH: Final = "/team/new"
MODEL_LIST_PATH: Final = "/v1/models"
MODEL_RETRIEVE_PATH: Final = "/v1/models/gpt-4o"
INBOUND_TRACEPARENT: Final = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def _request(method: str, path: str, body: bytes = b"") -> Request:
    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "root_path": "",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )


async def _answer_like_the_proxy(request: Request, exception: Exception) -> None:
    from litellm.proxy.proxy_server import app

    handler: Final = next(
        app.exception_handlers[cls] for cls in type(exception).__mro__ if cls in app.exception_handlers
    )
    await handler(request, exception)


@pytest.mark.parametrize(
    ("wrapper_name", "exports_payload"),
    [("management_endpoint_wrapper", True), ("management_endpoint_span_wrapper", False)],
)
def test_span_wrapper_omits_payload(server_span_factory, otel_with_exporter, monkeypatch, wrapper_name, exports_payload):
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.management_helpers import utils as mgmt_utils

    otel, exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", otel, raising=False)

    server_span: Final = server_span_factory(TEAM_NEW_PATH)
    wrap: Final = getattr(mgmt_utils, wrapper_name)

    @wrap
    async def fake_fn(http_request: Request, user_api_key_dict: UserAPIKeyAuth) -> dict[str, object]:
        return {"data": [{"id": "gpt-4o"}]}

    asyncio.run(
        fake_fn(
            http_request=_request("POST", TEAM_NEW_PATH, b'{"team_alias": "t"}'),
            user_api_key_dict=_real_user_api_key_dict(server_span),
        )
    )

    spans: Final = [s for s in exporter.get_finished_spans() if s.name == TEAM_NEW_PATH]
    assert len(spans) == 1
    attribute_keys: Final = tuple(spans[0].attributes)
    assert any(key.startswith("request.") for key in attribute_keys) is exports_payload
    assert any(key.startswith("response.") for key in attribute_keys) is exports_payload


@pytest.mark.parametrize(
    ("wrapper_name", "exports_request_body"),
    [("management_endpoint_wrapper", True), ("management_endpoint_span_wrapper", False)],
)
def test_span_wrapper_omits_request_body_on_failure(
    server_span_factory, otel_with_exporter, monkeypatch, wrapper_name, exports_request_body
):
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.management_helpers import utils as mgmt_utils

    otel, exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", otel, raising=False)

    server_span: Final = server_span_factory(TEAM_NEW_PATH)
    wrap: Final = getattr(mgmt_utils, wrapper_name)

    @wrap
    async def fake_fn(http_request: Request, user_api_key_dict: UserAPIKeyAuth) -> dict[str, object]:
        raise make_fastapi_http_exception(400, "team_alias taken")

    with pytest.raises(Exception, match="team_alias taken"):
        asyncio.run(
            fake_fn(
                http_request=_request("POST", TEAM_NEW_PATH, b'{"team_alias": "t"}'),
                user_api_key_dict=_real_user_api_key_dict(server_span),
            )
        )

    spans: Final = [s for s in exporter.get_finished_spans() if s.name == TEAM_NEW_PATH]
    assert len(spans) == 1
    assert any(key.startswith("request.") for key in spans[0].attributes) is exports_request_body


@pytest.mark.parametrize(
    ("raised", "status_code"),
    [
        (RuntimeError("router settings lookup failed"), 500),
        (httpx.ConnectError("All connection attempts failed"), 503),
        (make_fastapi_http_exception(400, "Invalid scope parameter"), 400),
    ],
)
def test_span_wrapper_failure_keeps_the_answered_status(otel_with_exporter, monkeypatch, raised, status_code):
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.management_helpers import utils as mgmt_utils

    otel, exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", otel, raising=False)

    server_span: Final = _server_span(otel, MODEL_LIST_PATH, headers={})
    request: Final = _request("GET", MODEL_LIST_PATH)
    request.state.parent_otel_span = server_span

    @mgmt_utils.management_endpoint_span_wrapper
    async def failing_model_list(request: Request, user_api_key_dict: UserAPIKeyAuth) -> dict[str, object]:
        raise raised

    async def call_and_answer() -> None:
        try:
            await failing_model_list(request=request, user_api_key_dict=_real_user_api_key_dict(server_span))
        except Exception as exc:
            await _answer_like_the_proxy(request, exc)

    asyncio.run(call_and_answer())

    assert_server_span_attrs(
        exporter,
        expected_status=status_code,
        expected_url_path=MODEL_LIST_PATH,
        where=f"model list {type(raised).__name__}",
    )
    exported: Final = get_server_span(exporter)
    assert exported is not None
    assert exported.attributes.get("error.code") == str(status_code)


def _server_span(otel, path: str, headers: dict[str, str]):
    server_span: Final = otel.create_litellm_proxy_request_started_span(start_time=datetime.now(), headers=headers)
    otel.set_proxy_request_route_attributes(server_span, url_path=path, http_route=path)
    return server_span


def _assert_ended_in_inbound_trace(exporter, path: str, handler_name: str) -> None:
    assert_server_span_attrs(exporter, expected_status=200, expected_url_path=path, where=f"{path} success")
    exported: Final = get_server_span(exporter)
    assert exported is not None
    assert exported.context.trace_id == int(INBOUND_TRACEPARENT.split("-")[1], 16)

    endpoint_spans: Final = [s for s in exporter.get_finished_spans() if s.name == handler_name]
    assert len(endpoint_spans) == 1
    assert endpoint_spans[0].parent is not None
    assert endpoint_spans[0].parent.span_id == exported.context.span_id
    assert not any(key.startswith("response.") for key in endpoint_spans[0].attributes)


def _install_proxy_state(monkeypatch, router: Router | None) -> None:
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)


def test_model_list_ends_server_span_in_inbound_trace(otel_with_exporter, monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    otel, exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", otel, raising=False)
    _install_proxy_state(monkeypatch, None)
    server_span: Final = _server_span(otel, MODEL_LIST_PATH, headers={"traceparent": INBOUND_TRACEPARENT})

    result: Final = asyncio.run(proxy_server.model_list(user_api_key_dict=_real_user_api_key_dict(server_span)))

    assert result == {"data": [], "object": "list"}
    _assert_ended_in_inbound_trace(exporter, MODEL_LIST_PATH, "model_list")


def test_model_retrieve_ends_server_span_in_inbound_trace(otel_with_exporter, monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    otel, exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", otel, raising=False)
    _install_proxy_state(
        monkeypatch,
        Router(model_list=[{"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-x"}}]),
    )
    server_span: Final = _server_span(otel, MODEL_RETRIEVE_PATH, headers={"traceparent": INBOUND_TRACEPARENT})

    result: Final = asyncio.run(
        proxy_server.model_info(model_id="gpt-4o", user_api_key_dict=_real_user_api_key_dict(server_span))
    )

    assert result["id"] == "gpt-4o"
    _assert_ended_in_inbound_trace(exporter, MODEL_RETRIEVE_PATH, "model_info")
