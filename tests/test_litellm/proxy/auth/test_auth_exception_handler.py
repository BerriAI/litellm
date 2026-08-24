import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Request, status
from prisma import errors as prisma_errors
from prisma.engine.errors import (
    BinaryNotFoundError,
    EngineConnectionError,
    MismatchedVersionsError,
)
from prisma.errors import (
    ClientNotConnectedError,
    DataError,
    ForeignKeyViolationError,
    HTTPClientClosedError,
    MissingRequiredValueError,
    PrismaError,
    RawQueryError,
    RecordNotFoundError,
    TableNotFoundError,
    UniqueViolationError,
)


from litellm._logging import verbose_proxy_logger
from litellm.exceptions import BudgetExceededError
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_exception_handler import UserAPIKeyAuthExceptionHandler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "db_error",
    [
        pytest.param(httpx.ConnectError("All connection attempts failed"), id="ConnectError"),
        pytest.param(httpx.ReadError("read failed"), id="ReadError"),
        pytest.param(httpx.ReadTimeout("timed out"), id="ReadTimeout"),
        pytest.param(EngineConnectionError(), id="EngineConnectionError"),
    ],
)
async def test_handle_authentication_error_db_unavailable_connectivity(db_error):
    """A database that is temporarily unreachable triggers the HA fallback.

    These are the failures a real outage actually produces: the query engine is
    a local HTTP server, so an unreachable database surfaces as a transport
    error against it."""
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": True},
    ):
        result = await handler._handle_authentication_error(
            db_error,
            mock_request,
            {},
            "/test",
            None,
            "test-key",
        )
        assert result.key_name == "failed-to-connect-to-db"
        assert result.token == "failed-to-connect-to-db"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        pytest.param(BinaryNotFoundError("query engine binary not found"), id="BinaryNotFoundError"),
        pytest.param(MismatchedVersionsError(expected="1", got="2"), id="MismatchedVersionsError"),
        pytest.param(HTTPClientClosedError(), id="HTTPClientClosedError"),
        pytest.param(ClientNotConnectedError(), id="ClientNotConnectedError"),
        pytest.param(PrismaError(), id="bare_PrismaError"),
    ],
)
async def test_handle_authentication_error_permanent_fault_gets_no_fallback_identity(
    prisma_error,
):
    """A fault that cannot resolve on its own must not mint a fallback identity,
    even with ``allow_requests_on_db_unavailable`` enabled.

    That setting trades verification for availability on the assumption the
    database returns. When it never will, the trade buys nothing and the proxy
    would keep admitting callers it cannot verify for as long as it runs, so the
    fault has to reach the caller instead.

    It must still reach them as a service failure. Denying the fallback is not
    licence to report a database fault as a rejected credential, which would
    send an operator hunting a key problem that does not exist."""
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": True},
    ):
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(
                prisma_error,
                mock_request,
                {},
                "/test",
                None,
                "test-key",
            )

    assert exc_info.value.type == ProxyErrorTypes.no_db_connection
    assert exc_info.value.code == str(status.HTTP_503_SERVICE_UNAVAILABLE)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        DataError(data={"user_facing_error": {"meta": {"table": "test_table"}}}),
        UniqueViolationError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        ForeignKeyViolationError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        MissingRequiredValueError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        RawQueryError(data={"user_facing_error": {"meta": {"table": "test_table"}}}),
        TableNotFoundError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
        RecordNotFoundError(
            data={"user_facing_error": {"meta": {"table": "test_table"}}}
        ),
    ],
)
async def test_handle_authentication_error_data_layer_errors_do_not_fall_back(
    prisma_error,
):
    """Known data-layer PrismaError subclasses (UniqueViolation,
    RecordNotFound, etc.) mean the DB IS reachable — they must propagate
    instead of triggering the HA fallback, which would grant the
    restricted INTERNAL_USER token to a request that should have
    returned 401."""
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": True},
    ):
        with pytest.raises(ProxyException):
            await handler._handle_authentication_error(
                prisma_error,
                mock_request,
                {},
                "/test",
                None,
                "test-key",
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "db_error",
    [
        ConnectionError("connection refused"),
        TimeoutError("timed out"),
        asyncio.TimeoutError(),
        OSError("network is unreachable"),
        HTTPClientClosedError(),
        PrismaError("can't reach database server"),
        RawQueryError(
            data={
                "user_facing_error": {
                    "message": "cached plan must not change result type",
                    "meta": {"table": "t"},
                }
            }
        ),
    ],
)
async def test_handle_authentication_error_db_infra_error_returns_503(db_error):
    """Regression for the outage where valid keys got 401 for 4 hours: an
    infrastructure-level DB failure during auth must surface as 503 (the DB
    could not confirm the key), never as 401 ("Invalid API key")."""
    handler = UserAPIKeyAuthExceptionHandler()

    with (
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.proxy.auth.auth_exception_handler.seed_request_identity",
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(
                db_error,
                MagicMock(),
                {},
                "/v1/chat/completions",
                None,
                "sk-valid-but-db-down",
            )

    assert int(exc_info.value.code) == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.type == ProxyErrorTypes.no_db_connection
    assert "Invalid API key" not in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_handle_authentication_error_prisma_engine_teardown_returns_503():
    """Regression for the first-request-of-an-outage edge case: at the instant
    the DB socket drops, the prisma query engine returns a malformed error
    payload and prisma-client-py crashes with a bare
    ``AttributeError: 'NoneType' object has no attribute 'get'`` before it can
    raise P1001. That AttributeError reached auth and fell through to 401. It
    must surface as 503 like every other infra failure during the outage."""
    from prisma.engine import utils as prisma_engine_utils

    malformed_payload = [
        {
            "error": "Can't reach database server",
            "user_facing_error": {
                "error_code": "P1001",
                "message": "Can't reach database server at `localhost`:`5503`",
                "meta": None,
            },
        }
    ]
    try:
        prisma_engine_utils.handle_response_errors(None, malformed_payload)
        raise AssertionError("expected prisma to raise AttributeError")
    except AttributeError as e:
        teardown_error = e

    handler = UserAPIKeyAuthExceptionHandler()

    with (
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.proxy.auth.auth_exception_handler.seed_request_identity",
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(
                teardown_error,
                MagicMock(),
                {},
                "/v1/chat/completions",
                None,
                "sk-valid-but-db-down",
            )

    assert int(exc_info.value.code) == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.type == ProxyErrorTypes.no_db_connection
    assert "Invalid API key" not in str(exc_info.value.message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth_error",
    [
        # DB returned no row -> get_key_object raises this exact 401.
        ProxyException(
            message="Authentication Error, Invalid proxy server token passed.",
            type=ProxyErrorTypes.token_not_found_in_db,
            param="key",
            code=status.HTTP_401_UNAUTHORIZED,
        ),
        # A bare auth failure raised as a plain Exception (e.g. master-key-only
        # route) must keep returning 401, not get reclassified as 503.
        Exception("Invalid proxy server token passed"),
    ],
)
async def test_handle_authentication_error_genuine_auth_failure_stays_401(auth_error):
    """Guard against the 503 conversion being too broad: a genuine auth
    failure (missing key / wrong key) must still be 401."""
    handler = UserAPIKeyAuthExceptionHandler()

    with (
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.proxy.auth.auth_exception_handler.seed_request_identity",
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(
                auth_error,
                MagicMock(),
                {},
                "/v1/chat/completions",
                None,
                "sk-bad-key",
            )

    assert int(exc_info.value.code) == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_handle_authentication_error_budget_exceeded():
    handler = UserAPIKeyAuthExceptionHandler()

    # Mock request and other dependencies
    mock_request = MagicMock()
    mock_request_data = {}
    mock_route = "/test"
    mock_span = None
    mock_api_key = "test-key"

    # Test with budget exceeded error
    from litellm.exceptions import BudgetExceededError

    budget_error = BudgetExceededError(
        message="Budget exceeded", current_cost=100, max_budget=100
    )

    with pytest.raises(ProxyException) as exc_info:
        await handler._handle_authentication_error(
            budget_error,
            mock_request,
            mock_request_data,
            mock_route,
            mock_span,
            mock_api_key,
        )

    assert exc_info.value.type == ProxyErrorTypes.budget_exceeded
    assert int(exc_info.value.code) == status.HTTP_429_TOO_MANY_REQUESTS


@pytest.mark.asyncio
async def test_route_passed_to_post_call_failure_hook():
    """
    This route is used by proxy track_cost_callback's async_post_call_failure_hook to check if the route is an LLM route
    """
    handler = UserAPIKeyAuthExceptionHandler()

    # Mock request and other dependencies
    mock_request = MagicMock()
    mock_request_data = {}
    test_route = "/custom/route"
    mock_span = None
    mock_api_key = "test-key"

    # Mock proxy_logging_obj.post_call_failure_hook
    with patch(
        "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
        new_callable=AsyncMock,
    ) as mock_post_call_failure_hook:
        # Test with DB connection error
        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ):
            try:
                await handler._handle_authentication_error(
                    PrismaError(),
                    mock_request,
                    mock_request_data,
                    test_route,
                    mock_span,
                    mock_api_key,
                )
            except Exception as e:
                pass
            asyncio.sleep(1)
            # Verify post_call_failure_hook was called with the correct route
            mock_post_call_failure_hook.assert_called_once()
            call_args = mock_post_call_failure_hook.call_args[1]
            assert call_args["user_api_key_dict"].request_route == test_route


@pytest.mark.asyncio
async def test_resolved_identity_exported_on_auth_failure():
    """Regression: when auth fails AFTER the key/team/user identity is resolved
    (e.g. an expired key), that identity must still reach the failure logging /
    span instead of being dropped for a blank UserAPIKeyAuth. Before the fix the
    handler built a fresh empty object, so the failed trace showed no team alias,
    team id, or metadata."""
    handler = UserAPIKeyAuthExceptionHandler()

    resolved_identity = UserAPIKeyAuth(
        token="hashed-token",
        team_id="team-123",
        team_alias="acme-team",
        user_id="user-456",
        metadata={"foo": "bar"},
        team_metadata={"baz": "qux"},
    )

    expired_key_error = ProxyException(
        message="Authentication Error - Expired Key.",
        type=ProxyErrorTypes.expired_key,
        param="sk-...",
        code=status.HTTP_401_UNAUTHORIZED,
    )

    seeded = {}

    def _capture_seed(user_api_key_dict, model=None):
        seeded["dict"] = user_api_key_dict
        seeded["model"] = model

    with (
        patch(
            "litellm.proxy.auth.auth_exception_handler.seed_request_identity",
            side_effect=_capture_seed,
        ) as mock_seed,
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
        ) as mock_hook,
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        with pytest.raises(ProxyException):
            await handler._handle_authentication_error(
                expired_key_error,
                MagicMock(),
                {"model": "gpt-4o"},
                "/v1/chat/completions",
                None,
                "sk-raw-key",
                resolved_identity=resolved_identity,
            )

    # The identity that auth already resolved is what gets logged on failure.
    logged = mock_hook.call_args[1]["user_api_key_dict"]
    assert logged.team_id == "team-123"
    assert logged.team_alias == "acme-team"
    assert logged.user_id == "user-456"
    assert logged.metadata == {"foo": "bar"}
    assert logged.team_metadata == {"baz": "qux"}
    assert logged.request_route == "/v1/chat/completions"

    # And it is stamped onto the span eagerly, before the request is rejected.
    mock_seed.assert_called_once()
    assert seeded["dict"] is logged
    assert seeded["dict"].team_alias == "acme-team"
    assert seeded["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_auth_failure_without_resolved_identity_still_logs():
    """When auth fails before any identity is resolved (e.g. an unknown key),
    the handler must still log a usable object carrying the raw api key and
    route, not crash on the missing identity."""
    handler = UserAPIKeyAuthExceptionHandler()

    with (
        patch(
            "litellm.proxy.auth.auth_exception_handler.seed_request_identity",
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
        ) as mock_hook,
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        with pytest.raises(ProxyException):
            await handler._handle_authentication_error(
                ProxyException(
                    message="Invalid API key",
                    type=ProxyErrorTypes.auth_error,
                    param=None,
                    code=status.HTTP_401_UNAUTHORIZED,
                ),
                MagicMock(),
                {},
                "/v1/chat/completions",
                None,
                "sk-unknown",
            )

    logged = mock_hook.call_args[1]["user_api_key_dict"]
    # Raw key must NOT land on the object — it would be promoted into telemetry
    # as litellm.api_key.hash and leak a real sk-... to anyone reading the trace.
    assert logged.api_key != "sk-unknown"
    assert logged.api_key == UserAPIKeyAuth(api_key="sk-unknown").api_key
    assert logged.request_route == "/v1/chat/completions"


def _http_request(client_host: str | None = "10.1.2.3", headers: dict[str, str] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "server": ("testserver", 80),
            "client": (client_host, 51234) if client_host is not None else None,
            "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth_error, general_settings, request_kwargs, expected_ip",
    [
        pytest.param(
            ProxyException(
                message="Invalid API key",
                type=ProxyErrorTypes.auth_error,
                param=None,
                code=status.HTTP_401_UNAUTHORIZED,
            ),
            {"allow_requests_on_db_unavailable": False},
            {},
            "10.1.2.3",
            id="401_socket_peer",
        ),
        pytest.param(
            ProxyException(
                message="Invalid API key",
                type=ProxyErrorTypes.auth_error,
                param=None,
                code=status.HTTP_401_UNAUTHORIZED,
            ),
            {"allow_requests_on_db_unavailable": False, "use_x_forwarded_for": True},
            {"headers": {"x-forwarded-for": "203.0.113.9"}},
            "203.0.113.9",
            id="401_x_forwarded_for",
        ),
        pytest.param(
            BudgetExceededError(message="Budget exceeded", current_cost=100, max_budget=100),
            {"allow_requests_on_db_unavailable": False},
            {},
            "10.1.2.3",
            id="429_budget_exceeded",
        ),
    ],
)
async def test_auth_failure_logs_requester_ip_address(
    auth_error: Exception,
    general_settings: dict[str, bool],
    request_kwargs: dict[str, dict[str, str]],
    expected_ip: str,
) -> None:
    """401s and budget 429s are rejected before `add_litellm_data_to_request` stamps
    the caller IP, so without this the failure logs (spend logs, prometheus client_ip)
    had no IP, and a 401 rarely carries a key or user identity either."""
    with (
        patch("litellm.proxy.auth.auth_exception_handler.seed_request_identity"),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_hook,
        patch("litellm.proxy.proxy_server.general_settings", general_settings),
    ):
        with pytest.raises(ProxyException):
            await UserAPIKeyAuthExceptionHandler._handle_authentication_error(
                auth_error,
                _http_request(**request_kwargs),
                {"model": "gpt-4o"},
                "/v1/chat/completions",
                None,
                "sk-bad-key",
            )

    logged_request_data = mock_hook.call_args[1]["request_data"]
    assert logged_request_data["metadata"]["requester_ip_address"] == expected_ip


@pytest.mark.asyncio
async def test_auth_failure_keeps_existing_requester_ip_address():
    """An IP already recorded upstream (e.g. a trusted-proxy resolved value) wins over
    the socket peer."""
    with (
        patch("litellm.proxy.auth.auth_exception_handler.seed_request_identity"),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_hook,
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        with pytest.raises(ProxyException):
            await UserAPIKeyAuthExceptionHandler._handle_authentication_error(
                ProxyException(
                    message="Invalid API key",
                    type=ProxyErrorTypes.auth_error,
                    param=None,
                    code=status.HTTP_401_UNAUTHORIZED,
                ),
                _http_request(),
                {"metadata": {"requester_ip_address": "198.51.100.4"}},
                "/v1/chat/completions",
                None,
                "sk-bad-key",
            )

    logged_request_data = mock_hook.call_args[1]["request_data"]
    assert logged_request_data["metadata"]["requester_ip_address"] == "198.51.100.4"


@pytest.mark.asyncio
async def test_auth_failure_ip_uses_litellm_metadata_when_present():
    """Routes that keep proxy metadata under `litellm_metadata` (e.g. /responses) must
    get the IP there, since that is the dict the logging layer reads for them."""
    with (
        patch("litellm.proxy.auth.auth_exception_handler.seed_request_identity"),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_hook,
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        with pytest.raises(ProxyException):
            await UserAPIKeyAuthExceptionHandler._handle_authentication_error(
                ProxyException(
                    message="Invalid API key",
                    type=ProxyErrorTypes.auth_error,
                    param=None,
                    code=status.HTTP_401_UNAUTHORIZED,
                ),
                _http_request(),
                {"litellm_metadata": {}, "metadata": {"user_supplied": "keep-me"}},
                "/v1/responses",
                None,
                "sk-bad-key",
            )

    logged_request_data = mock_hook.call_args[1]["request_data"]
    assert logged_request_data["litellm_metadata"]["requester_ip_address"] == "10.1.2.3"
    assert logged_request_data["metadata"] == {"user_supplied": "keep-me"}


@pytest.mark.asyncio
async def test_auth_failure_ip_stamp_does_not_mutate_callers_request_data():
    """The handler must not rewrite the caller's dict; the IP is for the failure log only."""
    request_data = {"model": "gpt-4o"}

    with (
        patch("litellm.proxy.auth.auth_exception_handler.seed_request_identity"),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        with pytest.raises(ProxyException):
            await UserAPIKeyAuthExceptionHandler._handle_authentication_error(
                ProxyException(
                    message="Invalid API key",
                    type=ProxyErrorTypes.auth_error,
                    param=None,
                    code=status.HTTP_401_UNAUTHORIZED,
                ),
                _http_request(),
                request_data,
                "/v1/chat/completions",
                None,
                "sk-bad-key",
            )

    assert request_data == {"model": "gpt-4o"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth_error,expect_traceback",
    [
        pytest.param(
            ProxyException(
                message="Authentication Error", type=ProxyErrorTypes.auth_error, param=None, code=401
            ),
            False,
            id="expected_401_no_traceback",
        ),
        pytest.param(ValueError("unexpected internal error"), True, id="unexpected_error_keeps_traceback"),
    ],
)
async def test_handle_authentication_error_traceback_only_for_unexpected_errors(auth_error, expect_traceback, caplog):
    """Regression for LIT-6043: expected 4xx auth rejections must not format a
    traceback via logger.exception; unexpected errors must keep it."""
    handler = UserAPIKeyAuthExceptionHandler()

    with (
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.auth.auth_exception_handler.seed_request_identity",
        ),
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        verbose_proxy_logger.propagate = True
        try:
            try:
                raise auth_error
            except (ProxyException, ValueError) as caught:
                with caplog.at_level("ERROR", logger="LiteLLM Proxy"), pytest.raises(ProxyException):
                    await handler._handle_authentication_error(
                        caught,
                        MagicMock(),
                        {},
                        "/v1/chat/completions",
                        None,
                        "sk-bad-key",
                    )
        finally:
            verbose_proxy_logger.propagate = False

    records = [r for r in caplog.records if "user_api_key_auth(): Exception occured" in r.getMessage()]
    assert len(records) == 1
    assert (records[0].exc_info is not None) is expect_traceback
