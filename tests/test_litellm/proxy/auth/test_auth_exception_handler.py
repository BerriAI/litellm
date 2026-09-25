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
    EngineRequestError,
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
from litellm.constants import INVALID_VIRTUAL_KEY_ERROR_MARKER
from litellm.exceptions import BudgetExceededError
from litellm.proxy._types import (
    ModelAccessDeniedProxyException,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_exception_handler import UserAPIKeyAuthExceptionHandler, _as_proxy_exception
from litellm.proxy.auth.model_access_denied import ModelAccessDeniedHTTPException


class _EngineHttp500:
    """The response half of an EngineRequestError: the query engine answered a request with HTTP 500."""

    status = 500


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
        pytest.param(BinaryNotFoundError("query engine binary not found"), id="BinaryNotFoundError"),
        pytest.param(MismatchedVersionsError(expected="1", got="2"), id="MismatchedVersionsError"),
        pytest.param(EngineRequestError(_EngineHttp500(), "query engine crashed"), id="EngineRequestError"),
        pytest.param(PrismaError(), id="bare_PrismaError"),
    ],
)
async def test_handle_authentication_error_permanent_fault_503_is_not_worded_as_transient(prisma_error):
    """The 503 for a fault that never heals must not say the database is
    "temporarily unreachable" and ask the caller to retry. The status is right
    (the service is at fault) but that wording sends the operator to wait out an
    outage that is not one, so the message has to say retrying will not help and
    name the engine fault."""
    handler = UserAPIKeyAuthExceptionHandler()

    with patch(  # test-quality-ok: the handler reads general_settings off the proxy module, no injection seam
        "litellm.proxy.proxy_server.general_settings", {"allow_requests_on_db_unavailable": False}
    ):
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(prisma_error, MagicMock(), {}, "/test", None, "test-key")

    assert exc_info.value.code == str(status.HTTP_503_SERVICE_UNAVAILABLE)
    assert exc_info.value.type == ProxyErrorTypes.no_db_connection
    assert "temporarily unreachable" not in exc_info.value.message
    assert "retry shortly" not in exc_info.value.message.lower()
    assert "will not clear by retrying" in exc_info.value.message
    assert type(prisma_error).__name__ in exc_info.value.message


@pytest.mark.asyncio
async def test_handle_authentication_error_transport_error_raised_over_a_permanent_fault_names_the_fault():
    """A reconnect attempt that fails because the engine binary is missing surfaces as a transport
    error with the BinaryNotFoundError as __context__. The response must describe the binary, which is
    what keeps the database down, rather than promise the connection will come back."""
    try:
        raise BinaryNotFoundError("query engine binary not found")
    except BinaryNotFoundError:
        try:
            raise httpx.ConnectError("All connection attempts failed")
        except httpx.ConnectError as surfaced:
            transport_over_fault = surfaced
    handler = UserAPIKeyAuthExceptionHandler()

    with patch(  # test-quality-ok: the handler reads general_settings off the proxy module, no injection seam
        "litellm.proxy.proxy_server.general_settings", {"allow_requests_on_db_unavailable": False}
    ):
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(transport_over_fault, MagicMock(), {}, "/test", None, "k")

    assert exc_info.value.code == str(status.HTTP_503_SERVICE_UNAVAILABLE)
    assert "temporarily unreachable" not in exc_info.value.message
    assert "BinaryNotFoundError" in exc_info.value.message
    assert "will not clear by retrying" in exc_info.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "db_error",
    [
        pytest.param(httpx.ConnectError("All connection attempts failed"), id="ConnectError"),
        pytest.param(EngineConnectionError(), id="EngineConnectionError"),
        pytest.param(PrismaError("can't reach database server"), id="P1001_text"),
    ],
)
async def test_handle_authentication_error_transient_outage_503_keeps_retry_wording(db_error):
    """A genuine outage is expected to come back, so its 503 keeps telling the
    caller the database is temporarily unreachable and to retry."""
    handler = UserAPIKeyAuthExceptionHandler()

    with patch(  # test-quality-ok: the handler reads general_settings off the proxy module, no injection seam
        "litellm.proxy.proxy_server.general_settings", {"allow_requests_on_db_unavailable": False}
    ):
        with pytest.raises(ProxyException) as exc_info:
            await handler._handle_authentication_error(db_error, MagicMock(), {}, "/test", None, "test-key")

    assert exc_info.value.code == str(status.HTTP_503_SERVICE_UNAVAILABLE)
    assert exc_info.value.message == (
        "Service Unavailable, the authentication database is temporarily unreachable. Please retry shortly."
    )


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
    assert int(exc_info.value.code) == status.HTTP_422_UNPROCESSABLE_CONTENT


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
async def test_dynamic_route_normalized_on_auth_failure():
    handler = UserAPIKeyAuthExceptionHandler()

    with (
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
        ) as mock_post_call_failure_hook,
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.proxy_server.general_settings", {}
        ),
        pytest.raises(ProxyException),
    ):
        await handler._handle_authentication_error(
            HTTPException(status_code=401, detail="Authentication Error, Invalid proxy server token passed"),
            MagicMock(),
            {},
            "/v1/responses/resp_attacker_controlled_id",
            None,
            "sk-doesnotexist",
        )

    hook_kwargs = mock_post_call_failure_hook.call_args.kwargs
    assert hook_kwargs["route"] == "/v1/responses/resp_attacker_controlled_id"
    assert hook_kwargs["user_api_key_dict"].request_route == "/v1/responses/{response_id}"


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
@pytest.mark.parametrize(
    "resolved_identity, expected_fragment, absent_fragment",
    [
        pytest.param(
            UserAPIKeyAuth(
                token="hashed-token",
                key_alias="skip-laptop-key",
                user_id="skip-user",
                user_email="skip@example.com",
                team_id="team-123",
                team_alias="research-team",
            ),
            "Key Identity: key_alias=skip-laptop-key user_id=skip-user user_email=skip@example.com "
            "team_id=team-123 team_alias=research-team",
            None,
            id="expired_key_owner_named_in_log",
        ),
        pytest.param(
            UserAPIKeyAuth(token="hashed-token", user_id="skip-user"),
            "Key Identity: user_id=skip-user",
            "key_alias=",
            id="unset_fields_omitted",
        ),
        pytest.param(
            UserAPIKeyAuth(token="hashed-token", team_alias="ops\nRequester IP Address:10.0.0.1"),
            "Key Identity: team_alias=ops\\nRequester IP Address:10.0.0.1",
            "\nRequester IP Address:10.0.0.1",
            id="control_chars_in_alias_cannot_forge_log_lines",
        ),
        pytest.param(None, None, "Key Identity", id="unknown_key_has_no_identity_line"),
    ],
)
async def test_expired_key_error_log_names_the_key_owner(resolved_identity, expected_fragment, absent_fragment, caplog):
    """An expired key rejection is logged with the key alias, user and team auth already
    resolved, so an operator can trace the caller from the log line alone."""
    handler = UserAPIKeyAuthExceptionHandler()
    expired_key_error = ProxyException(
        message="Authentication Error - Expired Key.",
        type=ProxyErrorTypes.expired_key,
        param="sk-...",
        code=status.HTTP_401_UNAUTHORIZED,
    )

    with (
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("litellm.proxy.auth.auth_exception_handler.seed_request_identity"),
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        verbose_proxy_logger.propagate = True
        try:
            with caplog.at_level("ERROR", logger="LiteLLM Proxy"), pytest.raises(ProxyException):
                await handler._handle_authentication_error(
                    expired_key_error,
                    MagicMock(),
                    {"model": "gpt-4o"},
                    "/v1/chat/completions",
                    None,
                    "sk-raw-key",
                    resolved_identity=resolved_identity,
                )
        finally:
            verbose_proxy_logger.propagate = False

    records = [r for r in caplog.records if "user_api_key_auth(): Exception occured" in r.getMessage()]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    logged = records[0].getMessage()
    assert "Expired Key" in logged and "Requester IP Address:" in logged, logged
    if expected_fragment is not None:
        assert expected_fragment in logged, logged
    if absent_fragment is not None:
        assert absent_fragment not in logged, logged


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
            id="422_budget_exceeded",
        ),
    ],
)
async def test_auth_failure_logs_requester_ip_address(
    auth_error: Exception,
    general_settings: dict[str, bool],
    request_kwargs: dict[str, dict[str, str]],
    expected_ip: str,
) -> None:
    """401s and budget 422s are rejected before `add_litellm_data_to_request` stamps
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
    "request_data, metadata_key, route",
    [
        pytest.param({"model": "gpt-4o"}, "metadata", "/v1/chat/completions", id="chat_metadata"),
        pytest.param({"litellm_metadata": {}}, "litellm_metadata", "/v1/responses", id="responses_litellm_metadata"),
    ],
)
async def test_auth_failure_logs_user_agent(request_data: dict[str, object], metadata_key: str, route: str) -> None:
    """Auth gate rejections never reach `add_litellm_data_to_request`, which is what
    stamps `user_agent`, so the failure spend log and prometheus `user_agent` label
    had nothing to identify an abusive client by."""
    with (
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.auth.auth_exception_handler.seed_request_identity"
        ),
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_hook,
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
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
                _http_request(headers={"user-agent": "abusive-client/9.9"}),
                request_data,
                route,
                None,
                "sk-bad-key",
            )

    logged_metadata = mock_hook.call_args[1]["request_data"][metadata_key]
    assert logged_metadata["user_agent"] == "abusive-client/9.9"
    assert logged_metadata["requester_ip_address"] == "10.1.2.3"


@pytest.mark.asyncio
async def test_auth_failure_without_headers_scope_still_raises_original_error() -> None:
    """A request scope with no `headers` entry must surface the auth error itself, not a
    `KeyError` from reading the User-Agent."""
    with (
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.auth.auth_exception_handler.seed_request_identity"
        ),
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_failure_hook",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_hook,
        patch(  # test-quality-ok: handler reads proxy_server globals at call time
            "litellm.proxy.proxy_server.general_settings",
            {"allow_requests_on_db_unavailable": False},
        ),
    ):
        with pytest.raises(ProxyException) as exc_info:
            await UserAPIKeyAuthExceptionHandler._handle_authentication_error(
                ProxyException(
                    message="Invalid API key",
                    type=ProxyErrorTypes.auth_error,
                    param=None,
                    code=status.HTTP_401_UNAUTHORIZED,
                ),
                Request(scope={"type": "http"}),
                {"model": "gpt-4o"},
                "/v1/chat/completions",
                None,
                "sk-bad-key",
            )

    assert str(exc_info.value.code) == str(status.HTTP_401_UNAUTHORIZED)
    assert "user_agent" not in mock_hook.call_args[1]["request_data"].get("metadata", {})


def _marked_malformed_key_error() -> HTTPException:
    """Build the malformed-key 401 as its raise site does: marker stamped on it."""
    error = HTTPException(status_code=401, detail="LiteLLM Virtual Key expected. Received=test")
    setattr(error, INVALID_VIRTUAL_KEY_ERROR_MARKER, True)
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth_error,expect_traceback,expect_level",
    [
        pytest.param(
            ProxyException(
                message="Authentication Error", type=ProxyErrorTypes.auth_error, param=None, code=401
            ),
            False,
            "ERROR",
            id="expected_401_no_traceback",
        ),
        pytest.param(ValueError("unexpected internal error"), True, "ERROR", id="unexpected_error_keeps_traceback"),
        pytest.param(
            _marked_malformed_key_error(),
            False,
            "WARNING",
            id="malformed_virtual_key_warning_no_traceback",
        ),
        pytest.param(
            HTTPException(status_code=401, detail="LiteLLM Virtual Key expected. Received=test"),
            False,
            "ERROR",
            id="phrase_without_marker_stays_loud",
        ),
    ],
)
async def test_handle_authentication_error_traceback_only_for_unexpected_errors(auth_error, expect_traceback, expect_level, caplog):
    """Regression for LIT-6043: expected 4xx auth rejections must not format a
    traceback via logger.exception; malformed virtual keys log at WARNING."""
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
            except (ProxyException, ValueError, HTTPException) as caught:
                with caplog.at_level(expect_level, logger="LiteLLM Proxy"), pytest.raises((ProxyException, HTTPException)):
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
    assert records[0].levelname == expect_level
    expected_logger_name = "LiteLLM Proxy.stdout" if expect_level == "WARNING" else "LiteLLM Proxy"
    assert records[0].name == expected_logger_name


_DENIED_CLIENT_MESSAGE = (
    "The requested model 'gpt-5.6' is not available for this API key, or the model name is invalid. "
    "Check the models available to you and try again."
)


def _denied_proxy_exception() -> ModelAccessDeniedProxyException:
    return ModelAccessDeniedProxyException(
        message=_DENIED_CLIENT_MESSAGE,
        internal_message="key not allowed to access model. This key can only access models=['internal-models']. "
        "Tried to access gpt-5.6\r\nWARNING forged log line",
        type=ProxyErrorTypes.key_model_access_denied,
        param="model",
        code=status.HTTP_403_FORBIDDEN,
    )


def _denied_jwt_exception() -> ModelAccessDeniedHTTPException:
    return ModelAccessDeniedHTTPException(
        internal_message="Role=engineer not allowed to call model=gpt-5.6\r\nWARNING forged log line. "
        "Allowed models=['internal-models']",
        status_code=status.HTTP_403_FORBIDDEN,
        detail=_DENIED_CLIENT_MESSAGE,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_denial",
    [
        pytest.param(_denied_proxy_exception, id="proxy_exception"),
        pytest.param(_denied_jwt_exception, id="jwt_http_exception"),
    ],
)
async def test_handle_authentication_error_keeps_internal_message_on_model_access_denial(make_denial, caplog):
    handler = UserAPIKeyAuthExceptionHandler()
    denial = make_denial()

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
        caplog.at_level("WARNING", logger="LiteLLM Proxy"),
        pytest.raises(ModelAccessDeniedProxyException) as exc_info,
    ):
        await handler._handle_authentication_error(denial, MagicMock(), {}, "/v1/chat/completions", None, "sk-bad-key")

    assert exc_info.value.code == str(status.HTTP_403_FORBIDDEN)
    assert "internal-models" not in str(exc_info.value.message)
    assert exc_info.value.internal_message == denial.internal_message
    assert [r for r in caplog.records if r.levelname == "WARNING" and "internal-models" in r.getMessage()] == []


def test_as_proxy_exception_keeps_jwt_scope_denial_message_shape():
    detail = {"error": _DENIED_CLIENT_MESSAGE}
    denial = ModelAccessDeniedHTTPException(
        internal_message="model=gpt-5.6 not allowed. Allowed_models=['internal-models']",
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail,
    )
    plain = _as_proxy_exception(HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail))

    converted = _as_proxy_exception(denial)

    assert converted.to_dict() == plain.to_dict()
    assert converted.internal_message == denial.internal_message
