import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, status
from prisma import errors as prisma_errors
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

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_exception_handler import UserAPIKeyAuthExceptionHandler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prisma_error",
    [
        # Specific connectivity subclasses.
        HTTPClientClosedError(),
        ClientNotConnectedError(),
        # Bare / generic PrismaError defaults to connectivity — we can't
        # tell what it is, so err on the safe side for genuine outages.
        PrismaError(),
    ],
)
async def test_handle_authentication_error_db_unavailable_connectivity(prisma_error):
    """Transport-level / connectivity failures (and generic PrismaError)
    trigger the HA fallback."""
    handler = UserAPIKeyAuthExceptionHandler()

    mock_request = MagicMock()
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"allow_requests_on_db_unavailable": True},
    ):
        result = await handler._handle_authentication_error(
            prisma_error,
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
async def test_handle_authentication_error_budget_exceeded():
    handler = UserAPIKeyAuthExceptionHandler()

    # Mock request and other dependencies
    mock_request = MagicMock()
    mock_request_data = {}
    mock_route = "/test"
    mock_span = None
    mock_api_key = "test-key"

    # Test with budget exceeded error
    with pytest.raises(ProxyException) as exc_info:
        from litellm.exceptions import BudgetExceededError

        budget_error = BudgetExceededError(
            message="Budget exceeded", current_cost=100, max_budget=100
        )
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
