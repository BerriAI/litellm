import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.proxy_track_cost_callback import (
    _ProxyDBLogger,
    _get_budget_reservation_from_metadata,
    _update_database_and_spend_counters,
)


@pytest.mark.asyncio
async def test_async_post_call_failure_hook():
    # Setup
    logger = _ProxyDBLogger()

    # Mock user_api_key_dict
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        key_alias="test_alias",
        user_email="test@example.com",
        user_id="test_user_id",
        team_id="test_team_id",
        org_id="test_org_id",
        team_alias="test_team_alias",
        end_user_id="test_end_user_id",
    )

    # Mock request data
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"original_key": "original_value"},
        "proxy_server_request": {"request_id": "test_request_id"},
    }

    # Mock exception
    original_exception = Exception("Test exception")

    # Mock update_database function
    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database:
        # Call the method
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )

        # Assertions
        mock_update_database.assert_called_once()

        # Check the arguments passed to update_database
        call_args = mock_update_database.call_args[1]
        assert call_args["token"] == "test_api_key"
        assert call_args["response_cost"] == 0.0
        assert call_args["user_id"] == "test_user_id"
        assert call_args["end_user_id"] == "test_end_user_id"
        assert call_args["team_id"] == "test_team_id"
        assert call_args["org_id"] == "test_org_id"
        assert call_args["completion_response"] == original_exception

        # Check that metadata was properly updated
        assert "litellm_params" in call_args["kwargs"]
        assert call_args["kwargs"]["litellm_params"]["proxy_server_request"] == {
            "request_id": "test_request_id"
        }
        metadata = call_args["kwargs"]["litellm_params"]["metadata"]
        assert metadata["user_api_key"] == "test_api_key"
        assert metadata["status"] == "failure"
        assert "error_information" in metadata
        assert metadata["original_key"] == "original_value"


@pytest.mark.asyncio
async def test_async_post_call_failure_hook_non_llm_route():
    # Setup
    logger = _ProxyDBLogger()

    # Mock user_api_key_dict with a non-LLM route
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        key_alias="test_alias",
        user_email="test@example.com",
        user_id="test_user_id",
        team_id="test_team_id",
        org_id="test_org_id",
        team_alias="test_team_alias",
        end_user_id="test_end_user_id",
        request_route="/custom/route",  # Non-LLM route
    )

    # Mock request data
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"original_key": "original_value"},
        "proxy_server_request": {"request_id": "test_request_id"},
    }

    # Mock exception
    original_exception = Exception("Test exception")

    # Mock update_database function
    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database:
        # Call the method
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )

        # Assert that update_database was NOT called for non-LLM routes
        mock_update_database.assert_not_called()


@pytest.mark.asyncio
async def test_async_post_call_failure_hook_releases_budget_reservation_before_route_skip():
    logger = _ProxyDBLogger()
    budget_reservation = {"reserved_cost": 0.5, "entries": []}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        request_route="/custom/route",
        budget_reservation=budget_reservation,
    )

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.release_budget_reservation",
            new_callable=AsyncMock,
        ) as mock_release_budget_reservation,
        patch(
            "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
            new_callable=AsyncMock,
        ) as mock_update_database,
    ):
        await logger.async_post_call_failure_hook(
            request_data={},
            original_exception=Exception("Test exception"),
            user_api_key_dict=user_api_key_dict,
        )

        assert mock_release_budget_reservation.await_count == 1
        assert (
            mock_release_budget_reservation.await_args.kwargs["budget_reservation"]
            is user_api_key_dict.budget_reservation
        )
        mock_update_database.assert_not_called()


@pytest.mark.asyncio
async def test_should_continue_failure_tracking_when_budget_release_fails():
    logger = _ProxyDBLogger()
    budget_reservation = {"reserved_cost": 0.5, "entries": []}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        team_id="test_team_id",
        request_route="/chat/completions",
        budget_reservation=budget_reservation,
    )

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.release_budget_reservation",
            new_callable=AsyncMock,
            side_effect=RuntimeError("redis unavailable"),
        ) as mock_release_budget_reservation,
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback._invalidate_budget_reservation_counters",
            new_callable=AsyncMock,
        ) as mock_invalidate_budget_reservation_counters,
        patch(
            "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
            new_callable=AsyncMock,
        ) as mock_update_database,
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback.verbose_proxy_logger.exception",
        ) as mock_log_exception,
    ):
        await logger.async_post_call_failure_hook(
            request_data={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            original_exception=Exception("provider failed"),
            user_api_key_dict=user_api_key_dict,
        )

        assert mock_release_budget_reservation.await_count == 1
        assert (
            mock_release_budget_reservation.await_args.kwargs["budget_reservation"]
            is user_api_key_dict.budget_reservation
        )
        assert mock_invalidate_budget_reservation_counters.await_count == 1
        assert (
            mock_invalidate_budget_reservation_counters.await_args.kwargs[
                "budget_reservation"
            ]
            is user_api_key_dict.budget_reservation
        )
        assert user_api_key_dict.budget_reservation["finalized"] is True
        mock_log_exception.assert_called_once()
        mock_update_database.assert_called_once()


@pytest.mark.asyncio
async def test_track_cost_callback_releases_budget_reservation_when_spend_tracking_skips():
    logger = _ProxyDBLogger()
    budget_reservation = {"reserved_cost": 0.5, "entries": []}
    user_api_key_auth = UserAPIKeyAuth(budget_reservation=budget_reservation)

    kwargs = {
        "model": "gpt-4",
        "litellm_params": {
            "metadata": {
                "user_api_key_auth": user_api_key_auth,
            },
        },
        "standard_logging_object": {
            "response_cost": 0.1,
            "request_tags": None,
        },
        "stream": False,
    }

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.release_budget_reservation",
        new_callable=AsyncMock,
    ) as mock_release_budget_reservation:
        await logger._PROXY_track_cost_callback(
            kwargs=kwargs,
            completion_response=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        mock_release_budget_reservation.assert_awaited_once_with(
            budget_reservation=budget_reservation,
        )


@pytest.mark.asyncio
async def test_track_cost_callback_releases_budget_reservation_when_response_cost_missing():
    logger = _ProxyDBLogger()
    budget_reservation = {"reserved_cost": 0.5, "entries": []}
    user_api_key_auth = UserAPIKeyAuth(budget_reservation=budget_reservation)

    kwargs = {
        "model": "gpt-4",
        "call_type": "acompletion",
        "litellm_params": {
            "metadata": {
                "user_api_key_auth": user_api_key_auth,
            },
        },
        "standard_logging_object": {
            "response_cost": None,
            "response_cost_failure_debug_info": "missing custom price",
            "request_tags": None,
        },
        "stream": False,
    }

    with (
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj",
        ) as mock_proxy_logging,
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.release_budget_reservation",
            new_callable=AsyncMock,
        ) as mock_release_budget_reservation,
    ):
        mock_proxy_logging.failed_tracking_alert = AsyncMock()

        await logger._PROXY_track_cost_callback(
            kwargs=kwargs,
            completion_response=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        mock_release_budget_reservation.assert_awaited_once_with(
            budget_reservation=budget_reservation,
        )


def test_get_budget_reservation_from_metadata_handles_dict_auth_object():
    budget_reservation = {
        "reserved_cost": 0.5,
        "entries": [{"counter_key": "spend:key:test_api_key"}],
    }

    assert (
        _get_budget_reservation_from_metadata(
            metadata={"user_api_key_auth": dict(UserAPIKeyAuth())}
        )
        is None
    )
    assert (
        _get_budget_reservation_from_metadata(
            metadata={
                "user_api_key_auth": UserAPIKeyAuth(
                    budget_reservation=budget_reservation
                )
            }
        )
        == budget_reservation
    )
    assert (
        _get_budget_reservation_from_metadata(
            metadata={
                "user_api_key_auth": dict(
                    UserAPIKeyAuth(budget_reservation=budget_reservation)
                )
            }
        )
        == budget_reservation
    )
    assert (
        _get_budget_reservation_from_metadata(
            metadata={"user_api_key_budget_reservation": budget_reservation}
        )
        is budget_reservation
    )


@pytest.mark.asyncio
async def test_update_database_and_spend_counters_releases_reservation_when_db_update_fails():
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.db_spend_update_writer.update_database = AsyncMock(
        side_effect=Exception("db unavailable")
    )
    increment_spend_counters = AsyncMock()
    budget_reservation = {"reserved_cost": 0.5, "entries": []}

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.release_budget_reservation",
        new_callable=AsyncMock,
    ) as mock_release_budget_reservation:
        with pytest.raises(Exception, match="db unavailable"):
            await _update_database_and_spend_counters(
                proxy_logging_obj=proxy_logging_obj,
                increment_spend_counters=increment_spend_counters,
                user_api_key="test_api_key",
                user_id="test_user_id",
                end_user_id=None,
                team_id="test_team_id",
                org_id="test_org_id",
                kwargs={},
                completion_response=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                response_cost=0.2,
                budget_reservation=budget_reservation,
            )

        mock_release_budget_reservation.assert_awaited_once_with(
            budget_reservation=budget_reservation,
        )

    increment_spend_counters.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_database_and_spend_counters_preserves_db_exception_when_release_fails():
    proxy_logging_obj = MagicMock()
    db_exception = RuntimeError("db unavailable")
    proxy_logging_obj.db_spend_update_writer.update_database = AsyncMock(
        side_effect=db_exception
    )
    increment_spend_counters = AsyncMock()
    budget_reservation = {"reserved_cost": 0.5, "entries": []}

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.release_budget_reservation",
            new_callable=AsyncMock,
            side_effect=RuntimeError("release unavailable"),
        ) as mock_release_budget_reservation,
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback.verbose_proxy_logger.exception",
        ) as mock_log_exception,
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback._invalidate_budget_reservation_counters",
            new_callable=AsyncMock,
            side_effect=RuntimeError("invalidate unavailable"),
        ) as mock_invalidate_budget_reservation_counters,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            await _update_database_and_spend_counters(
                proxy_logging_obj=proxy_logging_obj,
                increment_spend_counters=increment_spend_counters,
                user_api_key="test_api_key",
                user_id="test_user_id",
                end_user_id=None,
                team_id="test_team_id",
                org_id="test_org_id",
                kwargs={},
                completion_response=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                response_cost=0.2,
                budget_reservation=budget_reservation,
            )

        assert exc_info.value is db_exception
        mock_release_budget_reservation.assert_awaited_once_with(
            budget_reservation=budget_reservation,
        )
        mock_invalidate_budget_reservation_counters.assert_awaited_once_with(
            budget_reservation=budget_reservation,
        )
        assert mock_log_exception.call_count == 2
        mock_log_exception.assert_any_call(
            "Failed to release budget reservation after database update failed"
        )
        mock_log_exception.assert_any_call(
            "Failed to invalidate budget reservation counters after release failed"
        )

    increment_spend_counters.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_database_and_spend_counters_updates_counters_after_db_update():
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.db_spend_update_writer.update_database = AsyncMock()
    increment_spend_counters = AsyncMock()
    budget_reservation = {"reserved_cost": 0.5, "entries": []}

    await _update_database_and_spend_counters(
        proxy_logging_obj=proxy_logging_obj,
        increment_spend_counters=increment_spend_counters,
        user_api_key="test_api_key",
        user_id="test_user_id",
        end_user_id="test_end_user_id",
        team_id="test_team_id",
        org_id="test_org_id",
        kwargs={},
        completion_response=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
        response_cost=0.2,
        budget_reservation=budget_reservation,
        request_tags=["tag-a"],
    )

    proxy_logging_obj.db_spend_update_writer.update_database.assert_awaited_once()
    increment_spend_counters.assert_awaited_once_with(
        token="test_api_key",
        team_id="test_team_id",
        user_id="test_user_id",
        response_cost=0.2,
        org_id="test_org_id",
        budget_reservation=budget_reservation,
        end_user_id="test_end_user_id",
        tags=["tag-a"],
    )


@pytest.mark.asyncio
async def test_update_database_and_spend_counters_invalidates_reservation_when_counter_update_fails():
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.db_spend_update_writer.update_database = AsyncMock()
    increment_spend_counters = AsyncMock(side_effect=Exception("counter unavailable"))
    budget_reservation = {
        "reserved_cost": 0.5,
        "entries": [{"counter_key": "spend:key:test_api_key"}],
    }

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.invalidate_budget_reservation_counters",
        new_callable=AsyncMock,
    ) as mock_invalidate_budget_reservation_counters:
        with pytest.raises(Exception, match="counter unavailable"):
            await _update_database_and_spend_counters(
                proxy_logging_obj=proxy_logging_obj,
                increment_spend_counters=increment_spend_counters,
                user_api_key="test_api_key",
                user_id="test_user_id",
                end_user_id=None,
                team_id="test_team_id",
                org_id="test_org_id",
                kwargs={},
                completion_response=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                response_cost=0.2,
                budget_reservation=budget_reservation,
            )

        mock_invalidate_budget_reservation_counters.assert_awaited_once_with(
            budget_reservation=budget_reservation,
        )
        assert budget_reservation["finalized"] is True

    proxy_logging_obj.db_spend_update_writer.update_database.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_database_and_spend_counters_preserves_counter_exception_when_invalidation_fails():
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.db_spend_update_writer.update_database = AsyncMock()
    counter_exception = RuntimeError("counter unavailable")
    increment_spend_counters = AsyncMock(side_effect=counter_exception)
    budget_reservation = {
        "reserved_cost": 0.5,
        "entries": [{"counter_key": "spend:key:test_api_key"}],
    }

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.invalidate_budget_reservation_counters",
            new_callable=AsyncMock,
            side_effect=RuntimeError("invalidate unavailable"),
        ) as mock_invalidate_budget_reservation_counters,
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback.verbose_proxy_logger.exception",
        ) as mock_log_exception,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            await _update_database_and_spend_counters(
                proxy_logging_obj=proxy_logging_obj,
                increment_spend_counters=increment_spend_counters,
                user_api_key="test_api_key",
                user_id="test_user_id",
                end_user_id=None,
                team_id="test_team_id",
                org_id="test_org_id",
                kwargs={},
                completion_response=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                response_cost=0.2,
                budget_reservation=budget_reservation,
            )

        assert exc_info.value is counter_exception
        mock_invalidate_budget_reservation_counters.assert_awaited_once_with(
            budget_reservation=budget_reservation,
        )
        mock_log_exception.assert_called_once_with(
            "Failed to invalidate budget reservation counters after spend counter update failed"
        )
        assert budget_reservation["finalized"] is True

    proxy_logging_obj.db_spend_update_writer.update_database.assert_awaited_once()


@pytest.mark.asyncio
async def test_track_cost_callback_skips_when_no_standard_logging_object():
    """
    Reproduces the bug where _PROXY_track_cost_callback raises
    'Cost tracking failed for model=None' when kwargs has no
    standard_logging_object (e.g. call_type=afile_delete).

    File operations have no model and no standard_logging_object.
    The callback should skip gracefully instead of raising.
    """
    logger = _ProxyDBLogger()

    kwargs = {
        "call_type": "afile_delete",
        "model": None,
        "litellm_call_id": "test-call-id",
        "litellm_params": {},
        "stream": False,
    }

    with patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
    ) as mock_proxy_logging:
        mock_proxy_logging.failed_tracking_alert = AsyncMock()
        mock_proxy_logging.db_spend_update_writer = MagicMock()
        mock_proxy_logging.db_spend_update_writer.update_database = AsyncMock()

        await logger._PROXY_track_cost_callback(
            kwargs=kwargs,
            completion_response=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        # update_database should NOT be called — nothing to track
        mock_proxy_logging.db_spend_update_writer.update_database.assert_not_called()

        # failed_tracking_alert should NOT be called — this is not an error
        mock_proxy_logging.failed_tracking_alert.assert_not_called()


@pytest.mark.asyncio
async def test_async_post_call_failure_hook_propagates_trace_id_from_logging_obj():
    """
    When an LLM call fails, the proxy calls post_call_failure_hook with
    request_data that doesn't contain standard_logging_object. But the
    litellm_logging_obj (set by function_setup) is in request_data and
    holds the standard_logging_object with the correct trace_id.

    The failure hook should propagate this so the DB spend log's session_id
    matches the Langfuse trace_id.
    """
    logger = _ProxyDBLogger()

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        team_id="test_team_id",
    )

    # Simulate a litellm_logging_obj with model_call_details containing
    # the standard_logging_object (as set by _failure_handler_helper_fn)
    mock_logging_obj = MagicMock()
    mock_logging_obj.litellm_trace_id = "trace-id-from-logging-obj"
    mock_logging_obj.model_call_details = {
        "standard_logging_object": {
            "trace_id": "trace-id-from-logging-obj",
            "error_str": "InternalServerError",
        }
    }

    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {},
        "litellm_params": {},
        "litellm_logging_obj": mock_logging_obj,
        # Note: no "standard_logging_object" and no "litellm_trace_id"
    }

    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database:
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=Exception("Provider error"),
            user_api_key_dict=user_api_key_dict,
        )

        mock_update_database.assert_called_once()
        call_kwargs = mock_update_database.call_args[1]["kwargs"]

        # standard_logging_object should have been propagated from logging obj
        assert call_kwargs.get("standard_logging_object") is not None
        assert (
            call_kwargs["standard_logging_object"]["trace_id"]
            == "trace-id-from-logging-obj"
        )
        # litellm_trace_id should also be propagated as a fallback
        assert call_kwargs.get("litellm_trace_id") == "trace-id-from-logging-obj"


@pytest.mark.asyncio
async def test_enrich_failure_metadata_with_team_alias():
    """
    When team_id is set but team_alias is missing (and key_alias is present),
    _enrich_failure_metadata_with_key_info should look up the team from cache
    and populate user_api_key_team_alias.
    """
    mock_team_obj = MagicMock()
    mock_team_obj.team_alias = "my-team-alias"

    with patch(
        "litellm.proxy.hooks.proxy_track_cost_callback.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_team_obj,
    ):
        metadata = {
            "user_api_key": "hashed_key",
            "user_api_key_alias": "my-key-alias",  # already set
            "user_api_key_team_id": "test_team_id",
            "user_api_key_team_alias": None,
        }
        result = await _ProxyDBLogger._enrich_failure_metadata_with_key_info(metadata)
        assert result["user_api_key_team_alias"] == "my-team-alias"


@pytest.mark.asyncio
async def test_enrich_failure_metadata_with_full_key_lookup():
    """
    When all key fields are null (auth error 401 scenario), _enrich_failure_metadata_with_key_info
    should look up the key object from cache/DB and populate alias, user_id, team_id,
    then look up the team to get team_alias.
    """
    mock_key_obj = MagicMock()
    mock_key_obj.key_alias = "fetched-key-alias"
    mock_key_obj.user_id = "fetched-user-id"
    mock_key_obj.team_id = "fetched-team-id"
    mock_key_obj.org_id = "fetched-org-id"

    mock_team_obj = MagicMock()
    mock_team_obj.team_alias = "fetched-team-alias"

    with (
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback.get_key_object",
            new_callable=AsyncMock,
            return_value=mock_key_obj,
        ),
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback.get_team_object",
            new_callable=AsyncMock,
            return_value=mock_team_obj,
        ),
    ):
        metadata = {
            "user_api_key": "hashed_key",
            "user_api_key_alias": None,  # all null - simulates auth error path
            "user_api_key_user_id": None,
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
            "user_api_key_org_id": None,
        }
        result = await _ProxyDBLogger._enrich_failure_metadata_with_key_info(metadata)
        assert result["user_api_key_alias"] == "fetched-key-alias"
        assert result["user_api_key_user_id"] == "fetched-user-id"
        assert result["user_api_key_team_id"] == "fetched-team-id"
        assert result["user_api_key_org_id"] == "fetched-org-id"
        assert result["user_api_key_team_alias"] == "fetched-team-alias"


@pytest.mark.asyncio
async def test_enrich_failure_metadata_skips_when_team_alias_present():
    """
    When team_alias is already populated, _enrich_failure_metadata_with_key_info
    should not perform a team cache lookup.
    """
    with (
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback.get_key_object",
            new_callable=AsyncMock,
        ) as mock_get_key,
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback.get_team_object",
            new_callable=AsyncMock,
        ) as mock_get_team,
    ):
        metadata = {
            "user_api_key": "hashed_key",
            "user_api_key_alias": "existing-alias",
            "user_api_key_team_id": "test_team_id",
            "user_api_key_team_alias": "already-set",
        }
        result = await _ProxyDBLogger._enrich_failure_metadata_with_key_info(metadata)
        assert result["user_api_key_team_alias"] == "already-set"
        mock_get_key.assert_not_called()
        mock_get_team.assert_not_called()


@pytest.mark.asyncio
async def test_enrich_failure_metadata_skips_when_no_api_key():
    """
    When api_key hash is absent, _enrich_failure_metadata_with_key_info should
    not perform any lookups.
    """
    with patch(
        "litellm.proxy.hooks.proxy_track_cost_callback.get_key_object",
        new_callable=AsyncMock,
    ) as mock_get_key:
        metadata = {
            "user_api_key": None,
            "user_api_key_alias": None,
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
        }
        await _ProxyDBLogger._enrich_failure_metadata_with_key_info(metadata)
        mock_get_key.assert_not_called()


@pytest.mark.asyncio
async def test_async_post_call_failure_hook_enriches_auth_error_metadata():
    """
    Simulates a 401 ProxyException (e.g. can_key_call_model). In this case
    UserAPIKeyAuth is created with only api_key set. The failure hook should
    look up the key and team from cache/DB to populate all missing fields.
    """
    logger = _ProxyDBLogger()

    # This is what auth_exception_handler creates for 401 errors
    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed_key",
        # key_alias, user_id, team_id, team_alias are all None
    )

    request_data = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {},
        "litellm_params": {},
    }

    mock_key_obj = MagicMock()
    mock_key_obj.key_alias = "my-key-alias"
    mock_key_obj.user_id = "my-user-id"
    mock_key_obj.team_id = "my-team-id"
    mock_key_obj.org_id = None

    mock_team_obj = MagicMock()
    mock_team_obj.team_alias = "my-team-alias"

    with (
        patch(
            "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
            new_callable=AsyncMock,
        ) as mock_update_database,
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback.get_key_object",
            new_callable=AsyncMock,
            return_value=mock_key_obj,
        ),
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback.get_team_object",
            new_callable=AsyncMock,
            return_value=mock_team_obj,
        ),
    ):
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=Exception("401 - model not allowed"),
            user_api_key_dict=user_api_key_dict,
        )

        mock_update_database.assert_called_once()
        call_args = mock_update_database.call_args[1]
        metadata = call_args["kwargs"]["litellm_params"]["metadata"]
        assert metadata["user_api_key_alias"] == "my-key-alias"
        assert metadata["user_api_key_user_id"] == "my-user-id"
        assert metadata["user_api_key_team_id"] == "my-team-id"
        assert metadata["user_api_key_team_alias"] == "my-team-alias"


@pytest.mark.asyncio
async def test_async_post_call_failure_hook_enriches_missing_team_alias():
    """
    When user_api_key_dict has a team_id but no team_alias, async_post_call_failure_hook
    should look up the team from cache and populate user_api_key_team_alias in the
    spend log metadata written to the DB.
    """
    logger = _ProxyDBLogger()

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        key_alias="test_alias",
        user_id="test_user_id",
        team_id="test_team_id",
        team_alias=None,  # Missing - simulates regular key auth where SQL view omits team_alias
    )

    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {},
        "litellm_params": {},
    }

    mock_team_obj = MagicMock()
    mock_team_obj.team_alias = "enriched-team-alias"

    with (
        patch(
            "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
            new_callable=AsyncMock,
        ) as mock_update_database,
        patch(
            "litellm.proxy.hooks.proxy_track_cost_callback.get_team_object",
            new_callable=AsyncMock,
            return_value=mock_team_obj,
        ),
    ):
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=Exception("Provider rate limit"),
            user_api_key_dict=user_api_key_dict,
        )

        mock_update_database.assert_called_once()
        call_args = mock_update_database.call_args[1]
        metadata = call_args["kwargs"]["litellm_params"]["metadata"]
        assert metadata["user_api_key_team_alias"] == "enriched-team-alias"
        assert metadata["user_api_key_team_id"] == "test_team_id"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_value", [None, ""])
async def test_track_cost_callback_skips_for_falsy_model_and_no_slo(model_value):
    """
    Same bug as above but model can also be empty string (e.g. health check callbacks).
    The guard should catch all falsy model values when sl_object is missing.
    """
    logger = _ProxyDBLogger()

    kwargs = {
        "call_type": "acompletion",
        "model": model_value,
        "litellm_params": {},
        "stream": False,
    }

    with patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
    ) as mock_proxy_logging:
        mock_proxy_logging.failed_tracking_alert = AsyncMock()
        mock_proxy_logging.db_spend_update_writer = MagicMock()
        mock_proxy_logging.db_spend_update_writer.update_database = AsyncMock()

        await logger._PROXY_track_cost_callback(
            kwargs=kwargs,
            completion_response=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        mock_proxy_logging.failed_tracking_alert.assert_not_called()


@pytest.mark.asyncio
async def test_async_post_call_failure_hook_uses_actual_start_time():
    """
    Verify that failed requests record the actual request start time
    instead of datetime.now(), so the spend log shows the real duration.

    Previously both start_time and end_time were set to datetime.now()
    at failure-logging time, resulting in duration=0 for all failures.
    """
    from datetime import timedelta

    logger = _ProxyDBLogger()

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        team_id="test_team_id",
        org_id="test_org_id",
        end_user_id="test_end_user_id",
    )

    # Simulate a request that started 60 seconds ago
    simulated_start = datetime.now() - timedelta(seconds=60)

    mock_logging_obj = MagicMock()
    mock_logging_obj.start_time = simulated_start
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.litellm_trace_id = None

    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {},
        "proxy_server_request": {},
        "litellm_logging_obj": mock_logging_obj,
    }

    original_exception = Exception("Timeout error")

    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database:
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )

        mock_update_database.assert_called_once()
        call_args = mock_update_database.call_args[1]

        # start_time should be the simulated start, not datetime.now()
        assert call_args["start_time"] == simulated_start

        # end_time should be close to now (within a few seconds)
        time_diff = (datetime.now() - call_args["end_time"]).total_seconds()
        assert time_diff < 5, f"end_time should be close to now, was {time_diff}s ago"

        # Duration should be approximately 60 seconds, not 0
        duration = (call_args["end_time"] - call_args["start_time"]).total_seconds()
        assert duration >= 55, f"Duration should be ~60s, got {duration}s"


async def _invoke_failure_hook_with_raised_exception():
    """Run the failure hook with an exception that has a real ``__traceback__``.

    Returns the metadata dict that was forwarded to ``update_database`` so the
    caller can assert on its ``error_information`` payload.
    """
    logger = _ProxyDBLogger()
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="u",
        team_id="t",
    )
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {},
        "proxy_server_request": {},
    }

    try:
        raise RuntimeError("boom-with-traceback")
    except RuntimeError as exc:
        original_exception = exc

    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database:
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
        )
        call_args = mock_update_database.call_args[1]
        return call_args["kwargs"]["litellm_params"]["metadata"]


@pytest.mark.asyncio
async def test_failure_hook_keeps_error_information_traceback_by_default(monkeypatch):
    """Without the opt-in env var, the SpendLogs row carries the full traceback."""
    monkeypatch.delenv("LITELLM_SUPPRESS_SPEND_LOG_TRACEBACKS", raising=False)

    metadata = await _invoke_failure_hook_with_raised_exception()

    error_information = metadata["error_information"]
    assert error_information["error_class"] == "RuntimeError"
    assert error_information["error_message"] == "boom-with-traceback"
    assert error_information["traceback"], "expected a non-empty traceback by default"


@pytest.mark.asyncio
async def test_failure_hook_drops_error_information_traceback_when_env_set(
    monkeypatch,
):
    """With the opt-in env var, the traceback key is omitted from the
    SpendLogs row entirely so the per-row Metadata pane in the UI (which
    renders ``error_information`` as a JSON blob) doesn't show a noisy empty
    ``"traceback": ""`` line. The other fields (error_class / error_message /
    error_code) are preserved."""
    import logging

    from litellm._logging import verbose_proxy_logger

    monkeypatch.setenv("LITELLM_SUPPRESS_SPEND_LOG_TRACEBACKS", "true")
    original_level = verbose_proxy_logger.level
    verbose_proxy_logger.setLevel(logging.INFO)
    try:
        metadata = await _invoke_failure_hook_with_raised_exception()
    finally:
        verbose_proxy_logger.setLevel(original_level)

    error_information = metadata["error_information"]
    assert "traceback" not in error_information
    assert error_information["error_class"] == "RuntimeError"
    assert error_information["error_message"] == "boom-with-traceback"
