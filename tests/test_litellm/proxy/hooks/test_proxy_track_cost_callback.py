import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.types.utils import StandardLoggingPayload


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
        print("call_args", json.dumps(call_args, indent=4, default=str))
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

    with patch(
        "litellm.proxy.hooks.proxy_track_cost_callback.get_key_object",
        new_callable=AsyncMock,
        return_value=mock_key_obj,
    ), patch(
        "litellm.proxy.hooks.proxy_track_cost_callback.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_team_obj,
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
    with patch(
        "litellm.proxy.hooks.proxy_track_cost_callback.get_key_object",
        new_callable=AsyncMock,
    ) as mock_get_key, patch(
        "litellm.proxy.hooks.proxy_track_cost_callback.get_team_object",
        new_callable=AsyncMock,
    ) as mock_get_team:
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
        result = await _ProxyDBLogger._enrich_failure_metadata_with_key_info(metadata)
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

    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database, patch(
        "litellm.proxy.hooks.proxy_track_cost_callback.get_key_object",
        new_callable=AsyncMock,
        return_value=mock_key_obj,
    ), patch(
        "litellm.proxy.hooks.proxy_track_cost_callback.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_team_obj,
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

    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database, patch(
        "litellm.proxy.hooks.proxy_track_cost_callback.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_team_obj,
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
