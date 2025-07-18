import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()
import os

from litellm.proxy._types import LiteLLM_BudgetTableFull

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob

# Note: In our "fake" items we use dicts with fields that our fake reset functions modify.
# In a real-world scenario, these would be instances of LiteLLM_VerificationToken, LiteLLM_UserTable, etc.


@pytest.mark.asyncio
async def test_reset_budget_keys_partial_failure():
    """
    Test that if one key fails to reset, the failure for that key does not block processing of the other keys.
    We simulate two keys where the first fails and the second succeeds.
    """
    # Arrange
    key1 = {
        "id": "key1",
        "spend": 10.0,
        "budget_duration": 60,
    }  # Will trigger simulated failure
    key2 = {"id": "key2", "spend": 15.0, "budget_duration": 60}  # Should be updated
    key3 = {"id": "key3", "spend": 20.0, "budget_duration": 60}  # Should be updated
    key4 = {"id": "key4", "spend": 25.0, "budget_duration": 60}  # Should be updated
    key5 = {"id": "key5", "spend": 30.0, "budget_duration": 60}  # Should be updated
    key6 = {"id": "key6", "spend": 35.0, "budget_duration": 60}  # Should be updated

    prisma_client = MagicMock()
    prisma_client.get_data = AsyncMock(
        return_value=[key1, key2, key3, key4, key5, key6]
    )
    prisma_client.update_data = AsyncMock()

    # Using a dummy logging object with async hooks mocked out.
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    now = datetime.utcnow()

    async def fake_reset_key(key, current_time):
        if key["id"] == "key1":
            # Simulate a failure on key1 (for example, this might be due to an invariant check)
            raise Exception("Simulated failure for key1")
        else:
            # Simulate successful reset modification
            key["spend"] = 0.0
            # Compute a new reset time based on the budget duration
            key["budget_reset_at"] = (
                current_time + timedelta(seconds=key["budget_duration"])
            ).isoformat()
            return key

    with patch.object(
        ResetBudgetJob, "_reset_budget_for_key", side_effect=fake_reset_key
    ) as mock_reset_key:
        # Call the method; even though one key fails, the loop should process both
        await job.reset_budget_for_litellm_keys()
        # Allow any created tasks (logging hooks) to schedule
        await asyncio.sleep(0.1)

    # Assert that the helper was called for 6 keys
    assert mock_reset_key.call_count == 6

    # Assert that update_data was called once with a list containing all 6 keys
    prisma_client.update_data.assert_awaited_once()
    update_call = prisma_client.update_data.call_args
    assert update_call.kwargs.get("table_name") == "key"
    updated_keys = update_call.kwargs.get("data_list", [])
    assert len(updated_keys) == 5
    assert updated_keys[0]["id"] == "key2"
    assert updated_keys[1]["id"] == "key3"
    assert updated_keys[2]["id"] == "key4"
    assert updated_keys[3]["id"] == "key5"
    assert updated_keys[4]["id"] == "key6"

    # Verify that the failure logging hook was scheduled (due to the failure for key1)
    failure_hook_calls = (
        proxy_logging_obj.service_logging_obj.async_service_failure_hook.call_args_list
    )
    # There should be one failure hook call for keys (with call_type "reset_budget_keys")
    assert any(
        call.kwargs.get("call_type") == "reset_budget_keys"
        for call in failure_hook_calls
    )


@pytest.mark.asyncio
async def test_reset_budget_users_partial_failure():
    """
    Test that if one user fails to reset, the reset loop still processes the other users.
    We simulate two users where the first fails and the second is updated.
    """
    user1 = {
        "id": "user1",
        "spend": 20.0,
        "budget_duration": 120,
    }  # Will trigger simulated failure
    user2 = {"id": "user2", "spend": 25.0, "budget_duration": 120}  # Should be updated
    user3 = {"id": "user3", "spend": 30.0, "budget_duration": 120}  # Should be updated
    user4 = {"id": "user4", "spend": 35.0, "budget_duration": 120}  # Should be updated
    user5 = {"id": "user5", "spend": 40.0, "budget_duration": 120}  # Should be updated
    user6 = {"id": "user6", "spend": 45.0, "budget_duration": 120}  # Should be updated

    prisma_client = MagicMock()
    prisma_client.get_data = AsyncMock(
        return_value=[user1, user2, user3, user4, user5, user6]
    )
    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_user(user, current_time):
        if user["id"] == "user1":
            raise Exception("Simulated failure for user1")
        else:
            user["spend"] = 0.0
            user["budget_reset_at"] = (
                current_time + timedelta(seconds=user["budget_duration"])
            ).isoformat()
            return user

    with patch.object(
        ResetBudgetJob, "_reset_budget_for_user", side_effect=fake_reset_user
    ) as mock_reset_user:
        await job.reset_budget_for_litellm_users()
        await asyncio.sleep(0.1)

    assert mock_reset_user.call_count == 6
    prisma_client.update_data.assert_awaited_once()
    update_call = prisma_client.update_data.call_args
    assert update_call.kwargs.get("table_name") == "user"
    updated_users = update_call.kwargs.get("data_list", [])
    assert len(updated_users) == 5
    assert updated_users[0]["id"] == "user2"
    assert updated_users[1]["id"] == "user3"
    assert updated_users[2]["id"] == "user4"
    assert updated_users[3]["id"] == "user5"
    assert updated_users[4]["id"] == "user6"

    failure_hook_calls = (
        proxy_logging_obj.service_logging_obj.async_service_failure_hook.call_args_list
    )
    assert any(
        call.kwargs.get("call_type") == "reset_budget_users"
        for call in failure_hook_calls
    )


@pytest.mark.asyncio
async def test_reset_budget_endusers_partial_failure():
    """
    Test that if one enduser fails to reset, the reset loop still processes the other endusers.
    We simulate six endsers where the first fails and the others are updated.
    """
    user1 = {
        "user_id": "user1",
        "spend": 20.0,
        "budget_id": "budget1",
    }  # Will trigger simulated failure
    user2 = {
        "user_id": "user2",
        "spend": 25.0,
        "budget_id": "budget1",
    }  # Should be updated
    user3 = {
        "user_id": "user3",
        "spend": 30.0,
        "budget_id": "budget1",
    }  # Should be updated
    user4 = {
        "user_id": "user4",
        "spend": 35.0,
        "budget_id": "budget1",
    }  # Should be updated
    user5 = {
        "user_id": "user5",
        "spend": 40.0,
        "budget_id": "budget1",
    }  # Should be updated
    user6 = {
        "user_id": "user6",
        "spend": 45.0,
        "budget_id": "budget1",
    }  # Should be updated

    budget1 = LiteLLM_BudgetTableFull(
        **{
            "budget_id": "budget1",
            "max_budget": 65.0,
            "budget_duration": "2d",
            "created_at": datetime.now(timezone.utc) - timedelta(days=3),
        }
    )

    prisma_client = MagicMock()

    async def get_data_mock(table_name, *args, **kwargs):
        if table_name == "budget":
            return [budget1]
        elif table_name == "enduser":
            return [user1, user2, user3, user4, user5, user6]
        return []

    prisma_client.get_data = AsyncMock()
    prisma_client.get_data.side_effect = get_data_mock

    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_enduser(enduser):
        if enduser["user_id"] == "user1":
            raise Exception("Simulated failure for user1")
        enduser["spend"] = 0.0
        return enduser

    async def fake_reset_team_members(budgets_to_reset):
        return 1

    with patch.object(
        ResetBudgetJob,
        "_reset_budget_for_enduser",
        side_effect=fake_reset_enduser,
    ) as mock_reset_enduser, patch.object(
        ResetBudgetJob,
        "reset_budget_for_litellm_team_members",
        side_effect=fake_reset_team_members,
    ) as mock_reset_team_members:
        await job.reset_budget_for_litellm_budget_table()
        await asyncio.sleep(0.1)

    assert mock_reset_enduser.call_count == 6
    assert prisma_client.update_data.await_count == 2
    update_call = prisma_client.update_data.call_args
    assert update_call.kwargs.get("table_name") == "enduser"
    updated_users = update_call.kwargs.get("data_list", [])
    assert len(updated_users) == 5
    assert updated_users[0]["user_id"] == "user2"
    assert updated_users[1]["user_id"] == "user3"
    assert updated_users[2]["user_id"] == "user4"
    assert updated_users[3]["user_id"] == "user5"
    assert updated_users[4]["user_id"] == "user6"

    failure_hook_calls = (
        proxy_logging_obj.service_logging_obj.async_service_failure_hook.call_args_list
    )
    assert any(
        call.kwargs.get("call_type") == "reset_budget_endusers"
        for call in failure_hook_calls
    )


@pytest.mark.asyncio
async def test_reset_budget_teams_partial_failure():
    """
    Test that if one team fails to reset, the loop processes both teams and only updates the ones that succeeded.
    We simulate two teams where the first fails and the second is updated.
    """
    team1 = {
        "id": "team1",
        "spend": 30.0,
        "budget_duration": 180,
    }  # Will trigger simulated failure
    team2 = {"id": "team2", "spend": 35.0, "budget_duration": 180}  # Should be updated

    prisma_client = MagicMock()
    prisma_client.get_data = AsyncMock(return_value=[team1, team2])
    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_team(team, current_time):
        if team["id"] == "team1":
            raise Exception("Simulated failure for team1")
        else:
            team["spend"] = 0.0
            team["budget_reset_at"] = (
                current_time + timedelta(seconds=team["budget_duration"])
            ).isoformat()
            return team

    with patch.object(
        ResetBudgetJob, "_reset_budget_for_team", side_effect=fake_reset_team
    ) as mock_reset_team:
        await job.reset_budget_for_litellm_teams()
        await asyncio.sleep(0.1)

    assert mock_reset_team.call_count == 2
    prisma_client.update_data.assert_awaited_once()
    update_call = prisma_client.update_data.call_args
    assert update_call.kwargs.get("table_name") == "team"
    updated_teams = update_call.kwargs.get("data_list", [])
    assert len(updated_teams) == 1
    assert updated_teams[0]["id"] == "team2"

    failure_hook_calls = (
        proxy_logging_obj.service_logging_obj.async_service_failure_hook.call_args_list
    )
    assert any(
        call.kwargs.get("call_type") == "reset_budget_teams"
        for call in failure_hook_calls
    )


@pytest.mark.asyncio
async def test_reset_budget_continues_other_categories_on_failure():
    """
    Test that executing the overall reset_budget() method continues to process keys, users, and teams,
    even if one of the sub-categories (here, users) experiences a partial failure.

    In this simulation:
      - All keys are processed successfully.
      - One of the two users fails.
      - All teams are processed successfully.

    We then assert that:
      - update_data is called for each category with the correctly updated items.
      - Each get_data call is made (indicating that one failing category did not abort the others).
    """
    # Arrange dummy items for each table
    key1 = {"id": "key1", "spend": 10.0, "budget_duration": 60}
    key2 = {"id": "key2", "spend": 15.0, "budget_duration": 60}
    user1 = {
        "id": "user1",
        "spend": 20.0,
        "budget_duration": 120,
    }  # Will fail in user reset
    user2 = {"id": "user2", "spend": 25.0, "budget_duration": 120}  # Succeeds
    team1 = {"id": "team1", "spend": 30.0, "budget_duration": 180}
    team2 = {"id": "team2", "spend": 35.0, "budget_duration": 180}
    enduser1 = {"user_id": "user1", "spend": 25.0, "budget_id": "budget1"}
    budget1 = LiteLLM_BudgetTableFull(
        **{
            "budget_id": "budget1",
            "max_budget": 65.0,
            "budget_duration": "2d",
            "created_at": datetime.now(timezone.utc) - timedelta(days=3),
        }
    )

    prisma_client = MagicMock()

    async def fake_get_data(*, table_name, query_type, **kwargs):
        if table_name == "key":
            return [key1, key2]
        elif table_name == "user":
            return [user1, user2]
        elif table_name == "team":
            return [team1, team2]
        elif table_name == "budget":
            return [budget1]
        elif table_name == "enduser":
            return [enduser1]
        return []

    prisma_client.get_data = AsyncMock(side_effect=fake_get_data)
    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_key(key, current_time):
        key["spend"] = 0.0
        key["budget_reset_at"] = (
            current_time + timedelta(seconds=key["budget_duration"])
        ).isoformat()
        return key

    async def fake_reset_user(user, current_time):
        if user["id"] == "user1":
            raise Exception("Simulated failure for user1")
        user["spend"] = 0.0
        user["budget_reset_at"] = (
            current_time + timedelta(seconds=user["budget_duration"])
        ).isoformat()
        return user

    async def fake_reset_team(team, current_time):
        team["spend"] = 0.0
        team["budget_reset_at"] = (
            current_time + timedelta(seconds=team["budget_duration"])
        ).isoformat()
        return team

    async def fake_reset_enduser(enduser):
        enduser["spend"] = 0.0
        return enduser

    async def fake_reset_team_members(budgets_to_reset):
        return 1

    with patch.object(
        ResetBudgetJob, "_reset_budget_for_key", side_effect=fake_reset_key
    ) as mock_reset_key, patch.object(
        ResetBudgetJob, "_reset_budget_for_user", side_effect=fake_reset_user
    ) as mock_reset_user, patch.object(
        ResetBudgetJob, "_reset_budget_for_team", side_effect=fake_reset_team
    ) as mock_reset_team, patch.object(
        ResetBudgetJob, "_reset_budget_for_enduser", side_effect=fake_reset_enduser
    ) as mock_reset_enduser, patch.object(
        ResetBudgetJob,
        "reset_budget_for_litellm_team_members",
        side_effect=fake_reset_team_members,
    ) as mock_reset_team_members:
        # Call the overall reset_budget method.
        await job.reset_budget()
        await asyncio.sleep(0.1)

    # Verify that get_data was called for each table. We can check the table names across calls.
    called_tables = {
        call.kwargs.get("table_name") for call in prisma_client.get_data.await_args_list
    }
    if mock_reset_team_members.call_count > 0:
        called_tables.add("team_membership")
    assert called_tables == {
        "key",
        "user",
        "team",
        "budget",
        "enduser",
        "team_membership",
    }

    # Verify that update_data was called three times (one per category, enduser update includes two)
    assert prisma_client.update_data.await_count == 5
    calls = prisma_client.update_data.await_args_list

    # Check keys update: both keys succeed.
    keys_call = calls[0]
    assert keys_call.kwargs.get("table_name") == "key"
    assert len(keys_call.kwargs.get("data_list", [])) == 2

    # Check users update: only user2 succeeded.
    users_call = calls[1]
    assert users_call.kwargs.get("table_name") == "user"
    users_updated = users_call.kwargs.get("data_list", [])
    assert len(users_updated) == 1
    assert users_updated[0]["id"] == "user2"

    # Check teams update: both teams succeed.
    teams_call = calls[2]
    assert teams_call.kwargs.get("table_name") == "team"
    assert len(teams_call.kwargs.get("data_list", [])) == 2

    # Check enduser update: enduser succeed.
    enduser_call = calls[4]
    assert enduser_call.kwargs.get("table_name") == "enduser"
    assert len(enduser_call.kwargs.get("data_list", [])) == 1


# ---------------------------------------------------------------------------
# Additional tests for service logger behavior (keys, users, teams, endusers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_logger_keys_success():
    """
    Test that when resetting keys succeeds (all keys are updated) the service
    logger success hook is called with the correct event metadata and no exception is logged.
    """
    keys = [
        {"id": "key1", "spend": 10.0, "budget_duration": 60},
        {"id": "key2", "spend": 15.0, "budget_duration": 60},
    ]
    prisma_client = MagicMock()
    prisma_client.get_data = AsyncMock(return_value=keys)
    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_key(key, current_time):
        key["spend"] = 0.0
        key["budget_reset_at"] = (
            current_time + timedelta(seconds=key["budget_duration"])
        ).isoformat()
        return key

    with patch.object(
        ResetBudgetJob,
        "_reset_budget_for_key",
        side_effect=fake_reset_key,
    ):
        with patch(
            "litellm.proxy.common_utils.reset_budget_job.verbose_proxy_logger.exception"
        ) as mock_verbose_exc:
            await job.reset_budget_for_litellm_keys()
            # Allow async logging task to complete
            await asyncio.sleep(0.1)
            mock_verbose_exc.assert_not_called()

    # Verify success hook call
    proxy_logging_obj.service_logging_obj.async_service_success_hook.assert_called_once()
    (
        args,
        kwargs,
    ) = proxy_logging_obj.service_logging_obj.async_service_success_hook.call_args
    event_metadata = kwargs.get("event_metadata", {})
    assert event_metadata.get("num_keys_found") == len(keys)
    assert event_metadata.get("num_keys_updated") == len(keys)
    assert event_metadata.get("num_keys_failed") == 0
    # Failure hook should not be executed.
    proxy_logging_obj.service_logging_obj.async_service_failure_hook.assert_not_called()


@pytest.mark.asyncio
async def test_service_logger_keys_failure():
    """
    Test that when a key reset fails the service logger failure hook is called,
    the event metadata reflects the number of keys processed, and that the verbose
    logger exception is called.
    """
    keys = [
        {"id": "key1", "spend": 10.0, "budget_duration": 60},
        {"id": "key2", "spend": 15.0, "budget_duration": 60},
    ]
    prisma_client = MagicMock()
    prisma_client.get_data = AsyncMock(return_value=keys)
    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_key(key, current_time):
        if key["id"] == "key1":
            raise Exception("Simulated failure for key1")
        key["spend"] = 0.0
        key["budget_reset_at"] = (
            current_time + timedelta(seconds=key["budget_duration"])
        ).isoformat()
        return key

    with patch.object(
        ResetBudgetJob,
        "_reset_budget_for_key",
        side_effect=fake_reset_key,
    ):
        with patch(
            "litellm.proxy.common_utils.reset_budget_job.verbose_proxy_logger.exception"
        ) as mock_verbose_exc:
            await job.reset_budget_for_litellm_keys()
            await asyncio.sleep(0.1)
            # Expect at least one exception logged (the inner error and the outer catch)
            assert mock_verbose_exc.call_count >= 1
            # Verify exception was logged with correct message
            assert any(
                "Failed to reset budget for key" in str(call.args)
                for call in mock_verbose_exc.call_args_list
            )

    proxy_logging_obj.service_logging_obj.async_service_failure_hook.assert_called_once()
    (
        args,
        kwargs,
    ) = proxy_logging_obj.service_logging_obj.async_service_failure_hook.call_args
    event_metadata = kwargs.get("event_metadata", {})
    assert event_metadata.get("num_keys_found") == len(keys)
    keys_found_str = event_metadata.get("keys_found", "")
    assert "key1" in keys_found_str
    # Success hook should not be called.
    proxy_logging_obj.service_logging_obj.async_service_success_hook.assert_not_called()


@pytest.mark.asyncio
async def test_service_logger_users_success():
    """
    Test that when resetting users succeeds the service logger success hook is called with
    the correct metadata and no exception is logged.
    """
    users = [
        {"id": "user1", "spend": 20.0, "budget_duration": 120},
        {"id": "user2", "spend": 25.0, "budget_duration": 120},
    ]
    prisma_client = MagicMock()
    prisma_client.get_data = AsyncMock(return_value=users)
    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_user(user, current_time):
        user["spend"] = 0.0
        user["budget_reset_at"] = (
            current_time + timedelta(seconds=user["budget_duration"])
        ).isoformat()
        return user

    with patch.object(
        ResetBudgetJob,
        "_reset_budget_for_user",
        side_effect=fake_reset_user,
    ):
        with patch(
            "litellm.proxy.common_utils.reset_budget_job.verbose_proxy_logger.exception"
        ) as mock_verbose_exc:
            await job.reset_budget_for_litellm_users()
            await asyncio.sleep(0.1)
            mock_verbose_exc.assert_not_called()

    proxy_logging_obj.service_logging_obj.async_service_success_hook.assert_called_once()
    (
        args,
        kwargs,
    ) = proxy_logging_obj.service_logging_obj.async_service_success_hook.call_args
    event_metadata = kwargs.get("event_metadata", {})
    assert event_metadata.get("num_users_found") == len(users)
    assert event_metadata.get("num_users_updated") == len(users)
    assert event_metadata.get("num_users_failed") == 0
    proxy_logging_obj.service_logging_obj.async_service_failure_hook.assert_not_called()


@pytest.mark.asyncio
async def test_service_logger_users_failure():
    """
    Test that a failure during user reset calls the failure hook with appropriate metadata,
    logs the exception, and does not call the success hook.
    """
    users = [
        {"id": "user1", "spend": 20.0, "budget_duration": 120},
        {"id": "user2", "spend": 25.0, "budget_duration": 120},
    ]
    prisma_client = MagicMock()
    prisma_client.get_data = AsyncMock(return_value=users)
    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_user(user, current_time):
        if user["id"] == "user1":
            raise Exception("Simulated failure for user1")
        user["spend"] = 0.0
        user["budget_reset_at"] = (
            current_time + timedelta(seconds=user["budget_duration"])
        ).isoformat()
        return user

    with patch.object(
        ResetBudgetJob,
        "_reset_budget_for_user",
        side_effect=fake_reset_user,
    ):
        with patch(
            "litellm.proxy.common_utils.reset_budget_job.verbose_proxy_logger.exception"
        ) as mock_verbose_exc:
            await job.reset_budget_for_litellm_users()
            await asyncio.sleep(0.1)
            # Verify exception logging
            assert mock_verbose_exc.call_count >= 1
            # Verify exception was logged with correct message
            assert any(
                "Failed to reset budget for user" in str(call.args)
                for call in mock_verbose_exc.call_args_list
            )

    proxy_logging_obj.service_logging_obj.async_service_failure_hook.assert_called_once()
    (
        args,
        kwargs,
    ) = proxy_logging_obj.service_logging_obj.async_service_failure_hook.call_args
    event_metadata = kwargs.get("event_metadata", {})
    assert event_metadata.get("num_users_found") == len(users)
    users_found_str = event_metadata.get("users_found", "")
    assert "user1" in users_found_str
    proxy_logging_obj.service_logging_obj.async_service_success_hook.assert_not_called()


@pytest.mark.asyncio
async def test_service_logger_teams_success():
    """
    Test that when resetting teams is successful the service logger success hook is called with
    the proper metadata and nothing is logged as an exception.
    """
    teams = [
        {"id": "team1", "spend": 30.0, "budget_duration": 180},
        {"id": "team2", "spend": 35.0, "budget_duration": 180},
    ]
    prisma_client = MagicMock()
    prisma_client.get_data = AsyncMock(return_value=teams)
    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_team(team, current_time):
        team["spend"] = 0.0
        team["budget_reset_at"] = (
            current_time + timedelta(seconds=team["budget_duration"])
        ).isoformat()
        return team

    with patch.object(
        ResetBudgetJob,
        "_reset_budget_for_team",
        side_effect=fake_reset_team,
    ):
        with patch(
            "litellm.proxy.common_utils.reset_budget_job.verbose_proxy_logger.exception"
        ) as mock_verbose_exc:
            await job.reset_budget_for_litellm_teams()
            await asyncio.sleep(0.1)
            mock_verbose_exc.assert_not_called()

    proxy_logging_obj.service_logging_obj.async_service_success_hook.assert_called_once()
    (
        args,
        kwargs,
    ) = proxy_logging_obj.service_logging_obj.async_service_success_hook.call_args
    event_metadata = kwargs.get("event_metadata", {})
    assert event_metadata.get("num_teams_found") == len(teams)
    assert event_metadata.get("num_teams_updated") == len(teams)
    assert event_metadata.get("num_teams_failed") == 0
    proxy_logging_obj.service_logging_obj.async_service_failure_hook.assert_not_called()


@pytest.mark.asyncio
async def test_service_logger_teams_failure():
    """
    Test that a failure during team reset triggers the failure hook with proper metadata,
    results in an exception log and no success hook call.
    """
    teams = [
        {"id": "team1", "spend": 30.0, "budget_duration": 180},
        {"id": "team2", "spend": 35.0, "budget_duration": 180},
    ]
    prisma_client = MagicMock()
    prisma_client.get_data = AsyncMock(return_value=teams)
    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_team(team, current_time):
        if team["id"] == "team1":
            raise Exception("Simulated failure for team1")
        team["spend"] = 0.0
        team["budget_reset_at"] = (
            current_time + timedelta(seconds=team["budget_duration"])
        ).isoformat()
        return team

    with patch.object(
        ResetBudgetJob,
        "_reset_budget_for_team",
        side_effect=fake_reset_team,
    ):
        with patch(
            "litellm.proxy.common_utils.reset_budget_job.verbose_proxy_logger.exception"
        ) as mock_verbose_exc:
            await job.reset_budget_for_litellm_teams()
            await asyncio.sleep(0.1)
            # Verify exception logging
            assert mock_verbose_exc.call_count >= 1
            # Verify exception was logged with correct message
            assert any(
                "Failed to reset budget for team" in str(call.args)
                for call in mock_verbose_exc.call_args_list
            )

    proxy_logging_obj.service_logging_obj.async_service_failure_hook.assert_called_once()
    (
        args,
        kwargs,
    ) = proxy_logging_obj.service_logging_obj.async_service_failure_hook.call_args
    event_metadata = kwargs.get("event_metadata", {})
    assert event_metadata.get("num_teams_found") == len(teams)
    teams_found_str = event_metadata.get("teams_found", "")
    assert "team1" in teams_found_str
    proxy_logging_obj.service_logging_obj.async_service_success_hook.assert_not_called()


@pytest.mark.asyncio
async def test_service_logger_endusers_success():
    """
    Test that when resetting endusers succeeds the service logger success hook is called with
    the correct metadata and no exception is logged.
    """
    endusers = [
        {"user_id": "user1", "spend": 25.0, "budget_id": "budget1"},
        {"user_id": "user2", "spend": 25.0, "budget_id": "budget1"},
    ]
    budgets = [
        LiteLLM_BudgetTableFull(
            **{
                "budget_id": "budget1",
                "max_budget": 65.0,
                "budget_duration": "2d",
                "created_at": datetime.now(timezone.utc) - timedelta(days=3),
            }
        )
    ]

    async def fake_get_data(*, table_name, query_type, **kwargs):
        if table_name == "budget":
            return budgets
        elif table_name == "enduser":
            return endusers
        return []

    prisma_client = MagicMock()
    prisma_client.get_data = AsyncMock(side_effect=fake_get_data)
    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_enduser(enduser):
        enduser["spend"] = 0.0
        return enduser

    async def fake_reset_team_members(budgets_to_reset):
        return 1

    with patch.object(
        ResetBudgetJob,
        "_reset_budget_for_enduser",
        side_effect=fake_reset_enduser,
    ) as mock_reset_enduser, patch.object(
        ResetBudgetJob,
        "reset_budget_for_litellm_team_members",
        side_effect=fake_reset_team_members,
    ) as mock_reset_team_members:
        with patch(
            "litellm.proxy.common_utils.reset_budget_job.verbose_proxy_logger.exception"
        ) as mock_verbose_exc:
            await job.reset_budget_for_litellm_budget_table()
            await asyncio.sleep(0.1)
            mock_verbose_exc.assert_not_called()

    proxy_logging_obj.service_logging_obj.async_service_success_hook.assert_called_once()
    (
        args,
        kwargs,
    ) = proxy_logging_obj.service_logging_obj.async_service_success_hook.call_args
    event_metadata = kwargs.get("event_metadata", {})
    assert event_metadata.get("num_budgets_found") == len(budgets)
    assert event_metadata.get("num_endusers_found") == len(endusers)
    assert event_metadata.get("num_endusers_updated") == len(endusers)
    assert event_metadata.get("num_endusers_failed") == 0
    proxy_logging_obj.service_logging_obj.async_service_failure_hook.assert_not_called()


@pytest.mark.asyncio
async def test_service_logger_endusers_failure():
    """
    Test that a failure during enduser reset calls the failure hook with appropriate metadata,
    logs the exception, and does not call the success hook.
    """
    endusers = [
        {"user_id": "user1", "spend": 25.0, "budget_id": "budget1"},
        {"user_id": "user2", "spend": 25.0, "budget_id": "budget1"},
    ]
    budgets = [
        LiteLLM_BudgetTableFull(
            **{
                "budget_id": "budget1",
                "max_budget": 65.0,
                "budget_duration": "2d",
                "created_at": datetime.now(timezone.utc) - timedelta(days=3),
            }
        )
    ]

    async def fake_get_data(*, table_name, query_type, **kwargs):
        if table_name == "budget":
            return budgets
        elif table_name == "enduser":
            return endusers
        return []

    prisma_client = MagicMock()
    prisma_client.get_data = AsyncMock(side_effect=fake_get_data)
    prisma_client.update_data = AsyncMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_enduser(enduser):
        if enduser["user_id"] == "user1":
            raise Exception("Simulated failure for user1")
        enduser["spend"] = 0.0
        return enduser

    async def fake_reset_team_members(budgets_to_reset):
        return 1

    with patch.object(
        ResetBudgetJob,
        "_reset_budget_for_enduser",
        side_effect=fake_reset_enduser,
    ) as mock_reset_enduser, patch.object(
        ResetBudgetJob,
        "reset_budget_for_litellm_team_members",
        side_effect=fake_reset_team_members,
    ) as mock_reset_team_members:
        with patch(
            "litellm.proxy.common_utils.reset_budget_job.verbose_proxy_logger.exception"
        ) as mock_verbose_exc:
            await job.reset_budget_for_litellm_budget_table()
            await asyncio.sleep(0.1)
            # Verify exception logging
            assert mock_verbose_exc.call_count >= 1
            # Verify exception was logged with correct message
            assert any(
                "Failed to reset budget for enduser" in str(call.args)
                for call in mock_verbose_exc.call_args_list
            )

    proxy_logging_obj.service_logging_obj.async_service_failure_hook.assert_called_once()
    (
        args,
        kwargs,
    ) = proxy_logging_obj.service_logging_obj.async_service_failure_hook.call_args
    event_metadata = kwargs.get("event_metadata", {})
    assert event_metadata.get("num_budgets_found") == len(budgets)
    assert event_metadata.get("num_endusers_found") == len(endusers)
    endusers_found_str = event_metadata.get("endusers_found", "")
    assert "user1" in endusers_found_str
    proxy_logging_obj.service_logging_obj.async_service_success_hook.assert_not_called()


@pytest.mark.asyncio
async def test_reset_budget_for_litellm_team_members_called():
    """
    Test that when reset_budget_for_litellm_budget_table is called,
    team members' budgets are also reset via reset_budget_for_litellm_team_members
    """
    # Arrange
    budget1 = LiteLLM_BudgetTableFull(
        **{
            "budget_id": "budget1",
            "max_budget": 100.0,
            "budget_duration": "1d",
            "created_at": datetime.now(timezone.utc) - timedelta(days=2),
        }
    )

    enduser1 = {"user_id": "user1", "spend": 25.0, "budget_id": "budget1"}

    prisma_client = MagicMock()

    async def fake_get_data(*, table_name, query_type, **kwargs):
        if table_name == "budget":
            return [budget1]
        elif table_name == "enduser":
            return [enduser1]
        return []

    prisma_client.get_data = AsyncMock(side_effect=fake_get_data)
    prisma_client.update_data = AsyncMock()

    # Mock the db.litellm_teammembership.update_many call
    prisma_client.db = MagicMock()
    prisma_client.db.litellm_teammembership = MagicMock()
    prisma_client.db.litellm_teammembership.update_many = AsyncMock(
        return_value={"count": 2}
    )

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    job = ResetBudgetJob(proxy_logging_obj, prisma_client)

    async def fake_reset_enduser(enduser):
        enduser["spend"] = 0.0
        return enduser

    with patch.object(
        ResetBudgetJob,
        "_reset_budget_for_enduser",
        side_effect=fake_reset_enduser,
    ):
        # Act
        await job.reset_budget_for_litellm_budget_table()

    # Assert
    # Verify that the team membership update was called
    prisma_client.db.litellm_teammembership.update_many.assert_called_once()

    # Verify the call was made with correct parameters
    call_args = prisma_client.db.litellm_teammembership.update_many.call_args
    assert call_args.kwargs["where"]["budget_id"]["in"] == ["budget1"]
    assert call_args.kwargs["data"]["spend"] == 0
