# What is this?
## Unit tests for the /budget/* endpoints
from litellm._uuid import uuid
from datetime import datetime, timezone

import aiohttp
import pytest
import pytest_asyncio

from litellm.litellm_core_utils.duration_parser import get_next_standardized_reset_time
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_timezone


def _parse_budget_api_datetime(value: str) -> datetime:
    """Parse ISO timestamps returned by the proxy JSON API."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


async def delete_budget(session, budget_id):
    url = "http://0.0.0.0:4000/budget/delete"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"id": budget_id}
    async with session.post(url, headers=headers, json=data) as response:
        assert response.status == 200
        print(f"Deleted Budget {budget_id}")


async def create_budget(session, data):
    url = "http://0.0.0.0:4000/budget/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}

    async with session.post(url, headers=headers, json=data) as response:
        assert response.status == 200
        response_data = await response.json()
        budget_id = response_data["budget_id"]
        print(f"Created Budget {budget_id}")
        return response_data


@pytest_asyncio.fixture
async def budget_setup():
    """
    Fixture to create a budget for testing and clean it up afterward.

    This fixture performs the following steps:
      1. Opens an aiohttp ClientSession.
      2. Generates a random budget_id and defines the budget data (duration: 1 day, max_budget: 0.02).
      3. Calls create_budget to create the budget.
      4. Yields the budget_response (a dict) for use in the test.
      5. After the test completes, deletes the created budget by calling delete_budget.

    Returns:
        dict: The JSON response from create_budget, which includes the created budget's data.
    """

    async with aiohttp.ClientSession() as session:
        # Generate a unique budget_id and define the budget data.
        budget_id = f"budget-{uuid.uuid4()}"
        data = {"budget_id": budget_id, "budget_duration": "1d", "max_budget": 0.02}
        budget_response = await create_budget(session, data)

        # Yield the response so the test can use it.
        yield budget_response

        # After the test, delete the created budget to clean up.
        await delete_budget(session, budget_id)


@pytest.mark.asyncio
async def test_create_budget_with_duration(budget_setup):
    """
    Test creating a budget with a specified duration and verify that 'budget_reset_at'
    matches the next standardized reset (see get_budget_reset_time / new_budget), not
    necessarily created_at + wall-clock duration.
    """

    assert (
        budget_setup["budget_reset_at"] is not None
    ), "The budget_reset_at field should not be None"

    created_at = _parse_budget_api_datetime(budget_setup["created_at"])
    expected_reset_at = get_next_standardized_reset_time(
        duration=budget_setup["budget_duration"],
        current_time=created_at,
        timezone_str=get_budget_reset_timezone(),
    )

    actual_reset_at = _parse_budget_api_datetime(budget_setup["budget_reset_at"])

    tolerance_seconds = 3
    time_difference = abs(
        (actual_reset_at - expected_reset_at).total_seconds()
    )

    assert time_difference <= tolerance_seconds, (
        f"Expected budget_reset_at to be within {tolerance_seconds} seconds of {expected_reset_at}, "
        f"but the difference was {time_difference} seconds."
    )
