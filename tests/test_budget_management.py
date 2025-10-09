# What is this?
## Unit tests for the /budget/* endpoints
from litellm._uuid import uuid
from datetime import datetime, timedelta

import aiohttp
import pytest
import pytest_asyncio


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
    Test creating a budget with a specified duration and verify that the 'budget_reset_at'
    timestamp is correctly calculated as 'created_at' plus the budget duration (one day).

    This test uses the budget_setup fixture, which handles both the creation and cleanup of the budget.
    """

    # Verify that the response includes a 'budget_reset_at' timestamp.
    assert (
        budget_setup["budget_reset_at"] is not None
    ), "The budget_reset_at field should not be None"

    # Calculate the expected reset time: created_at + 1 day.
    expected_reset_at_date = datetime.fromisoformat(
        budget_setup["created_at"]
    ) + timedelta(days=1)

    # Allow for a small tolerance in seconds for the timestamp calculation.
    tolerance_seconds = 3
    actual_reset_at_date = datetime.fromisoformat(budget_setup["budget_reset_at"])
    time_difference = abs(
        (actual_reset_at_date - expected_reset_at_date).total_seconds()
    )

    assert time_difference <= tolerance_seconds, (
        f"Expected budget_reset_at to be within {tolerance_seconds} seconds of {expected_reset_at_date}, "
        f"but the difference was {time_difference} seconds."
    )
