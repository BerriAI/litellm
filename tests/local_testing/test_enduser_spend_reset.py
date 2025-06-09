import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import aiohttp
import pytest
import pytest_asyncio
from dotenv import load_dotenv

from litellm.caching.caching import DualCache
from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
from litellm.proxy.utils import PrismaClient, ProxyLogging

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

load_dotenv()


async def create_budget(session, data):
    url = "http://0.0.0.0:4000/budget/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}

    async with session.post(url, headers=headers, json=data) as response:
        assert response.status == 200
        response_data = await response.json()
        budget_id = response_data["budget_id"]
        print(f"Created Budget {budget_id}")
        return response_data


async def create_end_user(prisma_client, session, user_id, budget_id, spend=None):
    url = "http://0.0.0.0:4000/end_user/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "user_id": user_id,
        "budget_id": budget_id,
    }

    async with session.post(url, headers=headers, json=data) as response:
        assert response.status == 200
        response_data = await response.json()
        end_user_id = response_data["user_id"]
        print(f"Created End User {end_user_id}")

        if spend is not None:
            end_users = await prisma_client.get_data(
                table_name="enduser",
                query_type="find_all",
                budget_id_list=[budget_id],
            )
            end_user = [user for user in end_users if user.user_id == user_id][0]
            end_user.spend = spend
            await prisma_client.update_data(
                query_type="update_many",
                data_list=[end_user],
                table_name="enduser",
            )

        return response_data


async def delete_budget(session, budget_id):
    url = "http://0.0.0.0:4000/budget/delete"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"id": budget_id}
    async with session.post(url, headers=headers, json=data) as response:
        assert response.status == 200
        print(f"Deleted Budget {budget_id}")


async def delete_end_user(session, user_id):
    url = "http://0.0.0.0:4000/end_user/delete"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"user_ids": [user_id]}
    async with session.post(url, headers=headers, json=data) as response:
        assert response.status == 200
        print(f"Deleted End User {user_id}")


@pytest.fixture
def prisma_client():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )
    return prisma_client


class MockProxyLogging:
    class MockServiceLogging:
        async def async_service_success_hook(self, **kwargs):
            pass

        async def async_service_failure_hook(self, **kwargs):
            pass

    def __init__(self):
        self.service_logging_obj = self.MockServiceLogging()


@pytest.fixture
def mock_proxy_logging():
    return MockProxyLogging()


@pytest.fixture
def reset_budget_job(prisma_client, mock_proxy_logging):
    return ResetBudgetJob(
        proxy_logging_obj=mock_proxy_logging, prisma_client=prisma_client
    )


@pytest_asyncio.fixture
async def budget_and_enduser_setup(prisma_client):
    """
    Fixture to set up budgets and end users for testing and clean them up afterward.

    This fixture performs the following:
      - Creates two budgets:
          * Budget X with a short duration ("5s").
          * Budget Y with a long duration ("30d").
      - Stores the initial 'budget_reset_at' timestamps for both budgets.
      - Creates three end users:
          * End Users A and B are associated with Budget X and are given initial spend values.
          * End User C is associated with Budget Y with an initial spend.
      - After the test (after the yield), the created end users and budgets are deleted.
    """
    await prisma_client.connect()

    async with aiohttp.ClientSession() as session:
        # Create budgets
        id_budget_x = f"budget-{uuid.uuid4()}"
        data_budget_x = {
            "budget_id": id_budget_x,
            "budget_duration": "5s",
            "max_budget": 2,
        }
        id_budget_y = f"budget-{uuid.uuid4()}"
        data_budget_y = {
            "budget_id": id_budget_y,
            "budget_duration": "30d",
            "max_budget": 1,
        }
        response_budget_x = await create_budget(session, data_budget_x)
        initial_budget_x_reset_at_date = datetime.fromisoformat(
            response_budget_x["budget_reset_at"]
        )
        response_budget_y = await create_budget(session, data_budget_y)
        initial_budget_y_reset_at_date = datetime.fromisoformat(
            response_budget_y["budget_reset_at"]
        )

        # Create end users
        id_end_user_a = f"test-user-{uuid.uuid4()}"
        id_end_user_b = f"test-user-{uuid.uuid4()}"
        id_end_user_c = f"test-user-{uuid.uuid4()}"
        await create_end_user(
            prisma_client, session, id_end_user_a, id_budget_x, spend=0.16
        )
        await create_end_user(
            prisma_client, session, id_end_user_b, id_budget_x, spend=0.04
        )
        await create_end_user(
            prisma_client, session, id_end_user_c, id_budget_y, spend=0.2
        )

        # Bundle data needed for the test
        setup_data = {
            "budgets": {
                "id_budget_x": id_budget_x,
                "id_budget_y": id_budget_y,
                "initial_budget_x_reset_at_date": initial_budget_x_reset_at_date,
                "initial_budget_y_reset_at_date": initial_budget_y_reset_at_date,
            },
            "end_users": {
                "id_end_user_a": id_end_user_a,
                "id_end_user_b": id_end_user_b,
                "id_end_user_c": id_end_user_c,
            },
        }

        # Provide the setup data to the test
        yield setup_data

        # Clean-up: Delete the created test data
        await delete_end_user(session, id_end_user_a)
        await delete_end_user(session, id_end_user_b)
        await delete_end_user(session, id_end_user_c)
        await delete_budget(session, id_budget_x)
        await delete_budget(session, id_budget_y)


@pytest.mark.asyncio
async def test_reset_budget_for_endusers(
    reset_budget_job, prisma_client, budget_and_enduser_setup
):
    """
    Test the part "Reset End-User (Customer) Spend and corresponding Budget duration" in reset_budget function.

    This test uses the budget_and_enduser_setup fixture to create budgets and end users,
    waits for the short-duration budget to expire, calls reset_budget, and verifies that:
      - End users associated with the short-duration budget X have their spend reset to 0.
      - The budget_reset_at timestamp for the short-duration budget X is updated,
        while the long-duration budget Y remains unchanged.
    """

    # Unpack the required data from the fixture
    budgets = budget_and_enduser_setup["budgets"]
    end_users = budget_and_enduser_setup["end_users"]

    id_budget_x = budgets["id_budget_x"]
    id_budget_y = budgets["id_budget_y"]
    initial_budget_x_reset_at_date = budgets["initial_budget_x_reset_at_date"]
    initial_budget_y_reset_at_date = budgets["initial_budget_y_reset_at_date"]

    id_end_user_a = end_users["id_end_user_a"]
    id_end_user_b = end_users["id_end_user_b"]
    id_end_user_c = end_users["id_end_user_c"]

    # Wait for Budget X to expire (short duration "5s" plus a small buffer)
    await asyncio.sleep(6)

    # Call the reset_budget function:
    # It should reset the spend values for end users associated with Budget X.
    await reset_budget_job.reset_budget_for_litellm_endusers()

    # Retrieve updated data for end users
    updated_end_users = await prisma_client.get_data(
        table_name="enduser",
        query_type="find_all",
        budget_id_list=[id_budget_x, id_budget_y],
    )
    # Retrieve updated data for budgets
    updated_budgets = await prisma_client.get_data(
        table_name="budget",
        query_type="find_all",
        reset_at=datetime.now(timezone.utc) + timedelta(days=31),
    )

    # Assertions for end users
    user_a = [user for user in updated_end_users if user.user_id == id_end_user_a][0]
    user_b = [user for user in updated_end_users if user.user_id == id_end_user_b][0]
    user_c = [user for user in updated_end_users if user.user_id == id_end_user_c][0]

    assert user_a.spend == 0, "Spend for end_user_a was not reset to 0"
    assert user_b.spend == 0, "Spend for end_user_b was not reset to 0"
    assert user_c.spend > 0, "Spend for end_user_c should not be reset"

    # Assertions for budgets
    budget_x = [
        budget for budget in updated_budgets if budget.budget_id == id_budget_x
    ][0]
    budget_y = [
        budget for budget in updated_budgets if budget.budget_id == id_budget_y
    ][0]

    assert (
        budget_x.budget_reset_at > initial_budget_x_reset_at_date
    ), "Budget X budget_reset_at was not updated"
    assert (
        budget_y.budget_reset_at == initial_budget_y_reset_at_date
    ), "Budget Y budget_reset_at should remain unchanged"
