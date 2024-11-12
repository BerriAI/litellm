# What is this?
## Tests if 'get_end_user_object' works as expected

import sys, os, asyncio, time, random, uuid
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, litellm
import httpx
from litellm.proxy.auth.auth_checks import (
    _handle_failed_db_connection_for_get_key_object,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import get_end_user_object
from litellm.caching.caching import DualCache
from litellm.proxy._types import LiteLLM_EndUserTable, LiteLLM_BudgetTable
from litellm.proxy.utils import PrismaClient


@pytest.mark.parametrize("customer_spend, customer_budget", [(0, 10), (10, 0)])
@pytest.mark.asyncio
async def test_get_end_user_object(customer_spend, customer_budget):
    """
    Scenario 1: normal
    Scenario 2: user over budget
    """
    end_user_id = "my-test-customer"
    _budget = LiteLLM_BudgetTable(max_budget=customer_budget)
    end_user_obj = LiteLLM_EndUserTable(
        user_id=end_user_id,
        spend=customer_spend,
        litellm_budget_table=_budget,
        blocked=False,
    )
    _cache = DualCache()
    _key = "end_user_id:{}".format(end_user_id)
    _cache.set_cache(key=_key, value=end_user_obj)
    try:
        await get_end_user_object(
            end_user_id=end_user_id,
            prisma_client="RANDOM VALUE",  # type: ignore
            user_api_key_cache=_cache,
        )
        if customer_spend > customer_budget:
            pytest.fail(
                "Expected call to fail. Customer Spend={}, Customer Budget={}".format(
                    customer_spend, customer_budget
                )
            )
    except Exception as e:
        if (
            isinstance(e, litellm.BudgetExceededError)
            and customer_spend > customer_budget
        ):
            pass
        else:
            pytest.fail(
                "Expected call to work. Customer Spend={}, Customer Budget={}, Error={}".format(
                    customer_spend, customer_budget, str(e)
                )
            )


@pytest.mark.asyncio
async def test_handle_failed_db_connection():
    """
    Test cases:
    1. When allow_requests_on_db_unavailable=True -> return UserAPIKeyAuth
    2. When allow_requests_on_db_unavailable=False -> raise original error
    """
    from litellm.proxy.proxy_server import general_settings, litellm_proxy_admin_name

    # Test case 1: allow_requests_on_db_unavailable=True
    general_settings["allow_requests_on_db_unavailable"] = True
    mock_error = httpx.ConnectError("Failed to connect to DB")

    result = await _handle_failed_db_connection_for_get_key_object(e=mock_error)

    assert isinstance(result, UserAPIKeyAuth)
    assert result.key_name == "failed-to-connect-to-db"
    assert result.token == "failed-to-connect-to-db"
    assert result.user_id == litellm_proxy_admin_name

    # Test case 2: allow_requests_on_db_unavailable=False
    general_settings["allow_requests_on_db_unavailable"] = False

    with pytest.raises(httpx.ConnectError) as exc_info:
        await _handle_failed_db_connection_for_get_key_object(e=mock_error)
    print("_handle_failed_db_connection_for_get_key_object got exception", exc_info)

    assert str(exc_info.value) == "Failed to connect to DB"
