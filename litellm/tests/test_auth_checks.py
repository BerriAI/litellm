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
from litellm.proxy.auth.auth_checks import get_end_user_object
from litellm.caching import DualCache
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
