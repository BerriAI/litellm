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


@pytest.mark.parametrize(
    "model, expect_to_work",
    [("openai/gpt-4o-mini", True), ("openai/gpt-4o", False)],
)
@pytest.mark.asyncio
async def test_can_key_call_model(model, expect_to_work):
    """
    If wildcard model + specific model is used, choose the specific model settings
    """
    from litellm.proxy.auth.auth_checks import can_key_call_model
    from fastapi import HTTPException

    llm_model_list = [
        {
            "model_name": "openai/*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
                "db_model": False,
                "access_groups": ["public-openai-models"],
            },
        },
        {
            "model_name": "openai/gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "0cfcd87f2cb12a783a466888d05c6c89df66db23e01cecd75ec0b83aed73c9ad",
                "db_model": False,
                "access_groups": ["private-openai-models"],
            },
        },
    ]
    router = litellm.Router(model_list=llm_model_list)
    args = {
        "model": model,
        "llm_model_list": llm_model_list,
        "valid_token": UserAPIKeyAuth(
            models=["public-openai-models"],
        ),
        "llm_router": router,
    }
    if expect_to_work:
        await can_key_call_model(**args)
    else:
        with pytest.raises(Exception) as e:
            await can_key_call_model(**args)

        print(e)


@pytest.mark.parametrize(
    "model, expect_to_work",
    [("openai/gpt-4o", False), ("openai/gpt-4o-mini", True)],
)
@pytest.mark.asyncio
async def test_can_team_call_model(model, expect_to_work):
    from litellm.proxy.auth.auth_checks import model_in_access_group
    from fastapi import HTTPException

    llm_model_list = [
        {
            "model_name": "openai/*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
                "db_model": False,
                "access_groups": ["public-openai-models"],
            },
        },
        {
            "model_name": "openai/gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "0cfcd87f2cb12a783a466888d05c6c89df66db23e01cecd75ec0b83aed73c9ad",
                "db_model": False,
                "access_groups": ["private-openai-models"],
            },
        },
    ]
    router = litellm.Router(model_list=llm_model_list)

    args = {
        "model": model,
        "team_models": ["public-openai-models"],
        "llm_router": router,
    }
    if expect_to_work:
        assert model_in_access_group(**args)
    else:
        assert not model_in_access_group(**args)
