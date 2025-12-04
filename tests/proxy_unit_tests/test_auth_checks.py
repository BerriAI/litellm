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
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import get_end_user_object
from litellm.caching.caching import DualCache
from litellm.proxy._types import (
    LiteLLM_EndUserTable,
    LiteLLM_BudgetTable,
    LiteLLM_UserTable,
    LiteLLM_TeamTable,
)
from litellm.proxy.utils import PrismaClient
from litellm.proxy.auth.auth_checks import (
    can_team_access_model,
    _virtual_key_soft_budget_check,
)
from litellm.proxy.utils import ProxyLogging
from litellm.proxy.utils import CallInfo


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
    _cache.set_cache(key=_key, value=end_user_obj.model_dump())
    try:
        await get_end_user_object(
            end_user_id=end_user_id,
            prisma_client="RANDOM VALUE",  # type: ignore
            user_api_key_cache=_cache,
            route="/v1/chat/completions",
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


@pytest.mark.parametrize(
    "model, expect_to_work",
    [
        ("openai/gpt-4o-mini", True),
        ("openai/gpt-4o", False),
    ],
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


@pytest.mark.parametrize(
    "key_models, model, expect_to_work",
    [
        (["openai/*"], "openai/gpt-4o", True),
        (["openai/*"], "openai/gpt-4o-mini", True),
        (["openai/*"], "openaiz/gpt-4o-mini", False),
        (["bedrock/*"], "bedrock/anthropic.claude-3-5-sonnet-20240620", True),
        (["bedrock/*"], "bedrockz/anthropic.claude-3-5-sonnet-20240620", False),
        (["bedrock/us.*"], "bedrock/us.amazon.nova-micro-v1:0", True),
    ],
)
@pytest.mark.asyncio
async def test_can_key_call_model_wildcard_access(key_models, model, expect_to_work):
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
            },
        },
        {
            "model_name": "bedrock/*",
            "litellm_params": {
                "model": "bedrock/*",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
                "db_model": False,
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
            },
        },
    ]
    router = litellm.Router(model_list=llm_model_list)

    user_api_key_object = UserAPIKeyAuth(
        models=key_models,
    )

    if expect_to_work:
        await can_key_call_model(
            model=model,
            llm_model_list=llm_model_list,
            valid_token=user_api_key_object,
            llm_router=router,
        )
    else:
        with pytest.raises(Exception) as e:
            await can_key_call_model(
                model=model,
                llm_model_list=llm_model_list,
                valid_token=user_api_key_object,
                llm_router=router,
            )

            print(e)


@pytest.mark.asyncio
async def test_is_valid_fallback_model():
    from litellm.proxy.auth.auth_checks import is_valid_fallback_model
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "openai/gpt-3.5-turbo"},
            }
        ]
    )

    try:
        await is_valid_fallback_model(
            model="gpt-3.5-turbo", llm_router=router, user_model=None
        )
    except Exception as e:
        pytest.fail(f"Expected is_valid_fallback_model to work, got exception: {e}")

    try:
        await is_valid_fallback_model(
            model="gpt-4o", llm_router=router, user_model=None
        )
        pytest.fail("Expected is_valid_fallback_model to fail")
    except Exception as e:
        assert "Invalid" in str(e)


@pytest.mark.parametrize(
    "token_spend, max_budget, expect_budget_error",
    [
        (5.0, 10.0, False),  # Under budget
        (10.0, 10.0, True),  # At budget limit
        (15.0, 10.0, True),  # Over budget
    ],
)
@pytest.mark.asyncio
async def test_virtual_key_max_budget_check(
    token_spend, max_budget, expect_budget_error
):
    """
    Test if virtual key budget checks work as expected:
    1. Triggers budget alert for all cases
    2. Raises BudgetExceededError when spend >= max_budget
    """
    from litellm.proxy.auth.auth_checks import _virtual_key_max_budget_check
    from litellm.proxy.utils import ProxyLogging

    # Setup test data
    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=token_spend,
        max_budget=max_budget,
        user_id="test-user",
        key_alias="test-key",
    )

    user_obj = LiteLLM_UserTable(
        user_id="test-user",
        user_email="test@email.com",
        max_budget=None,
    )

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=None,
    )

    # Track if budget alert was called
    alert_called = False

    async def mock_budget_alert(*args, **kwargs):
        nonlocal alert_called
        alert_called = True

    proxy_logging_obj.budget_alerts = mock_budget_alert

    try:
        await _virtual_key_max_budget_check(
            valid_token=valid_token,
            proxy_logging_obj=proxy_logging_obj,
            user_obj=user_obj,
        )
        if expect_budget_error:
            pytest.fail(
                f"Expected BudgetExceededError for spend={token_spend}, max_budget={max_budget}"
            )
    except litellm.BudgetExceededError as e:
        if not expect_budget_error:
            pytest.fail(
                f"Unexpected BudgetExceededError for spend={token_spend}, max_budget={max_budget}"
            )
        assert e.current_cost == token_spend
        assert e.max_budget == max_budget

    await asyncio.sleep(1)

    # Verify budget alert was triggered
    assert alert_called, "Budget alert should be triggered"


@pytest.mark.parametrize(
    "model, team_models, expect_to_work",
    [
        ("gpt-4", ["gpt-4"], True),  # exact match
        ("gpt-4", ["all-proxy-models"], True),  # all-proxy-models access
        ("gpt-4", ["*"], True),  # wildcard access
        ("gpt-4", ["openai/*"], True),  # openai wildcard access
        (
            "bedrock/anthropic.claude-3-5-sonnet-20240620",
            ["bedrock/*"],
            True,
        ),  # wildcard access
        (
            "bedrockz/anthropic.claude-3-5-sonnet-20240620",
            ["bedrock/*"],
            False,
        ),  # non-match wildcard access
        ("bedrock/very_new_model", ["bedrock/*"], True),  # bedrock wildcard access
        (
            "bedrock/claude-3-5-sonnet-20240620",
            ["bedrock/claude-*"],
            True,
        ),  # match on pattern
        (
            "bedrock/claude-3-6-sonnet-20240620",
            ["bedrock/claude-3-5-*"],
            False,
        ),  # don't match on pattern
        ("openai/gpt-4o", ["openai/*"], True),  # openai wildcard access
        ("gpt-4", ["gpt-3.5-turbo"], False),  # model not in allowed list
        ("claude-3", [], True),  # empty model list (allows all)
    ],
)
@pytest.mark.asyncio
async def test_can_team_access_model(model, team_models, expect_to_work):
    """
    Test cases for can_team_access_model:
    1. Exact model match
    2. all-proxy-models access
    3. Wildcard (*) access
    4. OpenAI wildcard access
    5. Model not in allowed list
    6. Empty model list
    7. None model list
    """
    try:
        team_object = LiteLLM_TeamTable(
            team_id="test-team",
            models=team_models,
        )
        result = can_team_access_model(
            model=model,
            team_object=team_object,
            llm_router=None,
            team_model_aliases=None,
        )
        if not expect_to_work:
            pytest.fail(
                f"Expected model access check to fail for model={model}, team_models={team_models}"
            )
    except Exception as e:
        if expect_to_work:
            pytest.fail(
                f"Expected model access check to work for model={model}, team_models={team_models}. Got error: {str(e)}"
            )


@pytest.mark.parametrize(
    "spend, soft_budget, expect_alert",
    [
        (100, 50, True),  # Over soft budget
        (50, 50, True),  # At soft budget
        (25, 50, False),  # Under soft budget
        (100, None, False),  # No soft budget set
    ],
)
@pytest.mark.asyncio
async def test_virtual_key_soft_budget_check(spend, soft_budget, expect_alert):
    """
    Test cases for _virtual_key_soft_budget_check:
    1. Spend over soft budget
    2. Spend at soft budget
    3. Spend under soft budget
    4. No soft budget set
    """
    alert_triggered = False

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered
            alert_triggered = True
            assert type == "soft_budget"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=spend,
        soft_budget=soft_budget,
        user_id="test-user",
        team_id="test-team",
        key_alias="test-key",
    )

    proxy_logging_obj = MockProxyLogging()

    await _virtual_key_soft_budget_check(
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
    )

    await asyncio.sleep(0.1)  # Allow time for the alert task to complete

    assert (
        alert_triggered == expect_alert
    ), f"Expected alert_triggered to be {expect_alert} for spend={spend}, soft_budget={soft_budget}"


@pytest.mark.asyncio
async def test_can_user_call_model():
    from litellm.proxy.auth.auth_checks import can_user_call_model
    from litellm.proxy._types import ProxyException
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "anthropic-claude",
                "litellm_params": {"model": "anthropic/anthropic-claude"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "test-api-key"},
            },
        ]
    )

    args = {
        "model": "anthropic-claude",
        "llm_router": router,
        "user_object": LiteLLM_UserTable(
            user_id="testuser21@mycompany.com",
            max_budget=None,
            spend=0.0042295,
            model_max_budget={},
            model_spend={},
            user_email="testuser@mycompany.com",
            models=["gpt-3.5-turbo"],
        ),
    }

    with pytest.raises(ProxyException) as e:
        await can_user_call_model(**args)

    args["model"] = "gpt-3.5-turbo"
    await can_user_call_model(**args)


@pytest.mark.asyncio
async def test_can_user_call_model_with_no_default_models():
    from litellm.proxy.auth.auth_checks import can_user_call_model
    from litellm.proxy._types import ProxyException, SpecialModelNames
    from unittest.mock import MagicMock

    args = {
        "model": "anthropic-claude",
        "llm_router": MagicMock(),
        "user_object": LiteLLM_UserTable(
            user_id="testuser21@mycompany.com",
            max_budget=None,
            spend=0.0042295,
            model_max_budget={},
            model_spend={},
            user_email="testuser@mycompany.com",
            models=[SpecialModelNames.no_default_models.value],
        ),
    }

    with pytest.raises(ProxyException) as e:
        await can_user_call_model(**args)


@pytest.mark.asyncio
async def test_get_fuzzy_user_object():
    from litellm.proxy.auth.auth_checks import _get_fuzzy_user_object
    from litellm.proxy.utils import PrismaClient
    from unittest.mock import AsyncMock, MagicMock

    # Setup mock Prisma client
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_usertable = MagicMock()

    # Mock user data
    test_user = LiteLLM_UserTable(
        user_id="test_123",
        sso_user_id="sso_123",
        user_email="test@example.com",
        organization_memberships=[],
        max_budget=None,
    )

    # Test 1: Find user by SSO ID
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=test_user)
    result = await _get_fuzzy_user_object(
        prisma_client=mock_prisma, sso_user_id="sso_123", user_email="test@example.com"
    )
    assert result == test_user
    mock_prisma.db.litellm_usertable.find_unique.assert_called_with(
        where={"sso_user_id": "sso_123"}, include={"organization_memberships": True}
    )

    # Test 2: SSO ID not found, find by email
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=test_user)
    mock_prisma.db.litellm_usertable.update = AsyncMock()

    result = await _get_fuzzy_user_object(
        prisma_client=mock_prisma,
        sso_user_id="new_sso_456",
        user_email="test@example.com",
    )
    assert result == test_user
    mock_prisma.db.litellm_usertable.find_first.assert_called_with(
        where={"user_email": "test@example.com"},
        include={"organization_memberships": True},
    )

    # Test 3: Verify background SSO update task when user found by email
    await asyncio.sleep(0.1)  # Allow time for background task
    mock_prisma.db.litellm_usertable.update.assert_called_with(
        where={"user_id": "test_123"}, data={"sso_user_id": "new_sso_456"}
    )

    # Test 4: User not found by either method
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    result = await _get_fuzzy_user_object(
        prisma_client=mock_prisma,
        sso_user_id="unknown_sso",
        user_email="unknown@example.com",
    )
    assert result is None

    # Test 5: Only email provided (no SSO ID)
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=test_user)
    result = await _get_fuzzy_user_object(
        prisma_client=mock_prisma, user_email="test@example.com"
    )
    assert result == test_user
    mock_prisma.db.litellm_usertable.find_first.assert_called_with(
        where={"user_email": "test@example.com"},
        include={"organization_memberships": True},
    )

    # Test 6: Only SSO ID provided (no email)
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=test_user)
    result = await _get_fuzzy_user_object(
        prisma_client=mock_prisma, sso_user_id="sso_123"
    )
    assert result == test_user
    mock_prisma.db.litellm_usertable.find_unique.assert_called_with(
        where={"sso_user_id": "sso_123"}, include={"organization_memberships": True}
    )


@pytest.mark.parametrize(
    "model, alias_map, expect_to_work",
    [
        ("gpt-4", {"gpt-4": "gpt-4-team1"}, True),  # model matches alias value
        ("gpt-5", {"gpt-4": "gpt-4-team1"}, False),
    ],
)
@pytest.mark.asyncio
async def test_can_key_call_model_with_aliases(model, alias_map, expect_to_work):
    """
    Test if can_key_call_model correctly handles model aliases in the token
    """
    from litellm.proxy.auth.auth_checks import can_key_call_model

    llm_model_list = [
        {
            "model_name": "gpt-4-team1",
            "litellm_params": {
                "model": "gpt-4",
                "api_key": "test-api-key",
            },
        }
    ]
    router = litellm.Router(model_list=llm_model_list)

    user_api_key_object = UserAPIKeyAuth(
        models=[
            "gpt-4-team1",
        ],
        team_model_aliases=alias_map,
    )

    if expect_to_work:
        await can_key_call_model(
            model=model,
            llm_model_list=llm_model_list,
            valid_token=user_api_key_object,
            llm_router=router,
        )
    else:
        with pytest.raises(Exception) as e:
            await can_key_call_model(
                model=model,
                llm_model_list=llm_model_list,
                valid_token=user_api_key_object,
                llm_router=router,
            )
