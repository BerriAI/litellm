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
    Litellm_EntityType,
)
from litellm.proxy.utils import PrismaClient
from litellm.proxy.auth.auth_checks import (
    can_team_access_model,
    _virtual_key_soft_budget_check,
    _team_soft_budget_check,
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
        result = await can_team_access_model(
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


@pytest.mark.parametrize(
    "spend, soft_budget, expect_alert, metadata, expected_alert_emails",
    [
        (100, 50, False, None, None),  # Over soft budget, no metadata - no alert_emails configured, so no alert
        (50, 50, False, None, None),  # At soft budget, no metadata - no alert_emails configured, so no alert
        (25, 50, False, None, None),  # Under soft budget
        (100, None, False, None, None),  # No soft budget set
        (100, 50, True, {"soft_budget_alerting_emails": ["team1@example.com", "team2@example.com"]}, ["team1@example.com", "team2@example.com"]),  # Over soft budget with list of emails
        (100, 50, True, {"soft_budget_alerting_emails": "team1@example.com,team2@example.com"}, ["team1@example.com", "team2@example.com"]),  # Over soft budget with comma-separated emails
        (100, 50, True, {"soft_budget_alerting_emails": ["team1@example.com", "", "  ", "team2@example.com"]}, ["team1@example.com", "team2@example.com"]),  # Over soft budget with empty strings filtered
    ],
)
@pytest.mark.asyncio
async def test_team_soft_budget_check(spend, soft_budget, expect_alert, metadata, expected_alert_emails):
    """
    Test cases for _team_soft_budget_check:
    1. Spend over soft budget, no alert_emails configured - should NOT trigger alert (alerts only sent when alert_emails configured)
    2. Spend at soft budget, no alert_emails configured - should NOT trigger alert (alerts only sent when alert_emails configured)
    3. Spend under soft budget - should not trigger alert
    4. No soft budget set - should not trigger alert
    5. Team with alert emails in metadata (list) - should include alert_emails in CallInfo
    6. Team with alert emails in metadata (comma-separated string) - should parse and include alert_emails
    7. Team with alert emails containing empty strings - should filter them out
    """
    alert_triggered = False
    captured_call_info = None

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered, captured_call_info
            alert_triggered = True
            captured_call_info = user_info
            assert type == "soft_budget"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
        team_alias="test-team-alias",
        key_alias="test-key",
    )

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        spend=spend,
        soft_budget=soft_budget,
        max_budget=100.0,
        metadata=metadata,
    )

    proxy_logging_obj = MockProxyLogging()

    await _team_soft_budget_check(
        team_object=team_object,
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
    )

    await asyncio.sleep(0.1)  # Allow time for the alert task to complete

    assert (
        alert_triggered == expect_alert
    ), f"Expected alert_triggered to be {expect_alert} for spend={spend}, soft_budget={soft_budget}"

    if expect_alert:
        assert captured_call_info is not None
        assert captured_call_info.team_id == "test-team"
        assert captured_call_info.spend == spend
        assert captured_call_info.soft_budget == soft_budget
        assert captured_call_info.event_group == Litellm_EntityType.TEAM
        # Verify alert_emails if expected
        if expected_alert_emails is not None:
            assert captured_call_info.alert_emails == expected_alert_emails
        else:
            assert captured_call_info.alert_emails is None or captured_call_info.alert_emails == []


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
        where={"user_email": {"equals": "test@example.com", "mode": "insensitive"}},
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
        where={"user_email": {"equals": "test@example.com", "mode": "insensitive"}},
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


# ---------------------------------------------------------------------------
# Access group cache helpers (_cache_access_object, _delete_cache_access_object)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_access_object():
    """Test _cache_access_object stores access group in cache with correct key."""
    from litellm.proxy.auth.auth_checks import _cache_access_object
    from litellm.proxy._types import LiteLLM_AccessGroupTable

    cache = DualCache()
    ag_id = "ag-test-123"
    ag_table = LiteLLM_AccessGroupTable(
        access_group_id=ag_id,
        access_group_name="test-group",
        access_model_names=["gpt-4"],
    )
    await _cache_access_object(
        access_group_id=ag_id,
        access_group_table=ag_table,
        user_api_key_cache=cache,
    )
    cached = await cache.async_get_cache(key=f"access_group_id:{ag_id}")
    assert cached is not None
    if isinstance(cached, dict):
        assert cached.get("access_group_id") == ag_id
        assert cached.get("access_group_name") == "test-group"
    else:
        assert cached.access_group_id == ag_id
        assert cached.access_group_name == "test-group"


@pytest.mark.asyncio
async def test_delete_cache_access_object():
    """Test _delete_cache_access_object removes access group from in-memory cache."""
    from litellm.proxy.auth.auth_checks import _delete_cache_access_object
    from litellm.proxy._types import LiteLLM_AccessGroupTable

    cache = DualCache()
    ag_id = "ag-delete-test"
    ag_table = LiteLLM_AccessGroupTable(
        access_group_id=ag_id,
        access_group_name="to-delete",
    )
    await cache.async_set_cache(key=f"access_group_id:{ag_id}", value=ag_table, ttl=60)
    await _delete_cache_access_object(access_group_id=ag_id, user_api_key_cache=cache)
    cached = await cache.async_get_cache(key=f"access_group_id:{ag_id}")
    assert cached is None


# ---------------------------------------------------------------------------
# Access group resource fetchers (_get_models_from_access_groups, _get_agent_ids_from_access_groups)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resource_field, access_group_data, expected",
    [
        (
            "access_model_names",
            {"access_group_id": "ag-1", "access_model_names": ["gpt-4", "claude-3"]},
            ["gpt-4", "claude-3"],
        ),
        (
            "access_agent_ids",
            {"access_group_id": "ag-2", "access_agent_ids": ["agent-a", "agent-b"]},
            ["agent-a", "agent-b"],
        ),
        (
            "access_model_names",
            {"access_group_id": "ag-3", "access_model_names": []},
            [],
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_resources_from_access_groups(resource_field, access_group_data, expected):
    """Test _get_resources_from_access_groups returns correct resource list from access groups."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy._types import LiteLLM_AccessGroupTable
    from litellm.proxy.auth.auth_checks import (
        _get_agent_ids_from_access_groups,
        _get_models_from_access_groups,
    )

    ag_table = LiteLLM_AccessGroupTable(
        access_group_id=access_group_data["access_group_id"],
        access_group_name="test",
        access_model_names=access_group_data.get("access_model_names", []),
        access_agent_ids=access_group_data.get("access_agent_ids", []),
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_access_object",
        new_callable=AsyncMock,
        return_value=ag_table,
    ):
        if resource_field == "access_model_names":
            result = await _get_models_from_access_groups(
                access_group_ids=[access_group_data["access_group_id"]],
                prisma_client=MagicMock(),
                user_api_key_cache=DualCache(),
            )
        else:
            result = await _get_agent_ids_from_access_groups(
                access_group_ids=[access_group_data["access_group_id"]],
                prisma_client=MagicMock(),
                user_api_key_cache=DualCache(),
            )
        assert sorted(result) == sorted(expected)


@pytest.mark.asyncio
async def test_get_models_from_access_groups_empty_ids():
    """Test _get_models_from_access_groups returns empty list when access_group_ids is empty."""
    from litellm.proxy.auth.auth_checks import _get_models_from_access_groups

    result = await _get_models_from_access_groups(access_group_ids=[])
    assert result == []


# ---------------------------------------------------------------------------
# can_team_access_model with access_group_ids fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_can_team_access_model_via_access_group_ids():
    """Test can_team_access_model allows access when team has access_group_ids granting model access."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import can_team_access_model

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        models=[],
        access_group_ids=["ag-with-gpt4"],
    )

    with patch(
        "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
        new_callable=AsyncMock,
        return_value=["gpt-4"],
    ):
        result = await can_team_access_model(
            model="gpt-4",
            team_object=team_object,
            llm_router=None,
            team_model_aliases=None,
        )
        assert result is True


@pytest.mark.asyncio
async def test_can_team_access_model_access_group_ids_denied():
    """Test can_team_access_model denies when neither team models nor access_group_ids grant access."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import can_team_access_model
    from litellm.proxy._types import ProxyException

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-3.5-turbo"],
        access_group_ids=["ag-other"],
    )

    with patch(
        "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
        new_callable=AsyncMock,
        return_value=["claude-3"],
    ):
        with pytest.raises(ProxyException):
            await can_team_access_model(
                model="gpt-4",
                team_object=team_object,
                llm_router=None,
                team_model_aliases=None,
            )


# ---------------------------------------------------------------------------
# can_key_call_model with access_group_ids fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_can_key_call_model_via_access_group_ids():
    """Test can_key_call_model allows access when key has access_group_ids granting model access."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import can_key_call_model

    user_api_key_object = UserAPIKeyAuth(
        token="test-token",
        models=[],
        access_group_ids=["ag-with-gpt4"],
    )
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4", "api_key": "test"},
            }
        ]
    )

    with patch(
        "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
        new_callable=AsyncMock,
        return_value=["gpt-4"],
    ):
        await can_key_call_model(
            model="gpt-4",
            llm_model_list=[],
            valid_token=user_api_key_object,
            llm_router=router,
        )
