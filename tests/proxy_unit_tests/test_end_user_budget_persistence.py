import sys
import os
import uuid
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure litellm is in path
sys.path.insert(0, os.path.abspath("../.."))


# Define a real class for type checks
class MockPrismaClient:
    pass


# Mock litellm.proxy.proxy_server BEFORE any imports that might touch it

mock_proxy_server = MagicMock()
mock_proxy_server.prisma_client = MagicMock()
mock_proxy_server.PrismaClient = MockPrismaClient
mock_proxy_server.llm_router = MagicMock()
mock_proxy_server.litellm_proxy_admin_name = "admin"

# Mock proxy_logging_obj to avoid coroutine errors
mock_proxy_logging_obj = MagicMock()
mock_proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
mock_proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()
mock_proxy_server.proxy_logging_obj = mock_proxy_logging_obj

sys.modules["litellm.proxy.proxy_server"] = mock_proxy_server

# Also mock litellm.proxy.utils.PrismaClient
import litellm.proxy.utils as proxy_utils

proxy_utils.PrismaClient = MockPrismaClient

# Mock prisma module hierarchy using ModuleType to satisfy 'is a package' checks
import types

mock_prisma = types.ModuleType("prisma")
mock_prisma.__path__ = []
mock_prisma.errors = types.ModuleType("prisma.errors")
mock_prisma.errors.PrismaError = type("PrismaError", (Exception,), {})
mock_prisma.models = types.ModuleType("prisma.models")
mock_prisma.client = types.ModuleType("prisma.client")

sys.modules["prisma"] = mock_prisma
sys.modules["prisma.errors"] = mock_prisma.errors
sys.modules["prisma.models"] = mock_prisma.models
sys.modules["prisma.client"] = mock_prisma.client

import litellm
from litellm.proxy.auth.auth_checks import get_end_user_object
from litellm.proxy.utils import ProxyUpdateSpend
from litellm.proxy.management_endpoints.customer_endpoints import new_end_user

# from litellm.proxy.common_utils.http_parsing_utils import _safe_get_request_headers
# from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy._types import UserAPIKeyAuth, NewCustomerRequest
from litellm.caching import DualCache


def mock_record(data):
    """
    Helper to create a mock record that supports .dict() and .model_dump()
    """
    record = MagicMock()
    record.dict = lambda: data
    record.model_dump = lambda **kwargs: data
    # Add attributes for direct access if needed
    for k, v in data.items():
        setattr(record, k, v)
    return record


@pytest.mark.asyncio
async def test_apply_default_budget_persistence():
    """
    Test that _apply_default_budget_to_end_user persists the budget_id to the database.
    """
    end_user_id = f"test_user_{uuid.uuid4().hex}"
    default_budget_id = "default_budget_123"
    litellm.max_end_user_budget_id = default_budget_id

    # Mock data
    mock_end_user_data = {
        "user_id": end_user_id,
        "spend": 0.0,
        "litellm_budget_table": None,
        "blocked": False,
    }

    default_budget = {
        "budget_id": default_budget_id,
        "max_budget": 0.02,
    }

    # Mock find_unique for end user
    mock_proxy_server.prisma_client.db.litellm_endusertable.find_unique = AsyncMock(
        return_value=mock_record(mock_end_user_data)
    )
    # Mock find_unique for budget
    mock_proxy_server.prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=mock_record(default_budget)
    )

    # Mock update to check if it's called
    mock_update = AsyncMock()
    mock_proxy_server.prisma_client.db.litellm_endusertable.update = mock_update

    mock_cache = AsyncMock(spec=DualCache)
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.async_set_cache = AsyncMock()

    # Call get_end_user_object which calls _apply_default_budget_to_end_user
    with patch("litellm.proxy.auth.auth_checks.PrismaClient", MockPrismaClient):
        await get_end_user_object(
            end_user_id=end_user_id,
            prisma_client=mock_proxy_server.prisma_client,
            user_api_key_cache=mock_cache,
            route="/chat/completions",
        )

    # Allow some time for the background task to run
    await asyncio.sleep(0.1)

    # Verify that prisma_client.db.litellm_endusertable.update was called with the correct budget_id
    mock_proxy_server.prisma_client.db.litellm_endusertable.update.assert_called_with(
        where={"user_id": end_user_id}, data={"budget_id": default_budget_id}
    )

    litellm.max_end_user_budget_id = None


@pytest.mark.asyncio
async def test_update_end_user_spend_persistence():
    """
    Test that ProxyUpdateSpend.update_end_user_spend includes budget_id when creating a new user.
    """
    end_user_id = f"test_user_{uuid.uuid4().hex}"
    default_budget_id = "default_budget_123"
    litellm.max_end_user_budget_id = default_budget_id

    # Mock for prisma_client.db.tx() context manager
    mock_tx_context = MagicMock()
    mock_transaction = MagicMock()
    mock_tx_context.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_tx_context.__aexit__ = AsyncMock(return_value=None)
    mock_proxy_server.prisma_client.db.tx.return_value = mock_tx_context

    # Mock for transaction.batch_() context manager
    mock_batch_context = MagicMock()
    mock_batch = MagicMock()
    mock_batch_context.__aenter__ = AsyncMock(return_value=mock_batch)
    mock_batch_context.__aexit__ = AsyncMock(return_value=None)
    mock_transaction.batch_.return_value = mock_batch_context

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.failure_handler = AsyncMock()

    # Call update_end_user_spend
    await ProxyUpdateSpend.update_end_user_spend(
        n_retry_times=0,
        prisma_client=mock_proxy_server.prisma_client,
        proxy_logging_obj=mock_proxy_logging_obj,
        end_user_list_transactions={end_user_id: 10.0},
    )

    # Verify that upsert was called with budget_id in the create block
    mock_batch.litellm_endusertable.upsert.assert_called()
    args, kwargs = mock_batch.litellm_endusertable.upsert.call_args

    assert kwargs["where"] == {"user_id": end_user_id}
    assert kwargs["data"]["create"]["budget_id"] == default_budget_id
    assert kwargs["data"]["create"]["user_id"] == end_user_id
    assert kwargs["data"]["update"]["spend"]["increment"] == 10.0

    litellm.max_end_user_budget_id = None


@pytest.mark.asyncio
async def test_manual_user_creation_persistence():
    """
    Test that new_end_user assigns the default budget_id when none is provided.
    """
    default_budget_id = "default_budget_manual"
    litellm.max_end_user_budget_id = default_budget_id

    mock_proxy_server.prisma_client.db.litellm_endusertable.create = AsyncMock(
        return_value=mock_record(
            {
                "user_id": "manual_user_123",
                "budget_id": default_budget_id,
                "spend": 0.0,
                "blocked": False,
            }
        )
    )

    data = NewCustomerRequest(user_id="manual_user_123")
    user_api_key_dict = UserAPIKeyAuth(user_id="admin")

    # Mock _set_object_permission
    with patch(
        "litellm.proxy.management_endpoints.customer_endpoints._set_object_permission",
        side_effect=lambda data_json, prisma_client: data_json,
    ):
        await new_end_user(
            data=data,
            user_api_key_dict=user_api_key_dict,
        )

    # Verify budget_id was passed to create
    mock_proxy_server.prisma_client.db.litellm_endusertable.create.assert_called()
    (
        args,
        kwargs,
    ) = mock_proxy_server.prisma_client.db.litellm_endusertable.create.call_args
    assert kwargs["data"]["budget_id"] == default_budget_id

    litellm.max_end_user_budget_id = None


@pytest.mark.asyncio
async def test_get_end_user_object_cache_sync():
    """
    Test that get_end_user_object syncs the cache when a budget is newly applied.
    (This verifies the fix from PR #22501 logic)
    """
    end_user_id = "cache_sync_user"
    default_budget_id = "sync_budget"
    litellm.max_end_user_budget_id = default_budget_id

    # User is in cache but HAS NO BUDGET
    cached_user = {
        "user_id": end_user_id,
        "spend": 5.0,
        "litellm_budget_table": None,
        "blocked": False,
    }

    mock_cache = AsyncMock(spec=DualCache)

    async def mock_async_get_cache(key, **kwargs):
        if "end_user_id" in key:
            return cached_user
        return None  # Return None for budget cache to trigger DB lookup

    mock_cache.async_get_cache = AsyncMock(side_effect=mock_async_get_cache)
    mock_cache.async_set_cache = AsyncMock()

    mock_prisma_client = MagicMock()
    # Mock find_unique for budget
    mock_prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=mock_record({"budget_id": default_budget_id, "max_budget": 100})
    )
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock()

    with patch("litellm.proxy.auth.auth_checks.PrismaClient", MockPrismaClient):
        await get_end_user_object(
            end_user_id=end_user_id,
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_cache,
            route="/chat/completions",
        )

    # Verify cache was updated with the new budget
    mock_cache.async_set_cache.assert_called()
    args, kwargs = mock_cache.async_set_cache.call_args
    assert kwargs["value"]["budget_id"] == default_budget_id

    litellm.max_end_user_budget_id = None


# def test_robustness_header_parsing():
#     """
#     Test that _safe_get_request_headers handles non-dict cached headers.
#     """
#     mock_request = MagicMock()
#     mock_request.state = MagicMock()
#     # Simulate a corrupted or unexpected state
#     mock_request.state._cached_headers = "not_a_dict"
#     mock_request.headers = {"test": "header"}

#     headers = _safe_get_request_headers(mock_request)
#     assert isinstance(headers, dict)
#     assert headers["test"] == "header"


# @pytest.mark.asyncio
# async def test_robustness_team_config_parsing():
#     """
#     Test that add_team_based_callbacks_from_config handles bad team_config.
#     """
#     mock_proxy_config = MagicMock()
#     # load_team_config returns None or something non-dict
#     mock_proxy_config.load_team_config.return_value = None

#     result = LiteLLMProxyRequestSetup.add_team_based_callbacks_from_config(
#         team_id="team_123", proxy_config=mock_proxy_config
#     )
#     assert result is None
