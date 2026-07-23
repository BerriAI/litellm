import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.auth.auth_checks import common_checks
from litellm.proxy._types import UserAPIKeyAuth, LiteLLM_UserTable
from fastapi import Request


@pytest.mark.asyncio
async def test_zero_budget_model_discovery_bypasses_budget_checks():
    user_obj = LiteLLM_UserTable(
        user_id="test_internal_user",
        max_budget=0.0,
        spend=10.0,
    )

    valid_token = UserAPIKeyAuth(api_key="sk-1234", user_id="test_internal_user", user_role="internal_user")

    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request.query_params = {}

    proxy_logging_obj = MagicMock()

    # If skip_budget_checks=False, but route is /v1/models:
    # Actually, common_checks receives skip_budget_checks=False by default,
    # but the router/endpoint has logic that sets skip_budget_checks = True.
    # Let's see what happens if skip_budget_checks = True
    await common_checks(
        request_body={},
        team_object=None,
        user_object=user_obj,
        end_user_object=None,
        global_proxy_spend=None,
        general_settings={},
        route="/v1/models",
        llm_router=None,
        proxy_logging_obj=proxy_logging_obj,
        valid_token=valid_token,
        request=mock_request,
        skip_budget_checks=True,
    )
