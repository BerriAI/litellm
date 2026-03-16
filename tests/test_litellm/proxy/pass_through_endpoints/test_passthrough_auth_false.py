"""
Test that pass-through endpoints with auth: false work without Authorization header.

Regression test for the bug where auth: false pass-through routes still required
an Authorization header and failed with 'NoneType' object has no attribute 'split'
when no header was provided.
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder


@pytest.mark.asyncio
async def test_early_exit_for_pass_through_auth_false():
    """
    _user_api_key_auth_builder returns UserAPIKeyAuth() early for pass-through
    routes with auth: false, avoiding NoneType.split when api_key is None.
    """
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = "/v1/cuopt/request"

    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_request_route",
        return_value="/v1/cuopt/request",
    ), patch(
        "litellm.proxy.auth.user_api_key_auth.pre_db_read_auth_checks",
        new_callable=AsyncMock,
    ), patch(
        "litellm.proxy.proxy_server.general_settings",
        {
            "pass_through_endpoints": [
                {
                    "path": "/v1/cuopt/request",
                    "target": "https://httpbin.org/post",
                    "auth": False,
                    "headers": {"content-type": "application/json"},
                    "forward_headers": True,
                }
            ],
        },
    ):
        result = await _user_api_key_auth_builder(
            request=request,
            api_key="",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={},
        )

    assert isinstance(result, UserAPIKeyAuth)
