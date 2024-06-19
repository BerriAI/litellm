import asyncio
from unittest.mock import Mock

import pytest
from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.types.utils import SupportedCacheControls


@pytest.fixture
def mock_request(monkeypatch):
    mock_request = Mock(spec=Request)
    mock_request.query_params = {}  # Set mock query_params to an empty dictionary
    mock_request.headers = {}
    monkeypatch.setattr(
        "litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request", mock_request
    )
    return mock_request


@pytest.mark.parametrize("endpoint", ["/v1/threads", "/v1/thread/123"])
@pytest.mark.asyncio
async def test_add_litellm_data_to_request_thread_endpoint(endpoint, mock_request):
    mock_request.url.path = endpoint
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
    )
    proxy_config = Mock()

    data = {}
    await add_litellm_data_to_request(
        data, mock_request, user_api_key_dict, proxy_config
    )

    print("DATA: ", data)

    assert "litellm_metadata" in data
    assert "metadata" not in data


@pytest.mark.parametrize(
    "endpoint", ["/chat/completions", "/v1/completions", "/completions"]
)
@pytest.mark.asyncio
async def test_add_litellm_data_to_request_non_thread_endpoint(endpoint, mock_request):
    mock_request.url.path = endpoint
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
    )
    proxy_config = Mock()

    data = {}
    await add_litellm_data_to_request(
        data, mock_request, user_api_key_dict, proxy_config
    )

    print("DATA: ", data)

    assert "metadata" in data
    assert "litellm_metadata" not in data
