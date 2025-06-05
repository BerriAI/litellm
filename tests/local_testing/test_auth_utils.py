# What is this?
## Tests if proxy/auth/auth_utils.py works as expected

import sys, os, asyncio, time, random, uuid
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm.proxy.auth.auth_utils import (
    _allow_model_level_clientside_configurable_parameters,
)
from litellm.router import Router


@pytest.mark.parametrize(
    "allowed_param, input_value, should_return_true",
    [
        ("api_base", {"api_base": "http://dummy.com"}, True),
        (
            {"api_base": "https://api.openai.com/v1"},
            {"api_base": "https://api.openai.com/v1"},
            True,
        ),  # should return True
        (
            {"api_base": "https://api.openai.com/v1"},
            {"api_base": "https://api.anthropic.com/v1"},
            False,
        ),  # should return False
        (
            {"api_base": "^https://litellm.*direct\.fireworks\.ai/v1$"},
            {"api_base": "https://litellm-dev.direct.fireworks.ai/v1"},
            True,
        ),
        (
            {"api_base": "^https://litellm.*novice\.fireworks\.ai/v1$"},
            {"api_base": "https://litellm-dev.direct.fireworks.ai/v1"},
            False,
        ),
    ],
)
def test_configurable_clientside_parameters(
    allowed_param, input_value, should_return_true
):
    router = Router(
        model_list=[
            {
                "model_name": "dummy-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "dummy-key",
                    "configurable_clientside_auth_params": [allowed_param],
                },
            }
        ]
    )
    resp = _allow_model_level_clientside_configurable_parameters(
        model="dummy-model",
        param="api_base",
        request_body_value=input_value["api_base"],
        llm_router=router,
    )
    print(resp)
    assert resp == should_return_true


def test_get_end_user_id_from_request_body_always_returns_str():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body
    from fastapi import Request
    from unittest.mock import MagicMock

    # Create a mock Request object
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    
    request_body = {"user": 123}
    end_user_id = get_end_user_id_from_request_body(request_body, dict(mock_request.headers))
    assert end_user_id == "123"
    assert isinstance(end_user_id, str)


@pytest.mark.parametrize(
    "headers, general_settings_config, request_body, expected_user_id",
    [
        # Test 1: user_header_name configured and header present
        (
            {"X-User-ID": "header-user-123"},
            {"user_header_name": "X-User-ID"},
            {"user": "body-user-456"},
            "header-user-123"  # Header should take precedence
        ),
        # Test 2: user_header_name configured but header not present, fallback to body
        (
            {},
            {"user_header_name": "X-User-ID"},
            {"user": "body-user-456"},
            "body-user-456"  # Should fall back to body
        ),
        # Test 3: user_header_name not configured, should use body
        (
            {"X-User-ID": "header-user-123"},
            {},
            {"user": "body-user-456"},
            "body-user-456"  # Should ignore header when not configured
        ),
        # Test 4: user_header_name configured, header present, but no body user
        (
            {"X-Custom-User": "header-only-user"},
            {"user_header_name": "X-Custom-User"},
            {"model": "gpt-4"},
            "header-only-user"  # Should use header
        ),
        # Test 5: user_header_name configured but header is empty string
        (
            {"X-User-ID": ""},
            {"user_header_name": "X-User-ID"},
            {"user": "body-user-456"},
            "body-user-456"  # Should fall back to body when header is empty
        ),
        # Test 6: user_header_name configured with case-insensitive header
        (
            {"x-user-id": "lowercase-header-user"},
            {"user_header_name": "x-user-id"},
            {"user": "body-user-456"},
            "lowercase-header-user"
        ),
        # Test 7: user_header_name configured but set to None
        (
            {"X-User-ID": "header-user-123"},
            {"user_header_name": None},
            {"user": "body-user-456"},
            "body-user-456"  # Should fall back to body when header name is None
        ),
        # Test 8: user_header_name is not a string
        (
            {"X-User-ID": "header-user-123"},
            {"user_header_name": 123},
            {"user": "body-user-456"},
            "body-user-456"  # Should fall back to body when header name is not a string
        ),
        # Test 9: Multiple fallback sources - litellm_metadata
        (
            {},
            {"user_header_name": "X-User-ID"},
            {"litellm_metadata": {"user": "litellm-user-789"}},
            "litellm-user-789"
        ),
        # Test 10: Multiple fallback sources - metadata.user_id
        (
            {},
            {"user_header_name": "X-User-ID"},
            {"metadata": {"user_id": "metadata-user-999"}},
            "metadata-user-999"
        ),
        # Test 11: Header takes precedence over all body sources
        (
            {"X-User-ID": "header-priority"},
            {"user_header_name": "X-User-ID"},
            {
                "user": "body-user",
                "litellm_metadata": {"user": "litellm-user"},
                "metadata": {"user_id": "metadata-user"}
            },
            "header-priority"
        ),
    ]
)
def test_get_end_user_id_from_request_body_with_user_header_name(
    headers, general_settings_config, request_body, expected_user_id
):
    """Test that get_end_user_id_from_request_body respects user_header_name property"""
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body
    from fastapi import Request
    from unittest.mock import MagicMock, patch

    # Create a mock Request object with headers
    mock_request = MagicMock(spec=Request)
    mock_request.headers = headers
    
    # Mock general_settings at the proxy_server module level
    with patch('litellm.proxy.proxy_server.general_settings', general_settings_config):
        end_user_id = get_end_user_id_from_request_body(request_body, dict(mock_request.headers))
        assert end_user_id == expected_user_id


def test_get_end_user_id_from_request_body_no_user_found():
    """Test that function returns None when no user ID is found anywhere"""
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body
    from fastapi import Request
    from unittest.mock import MagicMock, patch

    # Create a mock Request object with no relevant headers
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"X-Other-Header": "some-value"}
    
    # Mock general_settings with user_header_name that doesn't match headers
    general_settings_config = {"user_header_name": "X-User-ID"}
    
    # Request body with no user identifiers
    request_body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
    
    with patch('litellm.proxy.proxy_server.general_settings', general_settings_config):
        end_user_id = get_end_user_id_from_request_body(request_body, dict(mock_request.headers))
        assert end_user_id is None


def test_get_end_user_id_from_request_body_backwards_compatibility():
    """Test that function works with just request_body parameter (backwards compatibility)"""
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    # Test with just request_body - should work like before
    request_body = {"user": "test-user-123"}
    end_user_id = get_end_user_id_from_request_body(request_body)
    assert end_user_id == "test-user-123"
    
    # Test with litellm_metadata
    request_body = {"litellm_metadata": {"user": "litellm-user-456"}}
    end_user_id = get_end_user_id_from_request_body(request_body)
    assert end_user_id == "litellm-user-456"
    
    # Test with metadata.user_id
    request_body = {"metadata": {"user_id": "metadata-user-789"}}
    end_user_id = get_end_user_id_from_request_body(request_body)
    assert end_user_id == "metadata-user-789"
    
    # Test with no user - should return None
    request_body = {"model": "gpt-4"}
    end_user_id = get_end_user_id_from_request_body(request_body)
    assert end_user_id is None

@pytest.mark.parametrize(
    "request_data, expected_model",
    [
        ({"target_model_names": "gpt-3.5-turbo, gpt-4o-mini-general-deployment"}, ["gpt-3.5-turbo", "gpt-4o-mini-general-deployment"]),
        ({"target_model_names": "gpt-3.5-turbo"}, ["gpt-3.5-turbo"]),
        ({"model": "gpt-3.5-turbo, gpt-4o-mini-general-deployment"}, ["gpt-3.5-turbo", "gpt-4o-mini-general-deployment"]),
        ({"model": "gpt-3.5-turbo"}, "gpt-3.5-turbo"),
        ({"model": "gpt-3.5-turbo, gpt-4o-mini-general-deployment"}, ["gpt-3.5-turbo", "gpt-4o-mini-general-deployment"]),
    ],
)
def test_get_model_from_request(request_data, expected_model):
    from litellm.proxy.auth.auth_utils import get_model_from_request

    request_data = {"target_model_names": "gpt-3.5-turbo, gpt-4o-mini-general-deployment"}
    route = "/openai/deployments/gpt-3.5-turbo"
    model = get_model_from_request(request_data, "/v1/files")
    assert model == ["gpt-3.5-turbo", "gpt-4o-mini-general-deployment"]

