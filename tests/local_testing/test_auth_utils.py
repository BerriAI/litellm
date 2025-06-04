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

    request_body = {"user": 123}
    end_user_id = get_end_user_id_from_request_body(request_body)
    assert end_user_id == "123"
    assert isinstance(end_user_id, str)

    result = get_end_user_id_from_request_body(request_body)
    assert result is None


def test_get_end_user_id_from_request_body_or_headers():
    """Test retrieving end_user_id from request body or headers."""
    from litellm.proxy.auth.auth_utils import (
        get_end_user_id_from_request_body_or_headers,
    )

    # Test case 1: User ID in request body (should take precedence)
    request_body_1 = {"user": "body_user_1", "model": "gpt-3.5-turbo"}
    headers_1 = {"X-Test-User-Id": "header_user_1"}
    general_settings_1 = {"user_header_name": "X-Test-User-Id"}
    result_1 = get_end_user_id_from_request_body_or_headers(
        request_body_1, headers_1, general_settings_1
    )
    assert result_1 == "body_user_1"

    # Test case 2: User ID in headers (body has no user)
    request_body_2 = {"model": "gpt-3.5-turbo"}
    headers_2 = {"X-Test-User-Id": "header_user_2"}
    general_settings_2 = {"user_header_name": "X-Test-User-Id"}
    result_2 = get_end_user_id_from_request_body_or_headers(
        request_body_2, headers_2, general_settings_2
    )
    assert result_2 == "header_user_2"

    # Test case 3: User ID in headers (case-insensitive header name)
    request_body_3 = {"model": "gpt-3.5-turbo"}
    headers_3 = {"x-test-user-id": "header_user_3_lower"}
    general_settings_3 = {"user_header_name": "X-Test-User-Id"} # Configured with different case
    result_3 = get_end_user_id_from_request_body_or_headers(
        request_body_3, headers_3, general_settings_3
    )
    assert result_3 == "header_user_3_lower"

    # Test case 4: No user_header_name configured, user in header should be ignored
    request_body_4 = {"model": "gpt-3.5-turbo"}
    headers_4 = {"X-Test-User-Id": "header_user_4"}
    general_settings_4 = {} # No user_header_name
    result_4 = get_end_user_id_from_request_body_or_headers(
        request_body_4, headers_4, general_settings_4
    )
    assert result_4 is None

    # Test case 5: user_header_name configured, but header not present
    request_body_5 = {"model": "gpt-3.5-turbo"}
    headers_5 = {"Another-Header": "some_value"}
    general_settings_5 = {"user_header_name": "X-Test-User-Id"}
    result_5 = get_end_user_id_from_request_body_or_headers(
        request_body_5, headers_5, general_settings_5
    )
    assert result_5 is None

    # Test case 6: No user info in body or headers
    request_body_6 = {"model": "gpt-3.5-turbo"}
    headers_6 = {}
    general_settings_6 = {"user_header_name": "X-Test-User-Id"}
    result_6 = get_end_user_id_from_request_body_or_headers(
        request_body_6, headers_6, general_settings_6
    )
    assert result_6 is None

    # Test case 7: general_settings is None
    request_body_7 = {"model": "gpt-3.5-turbo"}
    headers_7 = {"X-Test-User-Id": "header_user_7"}
    result_7 = get_end_user_id_from_request_body_or_headers(
        request_body_7, headers_7, None
    )
    assert result_7 is None

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

