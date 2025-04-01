import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import (
    _get_enforced_params,
    check_if_token_is_service_account,
)

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


def test_check_if_token_is_service_account():
    """
    Test that only keys with `service_account_id` in metadata are considered service accounts
    """
    # Test case 1: Service account token
    service_account_token = UserAPIKeyAuth(
        api_key="test-key", metadata={"service_account_id": "test-service-account"}
    )
    assert check_if_token_is_service_account(service_account_token) == True

    # Test case 2: Regular user token
    regular_token = UserAPIKeyAuth(api_key="test-key", metadata={})
    assert check_if_token_is_service_account(regular_token) == False

    # Test case 3: Token with other metadata
    other_metadata_token = UserAPIKeyAuth(
        api_key="test-key", metadata={"user_id": "test-user"}
    )
    assert check_if_token_is_service_account(other_metadata_token) == False


def test_get_enforced_params_for_service_account_settings():
    """
    Test that service account enforced params are only added to service account keys
    """
    service_account_token = UserAPIKeyAuth(
        api_key="test-key", metadata={"service_account_id": "test-service-account"}
    )
    general_settings_with_service_account_settings = {
        "service_account_settings": {"enforced_params": ["metadata.service"]},
    }
    result = _get_enforced_params(
        general_settings=general_settings_with_service_account_settings,
        user_api_key_dict=service_account_token,
    )
    assert result == ["metadata.service"]

    regular_token = UserAPIKeyAuth(
        api_key="test-key", metadata={"enforced_params": ["user"]}
    )
    result = _get_enforced_params(
        general_settings=general_settings_with_service_account_settings,
        user_api_key_dict=regular_token,
    )
    assert result == ["user"]


@pytest.mark.parametrize(
    "general_settings, user_api_key_dict, expected_enforced_params",
    [
        (
            {"enforced_params": ["param1", "param2"]},
            UserAPIKeyAuth(
                api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
            ),
            ["param1", "param2"],
        ),
        (
            {"service_account_settings": {"enforced_params": ["param1", "param2"]}},
            UserAPIKeyAuth(
                api_key="test_api_key",
                user_id="test_user_id",
                org_id="test_org_id",
                metadata={"service_account_id": "test_service_account_id"},
            ),
            ["param1", "param2"],
        ),
        (
            {"service_account_settings": {"enforced_params": ["param1", "param2"]}},
            UserAPIKeyAuth(
                api_key="test_api_key",
                metadata={
                    "enforced_params": ["param3", "param4"],
                    "service_account_id": "test_service_account_id",
                },
            ),
            ["param1", "param2", "param3", "param4"],
        ),
    ],
)
def test_get_enforced_params(
    general_settings, user_api_key_dict, expected_enforced_params
):
    from litellm.proxy.litellm_pre_call_utils import _get_enforced_params

    enforced_params = _get_enforced_params(general_settings, user_api_key_dict)
    assert enforced_params == expected_enforced_params
