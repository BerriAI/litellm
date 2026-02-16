import asyncio
import copy
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request

import litellm
from litellm.proxy._types import TeamCallbackMetadata, UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import (
    KeyAndTeamLoggingSettings,
    LiteLLMProxyRequestSetup,
    _get_dynamic_logging_metadata,
    _get_enforced_params,
    _update_model_if_key_alias_exists,
    add_guardrails_from_policy_engine,
    add_litellm_data_to_request,
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


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_parses_string_metadata():
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Simulate data with stringified metadata
    fake_metadata = {"generation_name": "gen123"}
    data = {"metadata": json.dumps(fake_metadata), "model": "gpt-3.5-turbo"}

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},  # this one can be a dict
        team_spend=0.0,
        team_max_budget=200.0,
    )

    # Call
    updated_data = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    # Assert
    litellm_metadata = updated_data.get("metadata", {})
    assert isinstance(litellm_metadata, dict)
    assert updated_data["metadata"]["generation_name"] == "gen123"


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_user_spend_and_budget():
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={},
        team_metadata={},
        user_spend=150.0,
        user_max_budget=500.0,
    )

    updated_data = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    metadata = updated_data.get("metadata", {})
    assert metadata["user_api_key_user_spend"] == 150.0
    assert metadata["user_api_key_user_max_budget"] == 500.0


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_audio_transcription_multipart():
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup request mock for /v1/audio/transcriptions
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/audio/transcriptions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/audio/transcriptions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {
        "Content-Type": "multipart/form-data",
        "Authorization": "Bearer sk-1234",
    }
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Simulate multipart data (metadata as string)
    metadata_dict = {
        "tags": ["jobID:214590dsff09fds", "taskName:run_page_classification"]
    }
    stringified_metadata = json.dumps(metadata_dict)

    data = {
        "model": "fake-openai-endpoint",
        "metadata": stringified_metadata,  # Simulating multipart-form field
        "file": b"Fake audio bytes",
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated_data = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    # Assert metadata was parsed correctly
    metadata_field = updated_data.get("metadata", {})
    litellm_metadata = updated_data.get("litellm_metadata", {})

    assert isinstance(metadata_field, dict)
    assert "tags" in metadata_field
    assert metadata_field["tags"] == [
        "jobID:214590dsff09fds",
        "taskName:run_page_classification",
    ]


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_disabled_callbacks():
    """
    Test that litellm_disabled_callbacks from key metadata is properly added to the request data.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup user API key with disabled callbacks in metadata
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata={"litellm_disabled_callbacks": ["langfuse", "langsmith", "datadog"]},
    )

    # Setup request data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Setup proxy config
    proxy_config = MagicMock()

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
    )

    # Verify that litellm_disabled_callbacks was added to the request data
    assert "litellm_disabled_callbacks" in result
    assert result["litellm_disabled_callbacks"] == ["langfuse", "langsmith", "datadog"]

    # Verify that other data is still present
    assert "model" in result
    assert result["model"] == "gpt-3.5-turbo"
    assert "messages" in result


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_disabled_callbacks_empty():
    """
    Test that litellm_disabled_callbacks is not added when it's empty.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup user API key with empty disabled callbacks
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata={"litellm_disabled_callbacks": []},
    )

    # Setup request data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Setup proxy config
    proxy_config = MagicMock()

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
    )

    # Verify that litellm_disabled_callbacks is not added when empty
    assert "litellm_disabled_callbacks" not in result

    # Verify that other data is still present
    assert "model" in result
    assert result["model"] == "gpt-3.5-turbo"
    assert "messages" in result


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_disabled_callbacks_not_present():
    """
    Test that litellm_disabled_callbacks is not added when it's not present in metadata.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup user API key without disabled callbacks in metadata
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata={},  # No litellm_disabled_callbacks
    )

    # Setup request data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Setup proxy config
    proxy_config = MagicMock()

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
    )

    # Verify that litellm_disabled_callbacks is not added when not present
    assert "litellm_disabled_callbacks" not in result

    # Verify that other data is still present
    assert "model" in result
    assert result["model"] == "gpt-3.5-turbo"
    assert "messages" in result


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_disabled_callbacks_invalid_type():
    """
    Test that litellm_disabled_callbacks is not added when it's not a list.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup user API key with invalid disabled callbacks type
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata={"litellm_disabled_callbacks": "not_a_list"},  # Should be a list
    )

    # Setup request data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Setup proxy config
    proxy_config = MagicMock()

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
    )

    # Verify that litellm_disabled_callbacks is not added when invalid type
    assert "litellm_disabled_callbacks" not in result

    # Verify that other data is still present
    assert "model" in result
    assert result["model"] == "gpt-3.5-turbo"
    assert "messages" in result


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_disabled_callbacks_with_logging_settings():
    """
    Test that litellm_disabled_callbacks works correctly alongside logging settings.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup user API key with both logging settings and disabled callbacks
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata={
            "logging": [
                {
                    "callback_name": "langfuse",
                    "callback_type": "success",
                    "callback_vars": {},
                }
            ],
            "litellm_disabled_callbacks": ["langsmith", "datadog"],
        },
    )

    # Setup request data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Setup proxy config
    proxy_config = MagicMock()

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
    )

    # Verify that both logging settings and disabled callbacks are handled correctly
    assert "litellm_disabled_callbacks" in result
    assert result["litellm_disabled_callbacks"] == ["langsmith", "datadog"]

    # Verify that other data is still present
    assert "model" in result
    assert result["model"] == "gpt-3.5-turbo"
    assert "messages" in result


def test_key_dynamic_logging_settings():
    """
    Test KeyAndTeamLoggingSettings.get_key_dynamic_logging_settings method with arize and langfuse callbacks
    """
    # Test with arize logging
    key_with_arize = UserAPIKeyAuth(
        api_key="test-key",
        metadata={"logging": [{"callback_name": "arize", "callback_type": "success"}]},
        team_metadata={},
    )
    result = KeyAndTeamLoggingSettings.get_key_dynamic_logging_settings(key_with_arize)
    assert result == [{"callback_name": "arize", "callback_type": "success"}]

    # Test with langfuse logging
    key_with_langfuse = UserAPIKeyAuth(
        api_key="test-key",
        metadata={
            "logging": [{"callback_name": "langfuse", "callback_type": "success"}]
        },
        team_metadata={},
    )
    result = KeyAndTeamLoggingSettings.get_key_dynamic_logging_settings(
        key_with_langfuse
    )
    assert result == [{"callback_name": "langfuse", "callback_type": "success"}]

    # Test with no logging metadata
    key_without_logging = UserAPIKeyAuth(
        api_key="test-key", metadata={}, team_metadata={}
    )
    result = KeyAndTeamLoggingSettings.get_key_dynamic_logging_settings(
        key_without_logging
    )
    assert result is None


def test_team_dynamic_logging_settings():
    """
    Test KeyAndTeamLoggingSettings.get_team_dynamic_logging_settings method with arize and langfuse callbacks
    """
    # Test with arize team logging
    key_with_team_arize = UserAPIKeyAuth(
        api_key="test-key",
        metadata={},
        team_metadata={
            "logging": [{"callback_name": "arize", "callback_type": "failure"}]
        },
    )
    result = KeyAndTeamLoggingSettings.get_team_dynamic_logging_settings(
        key_with_team_arize
    )
    assert result == [{"callback_name": "arize", "callback_type": "failure"}]

    # Test with langfuse team logging
    key_with_team_langfuse = UserAPIKeyAuth(
        api_key="test-key",
        metadata={},
        team_metadata={
            "logging": [{"callback_name": "langfuse", "callback_type": "success"}]
        },
    )
    result = KeyAndTeamLoggingSettings.get_team_dynamic_logging_settings(
        key_with_team_langfuse
    )
    assert result == [{"callback_name": "langfuse", "callback_type": "success"}]

    # Test with no team logging metadata
    key_without_team_logging = UserAPIKeyAuth(
        api_key="test-key", metadata={}, team_metadata={}
    )
    result = KeyAndTeamLoggingSettings.get_team_dynamic_logging_settings(
        key_without_team_logging
    )
    assert result is None


def test_get_dynamic_logging_metadata_with_arize_team_logging():
    """
    Test _get_dynamic_logging_metadata function with arize team logging and dynamic parameters
    """
    # Setup user with arize team logging including callback_vars
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        metadata={},
        team_metadata={
            "logging": [
                {
                    "callback_name": "arize",
                    "callback_type": "success",
                    "callback_vars": {
                        "arize_api_key": "test_arize_api_key",
                        "arize_space_id": "test_arize_space_id",
                    },
                }
            ]
        },
    )

    # Mock proxy_config (not used in this test path since we have team dynamic logging)
    mock_proxy_config = MagicMock()

    # Call the function
    result = _get_dynamic_logging_metadata(
        user_api_key_dict=user_api_key_dict, proxy_config=mock_proxy_config
    )

    # Verify the result
    assert result is not None
    assert isinstance(result, TeamCallbackMetadata)
    assert result.success_callback == ["arize"]
    assert result.callback_vars is not None
    assert result.callback_vars["arize_api_key"] == "test_arize_api_key"
    assert result.callback_vars["arize_space_id"] == "test_arize_space_id"


def test_get_num_retries_from_request():
    """
    Test LiteLLMProxyRequestSetup._get_num_retries_from_request method
    """
    # Test case 1: Header is present with valid integer string
    headers_with_retries = {"x-litellm-num-retries": "3"}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(
        headers_with_retries
    )
    assert result == 3

    # Test case 2: Header is not present
    headers_without_retries = {"Content-Type": "application/json"}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(
        headers_without_retries
    )
    assert result is None

    # Test case 3: Empty headers dictionary
    empty_headers = {}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(empty_headers)
    assert result is None

    # Test case 4: Header present with zero value
    headers_with_zero = {"x-litellm-num-retries": "0"}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(headers_with_zero)
    assert result == 0

    # Test case 5: Header present with large number
    headers_with_large_number = {"x-litellm-num-retries": "100"}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(
        headers_with_large_number
    )
    assert result == 100

    # Test case 6: Multiple headers with num retries header
    headers_multiple = {
        "Content-Type": "application/json",
        "x-litellm-num-retries": "5",
        "Authorization": "Bearer token",
    }
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(headers_multiple)
    assert result == 5

    # Test case 7: Header present with invalid value (should raise ValueError when int() is called)
    headers_with_invalid = {"x-litellm-num-retries": "invalid"}
    with pytest.raises(ValueError):
        LiteLLMProxyRequestSetup._get_num_retries_from_request(headers_with_invalid)

    # Test case 8: Header present with float string (should raise ValueError when int() is called)
    headers_with_float = {"x-litellm-num-retries": "3.5"}
    with pytest.raises(ValueError):
        LiteLLMProxyRequestSetup._get_num_retries_from_request(headers_with_float)

    # Test case 9: Header present with negative number
    headers_with_negative = {"x-litellm-num-retries": "-1"}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(
        headers_with_negative
    )
    assert result == -1


def test_add_user_api_key_auth_to_request_metadata():
    """
    Test that add_user_api_key_auth_to_request_metadata properly adds user API key authentication data to request metadata
    """
    # Setup test data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "litellm_metadata": {},  # This will be the metadata variable name
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-test-key-123",
        user_id="test-user-123",
        org_id="test-org-456",
        team_id="test-team-789",
        key_alias="test-key-alias",
        user_email="test@example.com",
        team_alias="test-team-alias",
        end_user_id="test-end-user-123",
        request_route="/chat/completions",
        end_user_max_budget=500.0,
    )

    metadata_variable_name = "litellm_metadata"

    # Call the function
    result = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data=data,
        user_api_key_dict=user_api_key_dict,
        _metadata_variable_name=metadata_variable_name,
    )

    # Verify the metadata was properly added
    metadata = result[metadata_variable_name]

    # Check that user API key information was added
    assert metadata["user_api_key_hash"] == "hashed-test-key-123"
    assert metadata["user_api_key_alias"] == "test-key-alias"
    assert metadata["user_api_key_team_id"] == "test-team-789"
    assert metadata["user_api_key_user_id"] == "test-user-123"
    assert metadata["user_api_key_org_id"] == "test-org-456"
    assert metadata["user_api_key_team_alias"] == "test-team-alias"
    assert metadata["user_api_key_end_user_id"] == "test-end-user-123"
    assert metadata["user_api_key_user_email"] == "test@example.com"
    assert metadata["user_api_key_request_route"] == "/chat/completions"

    # Check that the hashed API key was added
    assert metadata["user_api_key"] == "hashed-test-key-123"

    # Check that end user max budget was added
    assert metadata["user_api_end_user_max_budget"] == 500.0

    # Verify original data is preserved
    assert result["model"] == "gpt-3.5-turbo"
    assert result["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.parametrize(
    "data, model_group_settings, expected_headers_added",
    [
        # Test case 1: Model is in forward_client_headers_to_llm_api list
        (
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            MagicMock(forward_client_headers_to_llm_api=["gpt-4"]),
            True,
        ),
        # Test case 2: Model is not in forward_client_headers_to_llm_api list
        (
            {"model": "claude-3", "messages": [{"role": "user", "content": "Hello"}]},
            MagicMock(forward_client_headers_to_llm_api=["gpt-4"]),
            False,
        ),
        # Test case 3: Model group settings is None
        (
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            None,
            False,
        ),
        # Test case 4: forward_client_headers_to_llm_api is None
        (
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            MagicMock(forward_client_headers_to_llm_api=None),
            False,
        ),
        # Test case 5: Data has no model
        (
            {"messages": [{"role": "user", "content": "Hello"}]},
            MagicMock(forward_client_headers_to_llm_api=["gpt-4"]),
            False,
        ),
        # Test case 6: Model is None
        (
            {"model": None, "messages": [{"role": "user", "content": "Hello"}]},
            MagicMock(forward_client_headers_to_llm_api=["gpt-4"]),
            False,
        ),
    ],
)
def test_add_headers_to_llm_call_by_model_group(
    data, model_group_settings, expected_headers_added
):
    """
    Test LiteLLMProxyRequestSetup.add_headers_to_llm_call_by_model_group method

    This tests various scenarios:
    1. When model is in the forward_client_headers_to_llm_api list
    2. When model is not in the list
    3. When model_group_settings is None
    4. When forward_client_headers_to_llm_api is None
    5. When data has no model
    6. When model is None
    """
    import litellm

    # Setup test headers and user API key
    headers = {
        "Authorization": "Bearer token123",
        "User-Agent": "test-client/1.0",
        "X-Custom-Header": "custom-value",
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key", user_id="test-user", org_id="test-org"
    )

    # Mock the model_group_settings
    original_model_group_settings = getattr(litellm, "model_group_settings", None)
    litellm.model_group_settings = model_group_settings

    try:
        # Mock the add_headers_to_llm_call method to return expected headers
        expected_returned_headers = {
            "X-LiteLLM-User": "test-user",
            "X-LiteLLM-Org": "test-org",
        }

        with patch.object(
            LiteLLMProxyRequestSetup,
            "add_headers_to_llm_call",
            return_value=expected_returned_headers if expected_headers_added else {},
        ) as mock_add_headers:

            # Make a copy of original data to verify it's not mutated unexpectedly
            original_data = copy.deepcopy(data)

            # Call the method under test
            result = LiteLLMProxyRequestSetup.add_headers_to_llm_call_by_model_group(
                data=data, headers=headers, user_api_key_dict=user_api_key_dict
            )

            # Verify the result
            assert result is not None
            assert isinstance(result, dict)

            if expected_headers_added:
                # Verify that add_headers_to_llm_call was called
                mock_add_headers.assert_called_once_with(headers, user_api_key_dict)
                # Verify that headers were added to the data
                assert "headers" in result
                assert result["headers"] == expected_returned_headers
            else:
                # Verify that add_headers_to_llm_call was not called
                mock_add_headers.assert_not_called()
                # Verify that no headers were added
                assert "headers" not in result or result.get("headers") is None

            # Verify that original data fields are preserved
            for key, value in original_data.items():
                if key != "headers":  # headers might be added
                    assert result[key] == value

    finally:
        # Restore original model_group_settings
        litellm.model_group_settings = original_model_group_settings


def test_add_headers_to_llm_call_by_model_group_empty_headers_returned():
    """
    Test that when add_headers_to_llm_call returns empty dict, no headers are added to data
    """
    import litellm

    # Setup test data
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
    headers = {"Authorization": "Bearer token123"}
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

    # Mock model_group_settings with model in the list
    mock_settings = MagicMock(forward_client_headers_to_llm_api=["gpt-4"])
    original_model_group_settings = getattr(litellm, "model_group_settings", None)
    litellm.model_group_settings = mock_settings

    try:
        with patch.object(
            LiteLLMProxyRequestSetup,
            "add_headers_to_llm_call",
            return_value={},  # Return empty dict
        ) as mock_add_headers:

            result = LiteLLMProxyRequestSetup.add_headers_to_llm_call_by_model_group(
                data=data, headers=headers, user_api_key_dict=user_api_key_dict
            )

            # Verify that add_headers_to_llm_call was called
            mock_add_headers.assert_called_once_with(headers, user_api_key_dict)

            # Verify that no headers were added since returned headers were empty
            assert "headers" not in result

            # Verify original data is preserved
            assert result["model"] == "gpt-4"
            assert result["messages"] == [{"role": "user", "content": "Hello"}]

    finally:
        # Restore original model_group_settings
        litellm.model_group_settings = original_model_group_settings


def test_add_headers_to_llm_call_by_model_group_existing_headers_in_data():
    """
    Test that existing headers in data are overwritten when new headers are added
    """
    import litellm

    # Setup test data with existing headers
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "headers": {"Existing-Header": "existing-value"},
    }
    headers = {"Authorization": "Bearer token123"}
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

    # Mock model_group_settings with model in the list
    mock_settings = MagicMock(forward_client_headers_to_llm_api=["gpt-4"])
    original_model_group_settings = getattr(litellm, "model_group_settings", None)
    litellm.model_group_settings = mock_settings

    try:
        new_headers = {"X-LiteLLM-User": "test-user"}

        with patch.object(
            LiteLLMProxyRequestSetup,
            "add_headers_to_llm_call",
            return_value=new_headers,
        ) as mock_add_headers:

            result = LiteLLMProxyRequestSetup.add_headers_to_llm_call_by_model_group(
                data=data, headers=headers, user_api_key_dict=user_api_key_dict
            )

            # Verify that add_headers_to_llm_call was called
            mock_add_headers.assert_called_once_with(headers, user_api_key_dict)

            # Verify that headers were overwritten
            assert "headers" in result
            assert result["headers"] == new_headers
            assert result["headers"] != {"Existing-Header": "existing-value"}

            # Verify original data is preserved
            assert result["model"] == "gpt-4"
            assert result["messages"] == [{"role": "user", "content": "Hello"}]

    finally:
        # Restore original model_group_settings
        litellm.model_group_settings = original_model_group_settings

import json
import time
from typing import Optional
from unittest.mock import AsyncMock

from fastapi.responses import Response

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging
from litellm.types.utils import StandardLoggingPayload


class TestCustomLogger(CustomLogger):
    def __init__(self):
        self.standard_logging_object: Optional[StandardLoggingPayload] = None
        super().__init__()
        
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"SUCCESS CALLBACK CALLED! kwargs keys: {list(kwargs.keys())}")
        self.standard_logging_object = kwargs.get("standard_logging_object")
        print(f"Captured standard_logging_object: {self.standard_logging_object}")
        
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"FAILURE CALLBACK CALLED! kwargs keys: {list(kwargs.keys())}")

@pytest.mark.asyncio
async def test_add_litellm_metadata_from_request_headers():
    """
    Test that add_litellm_metadata_from_request_headers properly adds litellm metadata from request headers,
    makes an LLM request using base_process_llm_request, sleeps for 3 seconds, and checks standard_logging_payload has spend_logs_metadata from headers

    Relevant issue: https://github.com/BerriAI/litellm/issues/14008
    """
    # Set up test logger
    litellm._turn_on_debug()
    test_logger = TestCustomLogger()
    litellm.callbacks = [test_logger]

    # Prepare test data (ensure no streaming, add mock_response and api_key to route to litellm.acompletion)
    headers = {"x-litellm-spend-logs-metadata": '{"user_id": "12345", "project_id": "proj_abc", "request_type": "chat_completion", "timestamp": "2025-09-02T10:30:00Z"}'}
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": False, "mock_response": "Hi", "api_key": "fake-key"}
    
    # Create mock request with headers
    mock_request = MagicMock(spec=Request)
    mock_request.headers = headers
    mock_request.url.path = "/chat/completions"
    
    # Create mock response
    mock_fastapi_response = MagicMock(spec=Response)
    
    # Create mock user API key dict
    mock_user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        org_id="test-org"
    )
    
    # Create mock proxy logging object
    mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
    
    # Create async functions for the hooks
    async def mock_during_call_hook(*args, **kwargs):
        return None
        
    async def mock_pre_call_hook(*args, **kwargs):
        return data
        
    async def mock_post_call_success_hook(*args, **kwargs):
        # Return the response unchanged
        return kwargs.get('response', args[2] if len(args) > 2 else None)
        
    mock_proxy_logging_obj.during_call_hook = mock_during_call_hook
    mock_proxy_logging_obj.pre_call_hook = mock_pre_call_hook
    mock_proxy_logging_obj.post_call_success_hook = mock_post_call_success_hook
    
    # Create mock proxy config
    mock_proxy_config = MagicMock()
    
    # Create mock general settings
    general_settings = {}
    
    # Create mock select_data_generator with correct signature
    def mock_select_data_generator(response=None, user_api_key_dict=None, request_data=None):
        async def mock_generator():
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return mock_generator()
    
    # Create the processor
    processor = ProxyBaseLLMRequestProcessing(data=data)
    
    # Call base_process_llm_request (it will use the mock_response="Hi" parameter)
    result = await processor.base_process_llm_request(
        request=mock_request,
        fastapi_response=mock_fastapi_response,
        user_api_key_dict=mock_user_api_key_dict,
        route_type="acompletion",
        proxy_logging_obj=mock_proxy_logging_obj,
        general_settings=general_settings,
        proxy_config=mock_proxy_config,
        select_data_generator=mock_select_data_generator,
        llm_router=None,
        model="gpt-4",
        is_streaming_request=False
    )
    
    # Sleep for 3 seconds to allow logging to complete
    await asyncio.sleep(3)
    
    # Check if standard_logging_object was set
    assert test_logger.standard_logging_object is not None, "standard_logging_object should be populated after LLM request"
    
    # Verify the logging object contains expected metadata
    standard_logging_obj = test_logger.standard_logging_object

    print(f"Standard logging object captured: {json.dumps(standard_logging_obj, indent=4, default=str)}")

    SPEND_LOGS_METADATA = standard_logging_obj["metadata"]["spend_logs_metadata"]
    assert SPEND_LOGS_METADATA == dict(json.loads(headers["x-litellm-spend-logs-metadata"])), "spend_logs_metadata should be the same as the headers"

        

def test_get_internal_user_header_from_mapping_returns_expected_header():
    mappings = [
        {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"},
        {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"},
    ]

    header_name = LiteLLMProxyRequestSetup.get_internal_user_header_from_mapping(mappings)
    assert header_name == "X-OpenWebUI-User-Id"


def test_get_internal_user_header_from_mapping_none_when_absent():
    mappings = [
        {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"}
    ]
    header_name = LiteLLMProxyRequestSetup.get_internal_user_header_from_mapping(mappings)
    assert header_name is None

    single = {"header_name": "X-Only-Customer", "litellm_user_role": "customer"}
    header_name = LiteLLMProxyRequestSetup.get_internal_user_header_from_mapping(single)
    assert header_name is None


def test_add_internal_user_from_user_mapping_sets_user_id_when_header_present():
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
    headers = {"X-OpenWebUI-User-Id": "internal-user-123"}
    general_settings = {
        "user_header_mappings": [
            {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"},
            {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"},
        ]
    }

    result = LiteLLMProxyRequestSetup.add_internal_user_from_user_mapping(
        general_settings, user_api_key_dict, headers
    )

    assert result is user_api_key_dict
    assert user_api_key_dict.user_id == "internal-user-123"


def test_add_internal_user_from_user_mapping_no_header_or_mapping_returns_unchanged():
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

    result = LiteLLMProxyRequestSetup.add_internal_user_from_user_mapping(
        None, user_api_key_dict, {"X-OpenWebUI-User-Id": "abc"}
    )
    assert result is user_api_key_dict
    assert user_api_key_dict.user_id is None

    general_settings = {
        "user_header_mappings": [
            {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"}
        ]
    }
    result = LiteLLMProxyRequestSetup.add_internal_user_from_user_mapping(
        general_settings, user_api_key_dict, {"Other": "value"}
    )
    assert result is user_api_key_dict
    assert user_api_key_dict.user_id is None


def test_get_sanitized_user_information_from_key_includes_guardrails_metadata():
    """
    Test that get_sanitized_user_information_from_key includes guardrails field from key metadata in the returned payload
    """
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key-hash",
        key_alias="test-alias",
        user_id="test-user",
        metadata={"guardrails": ["presidio", "aporia"], "other_field": "value"},
    )

    result = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
        user_api_key_dict=user_api_key_dict
    )

    assert result["user_api_key_auth_metadata"] is not None
    assert "guardrails" in result["user_api_key_auth_metadata"]
    assert result["user_api_key_auth_metadata"]["guardrails"] == ["presidio", "aporia"]
    assert result["user_api_key_auth_metadata"]["other_field"] == "value"


@pytest.mark.asyncio
async def test_team_guardrails_append_to_key_guardrails():
    """
    Test that team guardrails are appended to key guardrails instead of overriding them.
    Team guardrails should only be added if they are not already present in key guardrails.
    """
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        metadata={"guardrails": ["key-guardrail-1", "key-guardrail-2"]},
        team_metadata={"guardrails": ["team-guardrail-1", "key-guardrail-1"]},
    )

    with patch("litellm.proxy.utils._premium_user_check"):
        updated_data = await add_litellm_data_to_request(
            data=data,
            request=request_mock,
            user_api_key_dict=user_api_key_dict,
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )

    metadata = updated_data.get("metadata", {})
    guardrails = metadata.get("guardrails", [])
    
    assert "key-guardrail-1" in guardrails
    assert "key-guardrail-2" in guardrails
    assert "team-guardrail-1" in guardrails
    assert guardrails.count("key-guardrail-1") == 1


@pytest.mark.asyncio
async def test_request_guardrails_do_not_override_key_guardrails():
    """
    Test that request-level guardrails do not override key-level guardrails.

    Key guardrails should be preserved when request contains guardrails (including empty array).
    """
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        metadata={"guardrails": ["key-guardrail-1"]},
        team_metadata={},
    )
    
    # Test case: Request with empty guardrails should not result in empty guardrails
    data_with_empty = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
        "guardrails": [],
    }

    with patch("litellm.proxy.utils._premium_user_check"):
        updated_data_empty = await add_litellm_data_to_request(
            data=data_with_empty,
            request=request_mock,
            user_api_key_dict=user_api_key_dict,
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )

    _metadata = updated_data_empty.get("metadata", {})
    requested_guardrails = _metadata.get("guardrails", [])
    
    assert "guardrails" not in updated_data_empty
    assert "key-guardrail-1" in requested_guardrails
    assert len(requested_guardrails) == 1


def test_update_model_if_key_alias_exists():
    """
    Test that _update_model_if_key_alias_exists properly updates the model when a key alias exists.
    """
    # Test case 1: Key alias exists and matches model
    data = {"model": "modelAlias", "messages": [{"role": "user", "content": "Hello"}]}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        aliases={"modelAlias": "xai/grok-4-fast-non-reasoning"},
    )
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert data["model"] == "xai/grok-4-fast-non-reasoning"

    # Test case 2: Key alias doesn't exist
    data = {"model": "unknown-model", "messages": [{"role": "user", "content": "Hello"}]}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        aliases={"modelAlias": "xai/grok-4-fast-non-reasoning"},
    )
    original_model = data["model"]
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert data["model"] == original_model  # Should remain unchanged

    # Test case 3: Model is None
    data = {"model": None, "messages": [{"role": "user", "content": "Hello"}]}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        aliases={"modelAlias": "xai/grok-4-fast-non-reasoning"},
    )
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert data["model"] is None  # Should remain None

    # Test case 4: Model key doesn't exist in data
    data = {"messages": [{"role": "user", "content": "Hello"}]}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        aliases={"modelAlias": "xai/grok-4-fast-non-reasoning"},
    )
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert "model" not in data  # Should not add model if it doesn't exist

    # Test case 5: Multiple aliases, matching one
    data = {"model": "alias1", "messages": [{"role": "user", "content": "Hello"}]}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        aliases={
            "alias1": "model1",
            "alias2": "model2",
            "alias3": "model3",
        },
    )
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert data["model"] == "model1"

    # Test case 6: Empty aliases dict
    data = {"model": "modelAlias", "messages": [{"role": "user", "content": "Hello"}]}
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key", aliases={})
    original_model = data["model"]
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert data["model"] == original_model  # Should remain unchanged


@pytest.mark.asyncio
async def test_embedding_header_forwarding_with_model_group():
    """
    Test that headers are properly forwarded for embedding requests when
    forward_client_headers_to_llm_api is configured for the model group.

    This test verifies the fix for embedding endpoints not forwarding headers
    similar to how chat completion endpoints do.
    """
    import importlib

    import litellm.proxy.litellm_pre_call_utils as pre_call_utils_module

    # Reload the module to ensure it has a fresh reference to litellm
    # This is necessary because conftest.py reloads litellm at module scope,
    # which can cause the module's litellm reference to become stale
    importlib.reload(pre_call_utils_module)

    # Re-import the function after reload to get the fresh version
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request for embeddings
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/embeddings"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/embeddings"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {
        "Content-Type": "application/json",
        "X-Custom-Header": "custom-value",
        "X-Request-ID": "test-request-123",
        "Authorization": "Bearer sk-test-key",
    }
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup embedding request data
    data = {
        "model": "local-openai/text-embedding-3-small",
        "input": ["Text to embed"],
    }

    # Setup user API key
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        org_id="test-org",
    )

    # Mock model_group_settings to enable header forwarding for the model
    # Use string-based patch to ensure we patch the current sys.modules['litellm']
    # This avoids issues with module reloading during parallel test execution
    mock_settings = MagicMock(forward_client_headers_to_llm_api=["local-openai/*"])
    with patch("litellm.model_group_settings", mock_settings):
        # Call add_litellm_data_to_request which includes header forwarding logic
        updated_data = await add_litellm_data_to_request(
            data=data,
            request=request_mock,
            user_api_key_dict=user_api_key_dict,
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )

        # Verify that headers were added to the request data
        assert "headers" in updated_data, "Headers should be added to embedding request"

        # Verify that only x- prefixed headers (except x-stainless) were forwarded
        forwarded_headers = updated_data["headers"]
        assert "X-Custom-Header" in forwarded_headers, "X-Custom-Header should be forwarded"
        assert forwarded_headers["X-Custom-Header"] == "custom-value"
        assert "X-Request-ID" in forwarded_headers, "X-Request-ID should be forwarded"
        assert forwarded_headers["X-Request-ID"] == "test-request-123"

        # Verify that authorization header was NOT forwarded (sensitive header)
        assert "Authorization" not in forwarded_headers, "Authorization header should not be forwarded"

        # Verify that Content-Type was NOT forwarded (doesn't start with x-)
        assert "Content-Type" not in forwarded_headers, "Content-Type should not be forwarded"

        # Verify original data fields are preserved
        assert updated_data["model"] == "local-openai/text-embedding-3-small"
        assert updated_data["input"] == ["Text to embed"]


@pytest.mark.asyncio
async def test_embedding_header_forwarding_without_model_group_config():
    """
    Test that headers are NOT forwarded for embedding requests when
    the model is not in the forward_client_headers_to_llm_api list.
    """
    import litellm

    # Setup mock request for embeddings
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/embeddings"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/embeddings"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {
        "Content-Type": "application/json",
        "X-Custom-Header": "custom-value",
    }
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup embedding request data with a model NOT in the forward list
    data = {
        "model": "text-embedding-ada-002",
        "input": ["Text to embed"],
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
    )

    # Mock model_group_settings with a different model in the forward list
    mock_settings = MagicMock(forward_client_headers_to_llm_api=["gpt-4", "claude-*"])
    original_model_group_settings = getattr(litellm, "model_group_settings", None)
    litellm.model_group_settings = mock_settings

    try:
        updated_data = await add_litellm_data_to_request(
            data=data,
            request=request_mock,
            user_api_key_dict=user_api_key_dict,
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )

        # Verify that headers were NOT added since model is not in forward list
        assert "headers" not in updated_data or updated_data.get("headers") is None, \
            "Headers should not be forwarded for models not in forward_client_headers_to_llm_api list"

        # Verify original data fields are preserved
        assert updated_data["model"] == "text-embedding-ada-002"
        assert updated_data["input"] == ["Text to embed"]

    finally:
        # Restore original model_group_settings
        litellm.model_group_settings = original_model_group_settings


def test_add_guardrails_from_policy_engine():
    """
    Test that add_guardrails_from_policy_engine adds guardrails from matching policies
    and tracks applied policies in metadata.
    """
    from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.types.proxy.policy_engine import (
        Policy,
        PolicyAttachment,
        PolicyGuardrails,
    )

    # Setup test data
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {},
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_alias="healthcare-team",
        key_alias="my-key",
    )

    # Setup mock policies in the registry (policies define WHAT guardrails to apply)
    policy_registry = get_policy_registry()
    policy_registry._policies = {
        "global-baseline": Policy(
            guardrails=PolicyGuardrails(add=["pii_blocker"]),
        ),
        "healthcare": Policy(
            guardrails=PolicyGuardrails(add=["hipaa_audit"]),
        ),
    }
    policy_registry._initialized = True

    # Setup attachments in the attachment registry (attachments define WHERE policies apply)
    attachment_registry = get_attachment_registry()
    attachment_registry._attachments = [
        PolicyAttachment(policy="global-baseline", scope="*"),  # applies to all
        PolicyAttachment(policy="healthcare", teams=["healthcare-team"]),  # applies to healthcare team
    ]
    attachment_registry._initialized = True

    # Call the function
    add_guardrails_from_policy_engine(
        data=data,
        metadata_variable_name="metadata",
        user_api_key_dict=user_api_key_dict,
    )

    # Verify guardrails were added
    assert "guardrails" in data["metadata"]
    assert "pii_blocker" in data["metadata"]["guardrails"]
    assert "hipaa_audit" in data["metadata"]["guardrails"]

    # Verify applied policies were tracked
    assert "applied_policies" in data["metadata"]
    assert "global-baseline" in data["metadata"]["applied_policies"]
    assert "healthcare" in data["metadata"]["applied_policies"]

    # Clean up registries
    policy_registry._policies = {}
    policy_registry._initialized = False
    attachment_registry._attachments = []
    attachment_registry._initialized = False


def test_add_guardrails_from_policy_engine_accepts_dynamic_policies_and_pops_from_data():
    """
    Test that add_guardrails_from_policy_engine accepts dynamic 'policies' from the request body
    and removes them to prevent forwarding to the LLM provider.
    
    This is critical because 'policies' is a LiteLLM proxy-specific parameter that should
    not be sent to the actual LLM API (e.g., OpenAI, Anthropic, etc.).
    """
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry

    # Setup test data with 'policies' in the request body
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "policies": ["PII-POLICY-GLOBAL", "HIPAA-POLICY"],  # Dynamic policies - should be accepted and removed
        "metadata": {},
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_alias="test-team",
        key_alias="test-key",
    )

    # Initialize empty policy registry (we're just testing the accept and pop behavior)
    policy_registry = get_policy_registry()
    policy_registry._policies = {}
    policy_registry._initialized = False

    # Call the function - should accept dynamic policies and not raise an error
    add_guardrails_from_policy_engine(
        data=data,
        metadata_variable_name="metadata",
        user_api_key_dict=user_api_key_dict,
    )

    # Verify that 'policies' was removed from the request body
    assert "policies" not in data, "'policies' should be removed from request body to prevent forwarding to LLM provider"

    # Verify that other fields are preserved
    assert "model" in data
    assert data["model"] == "gpt-4"
    assert "messages" in data
    assert data["messages"] == [{"role": "user", "content": "Hello"}]
    assert "metadata" in data
