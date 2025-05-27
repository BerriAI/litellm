import asyncio
import copy
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import (
    _get_enforced_params,
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
