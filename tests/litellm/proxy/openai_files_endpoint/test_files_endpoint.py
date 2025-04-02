import json
import os
import sys
from unittest.mock import ANY

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import LiteLLM_UserTableFiltered, UserAPIKeyAuth
from litellm.proxy.management_endpoints.internal_user_endpoints import ui_view_users
from litellm.proxy.proxy_server import app

client = TestClient(app)


def test_invalid_purpose(mocker: MockerFixture, monkeypatch):
    """
    Asserts 'create_file' is called with the correct arguments
    """
    # Create a simple test file content
    test_file_content = b"test audio content"
    test_file = ("test.wav", test_file_content, "audio/wav")

    response = client.post(
        "/v1/files",
        files={"file": test_file},
        data={
            "purpose": "my-bad-purpose",
            "target_model_names": ["azure-gpt-3-5-turbo", "gpt-3.5-turbo"],
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 400
    print(f"response: {response.json()}")
    assert "Invalid purpose: my-bad-purpose" in response.json()["error"]["message"]


def test_mock_create_audio_file(mocker: MockerFixture, monkeypatch):
    """
    Asserts 'create_file' is called with the correct arguments
    """
    from litellm import Router

    mock_create_file = mocker.patch("litellm.files.main.create_file")

    llm_router = Router(
        model_list=[
            {
                "model_name": "azure-gpt-3-5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": "azure_api_key",
                    "api_base": "azure_api_base",
                    "api_version": "azure_api_version",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "openai_api_key",
                },
            },
        ]
    )

    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)

    # Create a simple test file content
    test_file_content = b"test audio content"
    test_file = ("test.wav", test_file_content, "audio/wav")

    response = client.post(
        "/v1/files",
        files={"file": test_file},
        data={
            "purpose": "user_data",
            "target_model_names": ["azure-gpt-3-5-turbo", "gpt-3.5-turbo"],
        },
        headers={"Authorization": "Bearer test-key"},
    )

    print(f"response: {response.text}")
    assert response.status_code == 200

    # Get all calls made to create_file
    calls = mock_create_file.call_args_list

    # Check for Azure call
    azure_call_found = False
    for call in calls:
        kwargs = call.kwargs
        if (
            kwargs.get("custom_llm_provider") == "azure"
            and kwargs.get("model") == "azure/chatgpt-v-2"
            and kwargs.get("api_key") == "azure_api_key"
        ):
            azure_call_found = True
            break
    assert (
        azure_call_found
    ), f"Azure call not found with expected parameters. Calls: {calls}"

    # Check for OpenAI call
    openai_call_found = False
    for call in calls:
        kwargs = call.kwargs
        if (
            kwargs.get("custom_llm_provider") == "openai"
            and kwargs.get("model") == "openai/gpt-3.5-turbo"
            and kwargs.get("api_key") == "openai_api_key"
        ):
            openai_call_found = True
            break
    assert openai_call_found, "OpenAI call not found with expected parameters"
