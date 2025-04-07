import json
import os
import sys
from unittest.mock import ANY

import pytest
import respx
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router
from litellm.proxy._types import LiteLLM_UserTableFiltered, UserAPIKeyAuth
from litellm.proxy.hooks import get_proxy_hook
from litellm.proxy.management_endpoints.internal_user_endpoints import ui_view_users
from litellm.proxy.proxy_server import app

client = TestClient(app)
from litellm.caching.caching import DualCache
from litellm.proxy.proxy_server import hash_token
from litellm.proxy.utils import ProxyLogging


@pytest.fixture
def llm_router() -> Router:
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
                "model_info": {
                    "id": "azure-gpt-3-5-turbo-id",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "openai_api_key",
                },
                "model_info": {
                    "id": "gpt-3.5-turbo-id",
                },
            },
            {
                "model_name": "gemini-2.0-flash",
                "litellm_params": {
                    "model": "gemini/gemini-2.0-flash",
                },
                "model_info": {
                    "id": "gemini-2.0-flash-id",
                },
            },
        ]
    )
    return llm_router


def setup_proxy_logging_object(monkeypatch, llm_router: Router) -> ProxyLogging:
    proxy_logging_object = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    proxy_logging_object._add_proxy_hooks(llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_object
    )
    return proxy_logging_object


def test_invalid_purpose(mocker: MockerFixture, monkeypatch, llm_router: Router):
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


def test_mock_create_audio_file(mocker: MockerFixture, monkeypatch, llm_router: Router):
    """
    Asserts 'create_file' is called with the correct arguments
    """
    from litellm import Router

    mock_create_file = mocker.patch("litellm.files.main.create_file")

    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)

    # Create a simple test file content
    test_file_content = b"test audio content"
    test_file = ("test.wav", test_file_content, "audio/wav")

    response = client.post(
        "/v1/files",
        files={"file": test_file},
        data={
            "purpose": "user_data",
            "target_model_names": "azure-gpt-3-5-turbo, gpt-3.5-turbo",
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


@pytest.mark.skip(reason="mock respx fails on ci/cd - unclear why")
def test_create_file_and_call_chat_completion_e2e(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    1. Create a file
    2. Call a chat completion with the file
    3. Assert the file is used in the chat completion
    """
    # Create and enable respx mock instance
    mock = respx.mock()
    mock.start()
    try:
        from litellm.types.llms.openai import OpenAIFileObject

        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
        proxy_logging_object = setup_proxy_logging_object(monkeypatch, llm_router)

        # Create a simple test file content
        test_file_content = b"test audio content"
        test_file = ("test.wav", test_file_content, "audio/wav")

        # Mock the file creation response
        mock_file_response = OpenAIFileObject(
            id="test-file-id",
            object="file",
            bytes=123,
            created_at=1234567890,
            filename="test.wav",
            purpose="user_data",
            status="uploaded",
        )
        mock_file_response._hidden_params = {"model_id": "gemini-2.0-flash-id"}
        mocker.patch.object(llm_router, "acreate_file", return_value=mock_file_response)

        # Mock the Gemini API call using respx
        mock_gemini_response = {
            "candidates": [
                {"content": {"parts": [{"text": "This is a test audio file"}]}}
            ]
        }

        # Mock the Gemini API endpoint with a more flexible pattern
        gemini_route = mock.post(
            url__regex=r".*generativelanguage\.googleapis\.com.*"
        ).mock(
            return_value=respx.MockResponse(status_code=200, json=mock_gemini_response),
        )

        # Print updated mock setup
        print("\nAfter Adding Gemini Route:")
        print("==========================")
        print(f"Number of mocked routes: {len(mock.routes)}")
        for route in mock.routes:
            print(f"Mocked Route: {route}")
            print(f"Pattern: {route.pattern}")

        ## CREATE FILE
        file = client.post(
            "/v1/files",
            files={"file": test_file},
            data={
                "purpose": "user_data",
                "target_model_names": "gemini-2.0-flash, gpt-3.5-turbo",
            },
            headers={"Authorization": "Bearer test-key"},
        )

        print("\nAfter File Creation:")
        print("====================")
        print(f"File creation status: {file.status_code}")
        print(f"Recorded calls so far: {len(mock.calls)}")
        for call in mock.calls:
            print(f"Call made to: {call.request.method} {call.request.url}")

        assert file.status_code == 200
        assert file.json()["id"] != "test-file-id"  # unified file id used

        ## USE FILE IN CHAT COMPLETION
        try:
            completion = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-2.0-flash",
                    "modalities": ["text", "audio"],
                    "audio": {"voice": "alloy", "format": "wav"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What is in this recording?"},
                                {
                                    "type": "file",
                                    "file": {
                                        "file_id": file.json()["id"],
                                        "filename": "my-test-name",
                                        "format": "audio/wav",
                                    },
                                },
                            ],
                        },
                    ],
                    "drop_params": True,
                },
                headers={"Authorization": "Bearer test-key"},
            )
        except Exception as e:
            print(f"error: {e}")

        print("\nError occurred during chat completion:")
        print("=====================================")
        print("\nFinal Mock State:")
        print("=================")
        print(f"Total mocked routes: {len(mock.routes)}")
        for route in mock.routes:
            print(f"\nMocked Route: {route}")
            print(f"  Called: {route.called}")

        print("\nActual Requests Made:")
        print("=====================")
        print(f"Total calls recorded: {len(mock.calls)}")
        for idx, call in enumerate(mock.calls):
            print(f"\nCall {idx + 1}:")
            print(f"  Method: {call.request.method}")
            print(f"  URL: {call.request.url}")
            print(f"  Headers: {dict(call.request.headers)}")
            try:
                print(f"  Body: {call.request.content.decode()}")
            except:
                print("  Body: <could not decode>")

        # Verify Gemini API was called
        assert gemini_route.called, "Gemini API was not called"

        # Print the call details
        print("\nGemini API Call Details:")
        print(f"URL: {gemini_route.calls.last.request.url}")
        print(f"Method: {gemini_route.calls.last.request.method}")
        print(f"Headers: {dict(gemini_route.calls.last.request.headers)}")
        print(f"Content: {gemini_route.calls.last.request.content.decode()}")
        print(f"Response: {gemini_route.calls.last.response.content.decode()}")

        assert "test-file-id" in gemini_route.calls.last.request.content.decode()
    finally:
        # Stop the mock
        mock.stop()
