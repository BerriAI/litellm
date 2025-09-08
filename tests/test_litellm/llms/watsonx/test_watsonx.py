import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from unittest.mock import patch, Mock
import pytest
from typing import Optional


@pytest.fixture
def watsonx_chat_completion_call():
    def _call(
        model="watsonx/my-test-model",
        messages=None,
        api_key="test_api_key",
        space_id: Optional[str] = None,
        headers=None,
        client=None,
        patch_token_call=True,
    ):
        if messages is None:
            messages = [{"role": "user", "content": "Hello, how are you?"}]
        if client is None:
            client = HTTPHandler()

        if patch_token_call:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "mock_access_token",
                "expires_in": 3600,
            }
            mock_response.raise_for_status = Mock()  # No-op to simulate no exception

            with patch.object(client, "post") as mock_post, patch.object(
                litellm.module_level_client, "post", return_value=mock_response
            ) as mock_get:
                try:
                    completion(
                        model=model,
                        messages=messages,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)

                return mock_post, mock_get
        else:
            with patch.object(client, "post") as mock_post:
                try:
                    completion(
                        model=model,
                        messages=messages,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)
                return mock_post, None

    return _call


def test_watsonx_deployment_model_id_not_in_payload(
    monkeypatch, watsonx_chat_completion_call
):
    """Test that deployment models do not include 'model_id' in the request payload"""
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")
    model = "watsonx/deployment/test-deployment-id"
    messages = [{"role": "user", "content": "Test message"}]

    mock_post, _ = watsonx_chat_completion_call(model=model, messages=messages)

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])
    # Ensure model_id is not in the payload for deployment models
    assert "model_id" not in json_data or json_data["model_id"] is None
    # Ensure project_id is also not in the payload for deployment models
    assert "project_id" not in json_data or json_data["project_id"] is None


def test_watsonx_regular_model_includes_model_id(
    monkeypatch, watsonx_chat_completion_call
):
    """Test that regular models include 'model_id' in the request payload"""
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")
    model = "watsonx/regular-model"
    messages = [{"role": "user", "content": "Test message"}]

    mock_post, _ = watsonx_chat_completion_call(model=model, messages=messages)

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])
    # Ensure model_id is included in the payload for regular models
    assert "model_id" in json_data
    assert json_data["model_id"] == "regular-model"  # Provider prefix is stripped
    # Ensure project_id is also included for regular models
    assert "project_id" in json_data


@pytest.fixture
def watsonx_completion_call():
    def _call(
        model="watsonx_text/my-test-model",
        prompt="Hello, how are you?",
        api_key="test_api_key",
        space_id: Optional[str] = None,
        headers=None,
        client=None,
        patch_token_call=True,
    ):
        if client is None:
            client = HTTPHandler()

        if patch_token_call:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "mock_access_token",
                "expires_in": 3600,
            }
            mock_response.raise_for_status = Mock()

            with patch.object(client, "post") as mock_post, patch.object(
                litellm.module_level_client, "post", return_value=mock_response
            ) as mock_get:
                try:
                    litellm.text_completion(
                        model=model,
                        prompt=prompt,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)

                return mock_post, mock_get
        else:
            with patch.object(client, "post") as mock_post:
                try:
                    litellm.text_completion(
                        model=model,
                        prompt=prompt,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)
                return mock_post, None

    return _call


def test_watsonx_completion_deployment_model_id_not_in_payload(
    monkeypatch, watsonx_completion_call
):
    """Test that deployment models do not include 'model_id' in completion request payload"""
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")
    model = "watsonx_text/deployment/test-deployment-id"
    prompt = "Test prompt"

    mock_post, _ = watsonx_completion_call(model=model, prompt=prompt)

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])
    # Ensure model_id is not in the payload for deployment models
    assert "model_id" not in json_data
    # Ensure project_id is also not in the payload for deployment models
    assert "project_id" not in json_data


def test_watsonx_completion_regular_model_includes_model_id(
    monkeypatch, watsonx_completion_call
):
    """Test that regular models include 'model_id' in completion request payload"""
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("WATSONX_API_BASE", "https://test-api.watsonx.ai")
    model = "watsonx_text/regular-model"
    prompt = "Test prompt"

    mock_post, _ = watsonx_completion_call(model=model, prompt=prompt)

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])
    # Ensure model_id is included in the payload for regular models
    assert "model_id" in json_data
    assert json_data["model_id"] == "regular-model"  # Provider prefix is stripped
    # Ensure project_id is also included for regular models
    assert "project_id" in json_data
