import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion, embedding
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


@pytest.fixture
def watsonx_embedding_call():
    def _call(
        model="watsonx/my-test-model",
        input=None,
        api_key="test_api_key",
        space_id: Optional[str] = None,
        headers=None,
        client=None,
        patch_token_call=True,
    ):
        if input is None:
            input = ["Hello, how are you?"]
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
                    embedding(
                        model=model,
                        input=input,
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
                    embedding(
                        model=model,
                        input=input,
                        api_key=api_key,
                        headers=headers or {},
                        client=client,
                        space_id=space_id,
                    )
                except Exception as e:
                    print(e)
                return mock_post, None

    return _call


@pytest.mark.parametrize("with_custom_auth_header", [True, False])
def test_watsonx_custom_auth_header(
    with_custom_auth_header, watsonx_chat_completion_call
):
    headers = (
        {"Authorization": "Bearer my-custom-auth-header"}
        if with_custom_auth_header
        else {}
    )

    mock_post, _ = watsonx_chat_completion_call(headers=headers)

    assert mock_post.call_count == 1
    if with_custom_auth_header:
        assert (
            mock_post.call_args[1]["headers"]["Authorization"]
            == "Bearer my-custom-auth-header"
        )
    else:
        assert (
            mock_post.call_args[1]["headers"]["Authorization"]
            == "Bearer mock_access_token"
        )


@pytest.mark.parametrize("env_var_key", ["WATSONX_ZENAPIKEY", "WATSONX_TOKEN"])
def test_watsonx_token_in_env_var(
    monkeypatch, watsonx_chat_completion_call, env_var_key
):
    monkeypatch.setenv(env_var_key, "my-custom-token")

    mock_post, _ = watsonx_chat_completion_call(patch_token_call=False)

    assert mock_post.call_count == 1
    if env_var_key == "WATSONX_ZENAPIKEY":
        assert (
            mock_post.call_args[1]["headers"]["Authorization"]
            == "ZenApiKey my-custom-token"
        )
    else:
        assert (
            mock_post.call_args[1]["headers"]["Authorization"]
            == "Bearer my-custom-token"
        )


def test_watsonx_chat_completions_endpoint(watsonx_chat_completion_call):
    model = "watsonx/another-model"
    messages = [{"role": "user", "content": "Test message"}]

    mock_post, _ = watsonx_chat_completion_call(model=model, messages=messages)

    assert mock_post.call_count == 1
    assert "deployment" not in mock_post.call_args.kwargs["url"]


def test_watsonx_chat_completions_endpoint_space_id(
    monkeypatch, watsonx_chat_completion_call
):
    my_fake_space_id = "xxx-xxx-xxx-xxx-xxx"
    monkeypatch.setenv("WATSONX_SPACE_ID", my_fake_space_id)

    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)

    model = "watsonx/another-model"
    messages = [{"role": "user", "content": "Test message"}]

    mock_post, _ = watsonx_chat_completion_call(model=model, messages=messages)

    assert mock_post.call_count == 1
    assert "deployment" not in mock_post.call_args.kwargs["url"]

    json_data = json.loads(mock_post.call_args.kwargs["data"])
    assert my_fake_space_id == json_data["space_id"]
    assert not json_data.get("project_id")


@pytest.mark.parametrize(
    "model",
    [
        "watsonx/deployment/<xxxx.xxx.xxx.xxxx>",
        "watsonx_text/deployment/<xxxx.xxx.xxx.xxxx>",
    ],
)
def test_watsonx_deployment_space_id(monkeypatch, watsonx_chat_completion_call, model):
    my_fake_space_id = "xxx-xxx-xxx-xxx-xxx"
    monkeypatch.setenv("WATSONX_SPACE_ID", my_fake_space_id)

    mock_post, _ = watsonx_chat_completion_call(
        model=model,
        messages=[{"content": "Hello, how are you?", "role": "user"}],
    )

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])
    assert my_fake_space_id not in json_data


@pytest.mark.parametrize(
    "model",
    [
        "watsonx/deployment/<xxxx.xxx.xxx.xxxx>",
        "watsonx_text/deployment/<xxxx.xxx.xxx.xxxx>",
    ],
)
def test_watsonx_deployment(watsonx_chat_completion_call, model):
    messages = [{"content": "Hello, how are you?", "role": "user"}]
    mock_post, _ = watsonx_chat_completion_call(
        model=model,
        messages=messages,
    )

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])

    # nor space_id or project_id is required by wx.ai API when inferencing deployment
    assert "project_id" not in json_data and "space_id" not in json_data


def test_watsonx_deployment_space_id_embedding(monkeypatch, watsonx_embedding_call):
    my_fake_space_id = "xxx-xxx-xxx-xxx-xxx"
    monkeypatch.setenv("WATSONX_SPACE_ID", my_fake_space_id)

    mock_post, _ = watsonx_embedding_call(model="watsonx/deployment/my-test-model")

    assert mock_post.call_count == 1
    json_data = json.loads(mock_post.call_args.kwargs["data"])

    # nor space_id or project_id is required by wx.ai API when inferencing deployment
    assert "project_id" not in json_data and "space_id" not in json_data
