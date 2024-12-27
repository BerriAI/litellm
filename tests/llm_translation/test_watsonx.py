import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion
from litellm.llms.watsonx.common_utils import IBMWatsonXMixin
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from unittest.mock import patch, MagicMock, AsyncMock, Mock
import pytest


@pytest.fixture
def watsonx_chat_completion_call():
    def _call(
        model="watsonx/my-test-model",
        messages=None,
        api_key="test_api_key",
        headers=None,
        client=None,
    ):
        if messages is None:
            messages = [{"role": "user", "content": "Hello, how are you?"}]
        if client is None:
            client = HTTPHandler()

        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "mock_access_token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = Mock()  # No-op to simulate no exception

        with patch.object(client, "post") as mock_post, patch.object(
            litellm.module_level_client, "post", return_value=mock_response
        ) as mock_get:
            completion(
                model=model,
                messages=messages,
                api_key=api_key,
                headers=headers or {},
                client=client,
            )

            return mock_post, mock_get

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


def test_watsonx_chat_completions_endpoint(watsonx_chat_completion_call):
    model = "watsonx/another-model"
    messages = [{"role": "user", "content": "Test message"}]

    mock_post, _ = watsonx_chat_completion_call(model=model, messages=messages)

    assert mock_post.call_count == 1
    assert "deployment" not in mock_post.call_args.kwargs["url"]
