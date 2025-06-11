import json
import os

from unittest.mock import patch

import pytest

from litellm import completion
from litellm.llms.custom_httpx.http_handler import HTTPHandler


@patch.dict(os.environ, {}, clear=True)
def test_completion_datarobot():
    """Ensure that the completion function works with DataRobot API."""
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    try:
        client = HTTPHandler()
        with patch.object(client, "post") as mock_post:
            response = completion(
                model="datarobot/vertex_ai/gemini-1.5-flash-002",
                messages=messages,
                client=client,
                max_tokens=5,
                clientId="custom-model",
            )
            print(response)

            # Add any assertions here to check the response
            mock_post.assert_called_once()
            mocks_kwargs = mock_post.call_args.kwargs
            assert mocks_kwargs["url"] == "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/"
            assert mocks_kwargs["headers"]["Authorization"] == "Bearer fake-api-key"
            json_data = json.loads(mock_post.call_args.kwargs["data"])
            assert json_data["clientId"] == "custom-model"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@patch.dict(
    os.environ, {"DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2/deployments/deployment_id/"}, clear=True
)
def test_completion_datarobot_with_deployment():
    """Ensure that deployment URL is used correctly."""
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    try:
        client = HTTPHandler()
        with patch.object(client, "post") as mock_post:
            response = completion(
                model="datarobot/vertex_ai/gemini-1.5-flash-002",
                messages=messages,
                client=client,
                max_tokens=5,
                clientId="custom-model",
            )
            print(response)

            # Add any assertions here to check the response
            mock_post.assert_called_once()
            mocks_kwargs = mock_post.call_args.kwargs
            assert mocks_kwargs["url"] == "https://app.datarobot.com/api/v2/deployments/deployment_id/"
            assert mocks_kwargs["headers"]["Authorization"] == "Bearer fake-api-key"
            json_data = json.loads(mock_post.call_args.kwargs["data"])
            assert json_data["clientId"] == "custom-model"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_datarobot_with_environment_variables():
    """Allow the test to run with environment variables if they are set for integrations."""
    # If keys are not set, the test will be skipped
    if os.environ.get("DATAROBOT_API_TOKEN") is None:
        return

    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    try:
        response = completion(
            model="datarobot/vertex_ai/gemini-1.5-flash-002", messages=messages, max_tokens=5, clientId="custom-model"
        )
        print(response)
        assert response["object"] == "chat.completion"
        assert response["model"] == "gemini-1.5-flash-002"
        assert len(response["choices"]) == 1
        assert len(response["choices"][0]["message"]["content"]) > 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
