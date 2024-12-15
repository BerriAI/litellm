import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import json

import pytest

import litellm
from litellm import completion
from litellm.llms.cohere.completion.transformation import CohereTextConfig


def test_cohere_generate_api_completion():
    try:
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        from unittest.mock import patch, MagicMock

        client = HTTPHandler()
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]

        with patch.object(client, "post") as mock_client:
            try:
                completion(
                    model="cohere/command",
                    messages=messages,
                    max_tokens=10,
                    client=client,
                )
            except Exception as e:
                print(e)
            mock_client.assert_called_once()
            print("mock_client.call_args.kwargs", mock_client.call_args.kwargs)

            assert (
                mock_client.call_args.kwargs["url"]
                == "https://api.cohere.ai/v1/generate"
            )
            json_data = json.loads(mock_client.call_args.kwargs["data"])
            assert json_data["model"] == "command"
            assert json_data["prompt"] == "You're a good bot Hey"
            assert json_data["max_tokens"] == 10
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_cohere_generate_api_stream():
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = await litellm.acompletion(
            model="cohere/command",
            messages=messages,
            max_tokens=10,
            stream=True,
        )
        print("async cohere stream response", response)
        async for chunk in response:
            print(chunk)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_cohere_stream_bad_key():
    try:
        api_key = "bad-key"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        completion(
            model="command",
            messages=messages,
            stream=True,
            max_tokens=50,
            api_key=api_key,
        )

    except litellm.AuthenticationError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_transform_request():
    try:
        config = CohereTextConfig()
        messages = [
            {"role": "system", "content": "You're a helpful bot"},
            {"role": "user", "content": "Hello"},
        ]
        optional_params = {"max_tokens": 10, "temperature": 0.7}
        headers = {}

        transformed_request = config.transform_request(
            model="command",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers=headers,
        )

        print("transformed_request", json.dumps(transformed_request, indent=4))

        assert transformed_request["model"] == "command"
        assert transformed_request["prompt"] == "You're a helpful bot Hello"
        assert transformed_request["max_tokens"] == 10
        assert transformed_request["temperature"] == 0.7
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_transform_request_with_tools():
    try:
        config = CohereTextConfig()
        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        optional_params = {"tools": tools}

        transformed_request = config.transform_request(
            model="command",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        print("transformed_request", json.dumps(transformed_request, indent=4))
        assert "tools" in transformed_request
        assert transformed_request["tools"] == {"tools": tools}
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_map_openai_params():
    try:
        config = CohereTextConfig()
        openai_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "n": 2,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "stop": ["END"],
            "stream": True,
        }

        mapped_params = config.map_openai_params(
            non_default_params=openai_params,
            optional_params={},
            model="command",
            drop_params=False,
        )

        assert mapped_params["temperature"] == 0.7
        assert mapped_params["max_tokens"] == 100
        assert mapped_params["num_generations"] == 2
        assert mapped_params["p"] == 0.9
        assert mapped_params["frequency_penalty"] == 0.5
        assert mapped_params["presence_penalty"] == 0.5
        assert mapped_params["stop_sequences"] == ["END"]
        assert mapped_params["stream"] == True
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
