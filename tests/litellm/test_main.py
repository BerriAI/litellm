import os
import sys

import pytest
from copy import deepcopy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock, patch

import litellm


@pytest.fixture
def openai_api_response():
    mock_response_data = {
        "id": "chatcmpl-B0W3vmiM78Xkgx7kI7dr7PC949DMS",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "",
                    "refusal": None,
                    "role": "assistant",
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ],
        "created": 1739462947,
        "model": "gpt-4o-mini-2024-07-18",
        "object": "chat.completion",
        "service_tier": "default",
        "system_fingerprint": "fp_bd83329f63",
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 121,
            "total_tokens": 122,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
    }

    return mock_response_data


def test_completion_missing_role(openai_api_response):
    from openai import OpenAI

    from litellm.types.utils import ModelResponse

    client = OpenAI(api_key="test_api_key")

    mock_raw_response = MagicMock()
    mock_raw_response.headers = {
        "x-request-id": "123",
        "openai-organization": "org-123",
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
    }
    mock_raw_response.parse.return_value = ModelResponse(**openai_api_response)

    print(f"openai_api_response: {openai_api_response}")

    with patch.object(
        client.chat.completions.with_raw_response, "create", mock_raw_response
    ) as mock_create:
        litellm.completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hey"},
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_m0vFJjQmTH1McvaHBPR2YFwY",
                            "function": {
                                "arguments": '{"input": "dksjsdkjdhskdjshdskhjkhlk"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 0,
                        },
                        {
                            "id": "call_Vw6RaqV2n5aaANXEdp5pYxo2",
                            "function": {
                                "arguments": '{"input": "jkljlkjlkjlkjlk"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 1,
                        },
                        {
                            "id": "call_hBIKwldUEGlNh6NlSXil62K4",
                            "function": {
                                "arguments": '{"input": "jkjlkjlkjlkj;lj"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 2,
                        },
                    ],
                },
            ],
            client=client,
        )

        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_chat_completion_call_does_not_change_original_attributes():
    """
    Test that the chat object is created correctly and that the original attributes are not changed
    by the completion.create calls.

    This is important in order to concurrent calls to completion.create do not override each other's attributes.
    """
    initial_params = {"acompletion": True, "timeout": 6000, "max_retries": 0, "metadata": {"caching_groups": None}, "dynamic_param": 0}
    default_litellm_params=deepcopy(initial_params)

    chat = litellm.Chat(params=default_litellm_params, router_obj=None)
    assert chat.params == default_litellm_params

    model_name = "gpt-4o-mini"
    prompt = "Hello, world!"
    new_dynamic_param = 1

    with patch("litellm.main.completion") as mocked_completion:
        await chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            metadata={"generation_name": prompt},
            dynamic_param=new_dynamic_param
        )

    mocked_completion.assert_called_once()

    # We don't want to change the original chat params
    assert default_litellm_params["dynamic_param"] == initial_params["dynamic_param"]

    # We want to change the params for this completion call
    _, ckwargs = mocked_completion.call_args
    assert ckwargs["dynamic_param"] == new_dynamic_param
