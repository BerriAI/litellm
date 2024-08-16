import json
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

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.prompt_templates.factory import anthropic_messages_pt

# litellm.num_retries =3
litellm.cache = None
litellm.success_callback = []
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]
import logging

from litellm._logging import verbose_logger


def logger_fn(user_model_dict):
    print(f"user_model_dict: {user_model_dict}")


@pytest.fixture(autouse=True)
def reset_callbacks():
    print("\npytest fixture - resetting callbacks")
    litellm.success_callback = []
    litellm._async_success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []


@pytest.mark.asyncio()
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_completion_sagemaker(sync_mode):
    try:
        litellm.set_verbose = True
        verbose_logger.setLevel(logging.DEBUG)
        print("testing sagemaker")
        if sync_mode is True:
            response = litellm.completion(
                model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
                messages=[
                    {"role": "user", "content": "hi"},
                ],
                temperature=0.2,
                max_tokens=80,
                input_cost_per_second=0.000420,
            )
        else:
            response = await litellm.acompletion(
                model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
                messages=[
                    {"role": "user", "content": "hi"},
                ],
                temperature=0.2,
                max_tokens=80,
                input_cost_per_second=0.000420,
            )
        # Add any assertions here to check the response
        print(response)
        cost = completion_cost(completion_response=response)
        print("calculated cost", cost)
        assert (
            cost > 0.0 and cost < 1.0
        )  # should never be > $1 for a single completion call
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio()
@pytest.mark.parametrize("sync_mode", [False, True])
async def test_completion_sagemaker_stream(sync_mode):
    try:
        litellm.set_verbose = False
        print("testing sagemaker")
        verbose_logger.setLevel(logging.DEBUG)
        full_text = ""
        if sync_mode is True:
            response = litellm.completion(
                model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
                messages=[
                    {"role": "user", "content": "hi - what is ur name"},
                ],
                temperature=0.2,
                stream=True,
                max_tokens=80,
                input_cost_per_second=0.000420,
            )

            for chunk in response:
                print(chunk)
                full_text += chunk.choices[0].delta.content or ""

            print("SYNC RESPONSE full text", full_text)
        else:
            response = await litellm.acompletion(
                model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
                messages=[
                    {"role": "user", "content": "hi - what is ur name"},
                ],
                stream=True,
                temperature=0.2,
                max_tokens=80,
                input_cost_per_second=0.000420,
            )

            print("streaming response")

            async for chunk in response:
                print(chunk)
                full_text += chunk.choices[0].delta.content or ""

            print("ASYNC RESPONSE full text", full_text)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_acompletion_sagemaker_non_stream():
    mock_response = AsyncMock()

    def return_val():
        return {
            "generated_text": "This is a mock response from SageMaker.",
            "id": "cmpl-mockid",
            "object": "text_completion",
            "created": 1629800000,
            "model": "sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            "choices": [
                {
                    "text": "This is a mock response from SageMaker.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 8, "total_tokens": 9},
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    expected_payload = {
        "inputs": "hi",
        "parameters": {"temperature": 0.2, "max_new_tokens": 80},
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        # Act: Call the litellm.acompletion function
        response = await litellm.acompletion(
            model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            messages=[
                {"role": "user", "content": "hi"},
            ],
            temperature=0.2,
            max_tokens=80,
            input_cost_per_second=0.000420,
        )

        # Print what was called on the mock
        print("call args=", mock_post.call_args)

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        args_to_sagemaker = kwargs["json"]
        print("Arguments passed to sagemaker=", args_to_sagemaker)
        assert args_to_sagemaker == expected_payload
        assert (
            kwargs["url"]
            == "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/jumpstart-dft-hf-textgeneration1-mp-20240815-185614/invocations"
        )


@pytest.mark.asyncio
async def test_completion_sagemaker_non_stream():
    mock_response = MagicMock()

    def return_val():
        return {
            "generated_text": "This is a mock response from SageMaker.",
            "id": "cmpl-mockid",
            "object": "text_completion",
            "created": 1629800000,
            "model": "sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            "choices": [
                {
                    "text": "This is a mock response from SageMaker.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 8, "total_tokens": 9},
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    expected_payload = {
        "inputs": "hi",
        "parameters": {"temperature": 0.2, "max_new_tokens": 80},
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        # Act: Call the litellm.acompletion function
        response = litellm.completion(
            model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            messages=[
                {"role": "user", "content": "hi"},
            ],
            temperature=0.2,
            max_tokens=80,
            input_cost_per_second=0.000420,
        )

        # Print what was called on the mock
        print("call args=", mock_post.call_args)

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        args_to_sagemaker = kwargs["json"]
        print("Arguments passed to sagemaker=", args_to_sagemaker)
        assert args_to_sagemaker == expected_payload
        assert (
            kwargs["url"]
            == "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/jumpstart-dft-hf-textgeneration1-mp-20240815-185614/invocations"
        )


@pytest.mark.asyncio
async def test_completion_sagemaker_non_stream_with_aws_params():
    mock_response = MagicMock()

    def return_val():
        return {
            "generated_text": "This is a mock response from SageMaker.",
            "id": "cmpl-mockid",
            "object": "text_completion",
            "created": 1629800000,
            "model": "sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            "choices": [
                {
                    "text": "This is a mock response from SageMaker.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 8, "total_tokens": 9},
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    expected_payload = {
        "inputs": "hi",
        "parameters": {"temperature": 0.2, "max_new_tokens": 80},
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        # Act: Call the litellm.acompletion function
        response = litellm.completion(
            model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            messages=[
                {"role": "user", "content": "hi"},
            ],
            temperature=0.2,
            max_tokens=80,
            input_cost_per_second=0.000420,
            aws_access_key_id="gm",
            aws_secret_access_key="s",
            aws_region_name="us-west-5",
        )

        # Print what was called on the mock
        print("call args=", mock_post.call_args)

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        args_to_sagemaker = kwargs["json"]
        print("Arguments passed to sagemaker=", args_to_sagemaker)
        assert args_to_sagemaker == expected_payload
        assert (
            kwargs["url"]
            == "https://runtime.sagemaker.us-west-5.amazonaws.com/endpoints/jumpstart-dft-hf-textgeneration1-mp-20240815-185614/invocations"
        )
