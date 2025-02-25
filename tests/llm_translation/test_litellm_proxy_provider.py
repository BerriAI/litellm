import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import completion, embedding
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk():
    litellm.set_verbose = True
    messages = [
        {
            "role": "user",
            "content": "Hello world",
        }
    ]
    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(
        openai_client.chat.completions, "create", new=MagicMock()
    ) as mock_call:
        try:
            completion(
                model="litellm_proxy/my-vllm-model",
                messages=messages,
                response_format={"type": "json_object"},
                client=openai_client,
                api_base="my-custom-api-base",
                hello="world",
            )
        except Exception as e:
            print(e)

        mock_call.assert_called_once()

        print("Call KWARGS - {}".format(mock_call.call_args.kwargs))

        assert "hello" in mock_call.call_args.kwargs["extra_body"]


@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk_structured_output():
    from pydantic import BaseModel

    class Result(BaseModel):
        answer: str

    litellm.set_verbose = True
    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(
        openai_client.chat.completions, "create", new=MagicMock()
    ) as mock_call:
        try:
            litellm.completion(
                model="litellm_proxy/openai/gpt-4o",
                messages=[
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                api_key="my-test-api-key",
                user="test",
                response_format=Result,
                base_url="https://litellm.ml-serving-internal.scale.com",
                client=openai_client,
            )
        except Exception as e:
            print(e)

        mock_call.assert_called_once()

        print("Call KWARGS - {}".format(mock_call.call_args.kwargs))
        json_schema = mock_call.call_args.kwargs["response_format"]
        assert "json_schema" in json_schema


@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk_embedding():
    litellm.set_verbose = True
    litellm._turn_on_debug()
    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(openai_client.embeddings, "create", new=MagicMock()) as mock_call:
        try:
            litellm.embedding(
                model="litellm_proxy/my-vllm-model",
                input="Hello world",
                client=openai_client,
                api_base="my-custom-api-base",
            )
        except Exception as e:
            print(e)

        mock_call.assert_called_once()

        print("Call KWARGS - {}".format(mock_call.call_args.kwargs))

        assert "Hello world" == mock_call.call_args.kwargs["input"]
        assert "my-vllm-model" == mock_call.call_args.kwargs["model"]
