import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter
from unittest.mock import patch, MagicMock, AsyncMock

import litellm
from litellm import Choices, Message, ModelResponse, EmbeddingResponse, Usage
from litellm import completion


def test_completion_nvidia_nim():
    from openai import OpenAI

    litellm.set_verbose = True
    model_name = "nvidia_nim/databricks/dbrx-instruct"
    client = OpenAI(
        api_key="fake-api-key",
    )

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            completion(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in Boston today in Fahrenheit?",
                    }
                ],
                presence_penalty=0.5,
                frequency_penalty=0.1,
                client=client,
            )
        except Exception as e:
            print(e)
        # Add any assertions here to check the response

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["messages"] == [
            {
                "role": "user",
                "content": "What's the weather like in Boston today in Fahrenheit?",
            },
        ]
        assert request_body["model"] == "databricks/dbrx-instruct"
        assert request_body["frequency_penalty"] == 0.1
        assert request_body["presence_penalty"] == 0.5


def test_embedding_nvidia_nim():
    litellm.set_verbose = True
    from openai import OpenAI

    client = OpenAI(
        api_key="fake-api-key",
    )
    with patch.object(client.embeddings.with_raw_response, "create") as mock_client:
        try:
            litellm.embedding(
                model="nvidia_nim/nvidia/nv-embedqa-e5-v5",
                input="What is the meaning of life?",
                input_type="passage",
                dimensions=1024,
                client=client,
            )
        except Exception as e:
            print(e)
        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs
        print("request_body: ", request_body)
        assert request_body["input"] == "What is the meaning of life?"
        assert request_body["model"] == "nvidia/nv-embedqa-e5-v5"
        assert request_body["extra_body"]["input_type"] == "passage"
        assert request_body["dimensions"] == 1024
