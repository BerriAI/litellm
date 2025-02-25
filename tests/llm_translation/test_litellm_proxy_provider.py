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
from litellm.llms.custom_httpx.http_handler import HTTPHandler


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


@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk_image_generation():
    litellm.set_verbose = True
    litellm._turn_on_debug()
    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(openai_client.images, "generate", new=MagicMock()) as mock_call:
        try:
            response = litellm.image_generation(
                model="litellm_proxy/dall-e-3",
                prompt="A beautiful sunset over mountains",
                client=openai_client,
                api_base="my-custom-api-base",
            )
            print("response=", response)
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")

        mock_call.assert_called_once()

        print("Call KWARGS - {}".format(mock_call.call_args.kwargs))

        assert (
            "A beautiful sunset over mountains" == mock_call.call_args.kwargs["prompt"]
        )
        assert "dall-e-3" == mock_call.call_args.kwargs["model"]


@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk_transcription():
    litellm.set_verbose = True
    litellm._turn_on_debug()
    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(
        openai_client.audio.transcriptions, "create", new=MagicMock()
    ) as mock_call:
        try:
            litellm.transcription(
                model="litellm_proxy/whisper-1",
                file=b"sample_audio",
                client=openai_client,
                api_base="my-custom-api-base",
            )
        except Exception as e:
            print(e)

        mock_call.assert_called_once()

        print("Call KWARGS - {}".format(mock_call.call_args.kwargs))

        assert "whisper-1" == mock_call.call_args.kwargs["model"]


@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk_speech():
    litellm.set_verbose = True
    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(
        openai_client.audio.speech, "create", new=MagicMock()
    ) as mock_call:
        try:
            litellm.speech(
                model="litellm_proxy/tts-1",
                input="Hello, this is a test of text to speech",
                voice="alloy",
                client=openai_client,
                api_base="my-custom-api-base",
            )
        except Exception as e:
            print(e)

        mock_call.assert_called_once()

        print("Call KWARGS - {}".format(mock_call.call_args.kwargs))

        assert (
            "Hello, this is a test of text to speech"
            == mock_call.call_args.kwargs["input"]
        )
        assert "tts-1" == mock_call.call_args.kwargs["model"]
        assert "alloy" == mock_call.call_args.kwargs["voice"]


@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk_rerank():
    litellm.set_verbose = True
    litellm._turn_on_debug()
    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        mock_response = MagicMock()

        # Create a mock response similar to OpenAI's rerank response
        mock_response.text = json.dumps(
            {
                "id": "rerank-123456",
                "object": "reranking",
                "results": [
                    {
                        "index": 0,
                        "relevance_score": 0.9,
                        "document": {
                            "id": "0",
                            "text": "Machine learning is a field of study in artificial intelligence",
                        },
                    },
                    {
                        "index": 1,
                        "relevance_score": 0.2,
                        "document": {
                            "id": "1",
                            "text": "Biology is the study of living organisms",
                        },
                    },
                ],
                "model": "rerank-english-v2.0",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            }
        )

        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        response = litellm.rerank(
            model="litellm_proxy/rerank-english-v2.0",
            query="What is machine learning?",
            documents=[
                "Machine learning is a field of study in artificial intelligence",
                "Biology is the study of living organisms",
            ],
            client=client,
            api_base="my-custom-api-base",
        )

        # Verify the request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        print("call_args=", call_args)

        # Check that the URL is correct
        assert "my-custom-api-base/v1/rerank" == call_args.kwargs["url"]

        # Check that the request body contains the expected data
        request_body = json.loads(call_args.kwargs["data"])
        assert request_body["query"] == "What is machine learning?"
        assert request_body["model"] == "rerank-english-v2.0"
        assert len(request_body["documents"]) == 2
