import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import litellm
from litellm import completion, embedding
import pytest
from unittest.mock import MagicMock, patch
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
import pytest_asyncio
from openai import AsyncOpenAI


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
        openai_client.chat.completions.with_raw_response, "create", new=MagicMock()
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


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk_embedding(is_async):
    litellm.set_verbose = True
    litellm._turn_on_debug()

    if is_async:
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(api_key="fake-key")
        mock_method = AsyncMock()
        patch_target = openai_client.embeddings.create
    else:
        from openai import OpenAI

        openai_client = OpenAI(api_key="fake-key")
        mock_method = MagicMock()
        patch_target = openai_client.embeddings.create

    with patch.object(patch_target.__self__, patch_target.__name__, new=mock_method):
        try:
            if is_async:
                await litellm.aembedding(
                    model="litellm_proxy/my-vllm-model",
                    input="Hello world",
                    client=openai_client,
                    api_base="my-custom-api-base",
                )
            else:
                litellm.embedding(
                    model="litellm_proxy/my-vllm-model",
                    input="Hello world",
                    client=openai_client,
                    api_base="my-custom-api-base",
                )
        except Exception as e:
            print(e)

        mock_method.assert_called_once()

        print("Call KWARGS - {}".format(mock_method.call_args.kwargs))

        assert "Hello world" == mock_method.call_args.kwargs["input"]
        assert "my-vllm-model" == mock_method.call_args.kwargs["model"]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk_image_generation(is_async):
    litellm._turn_on_debug()

    if is_async:
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(api_key="fake-key")
        mock_method = AsyncMock()
        patch_target = openai_client.images.generate
    else:
        from openai import OpenAI

        openai_client = OpenAI(api_key="fake-key")
        mock_method = MagicMock()
        patch_target = openai_client.images.generate

    with patch.object(patch_target.__self__, patch_target.__name__, new=mock_method):
        try:
            if is_async:
                response = await litellm.aimage_generation(
                    model="litellm_proxy/dall-e-3",
                    prompt="A beautiful sunset over mountains",
                    client=openai_client,
                    api_base="my-custom-api-base",
                )
            else:
                response = litellm.image_generation(
                    model="litellm_proxy/dall-e-3",
                    prompt="A beautiful sunset over mountains",
                    client=openai_client,
                    api_base="my-custom-api-base",
                )
            print("response=", response)
        except Exception as e:
            print("got error", e)

        mock_method.assert_called_once()

        print("Call KWARGS - {}".format(mock_method.call_args.kwargs))

        assert (
            "A beautiful sunset over mountains"
            == mock_method.call_args.kwargs["prompt"]
        )
        assert "dall-e-3" == mock_method.call_args.kwargs["model"]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk_transcription(is_async):
    litellm.set_verbose = True
    litellm._turn_on_debug()

    if is_async:
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(api_key="fake-key")
        mock_method = AsyncMock()
        patch_target = openai_client.audio.transcriptions.create
    else:
        from openai import OpenAI

        openai_client = OpenAI(api_key="fake-key")
        mock_method = MagicMock()
        patch_target = openai_client.audio.transcriptions.create

    with patch.object(patch_target.__self__, patch_target.__name__, new=mock_method):
        try:
            if is_async:
                await litellm.atranscription(
                    model="litellm_proxy/whisper-1",
                    file=b"sample_audio",
                    client=openai_client,
                    api_base="my-custom-api-base",
                )
            else:
                litellm.transcription(
                    model="litellm_proxy/whisper-1",
                    file=b"sample_audio",
                    client=openai_client,
                    api_base="my-custom-api-base",
                )
        except Exception as e:
            print(e)

        mock_method.assert_called_once()

        print("Call KWARGS - {}".format(mock_method.call_args.kwargs))

        assert "whisper-1" == mock_method.call_args.kwargs["model"]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk_speech(is_async):
    litellm.set_verbose = True

    if is_async:
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(api_key="fake-key")
        mock_method = AsyncMock()
        patch_target = openai_client.audio.speech.create
    else:
        from openai import OpenAI

        openai_client = OpenAI(api_key="fake-key")
        mock_method = MagicMock()
        patch_target = openai_client.audio.speech.create

    with patch.object(patch_target.__self__, patch_target.__name__, new=mock_method):
        try:
            if is_async:
                await litellm.aspeech(
                    model="litellm_proxy/tts-1",
                    input="Hello, this is a test of text to speech",
                    voice="alloy",
                    client=openai_client,
                    api_base="my-custom-api-base",
                )
            else:
                litellm.speech(
                    model="litellm_proxy/tts-1",
                    input="Hello, this is a test of text to speech",
                    voice="alloy",
                    client=openai_client,
                    api_base="my-custom-api-base",
                )
        except Exception as e:
            print(e)

        mock_method.assert_called_once()

        print("Call KWARGS - {}".format(mock_method.call_args.kwargs))

        assert (
            "Hello, this is a test of text to speech"
            == mock_method.call_args.kwargs["input"]
        )
        assert "tts-1" == mock_method.call_args.kwargs["model"]
        assert "alloy" == mock_method.call_args.kwargs["voice"]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_litellm_gateway_from_sdk_rerank(is_async):
    litellm.set_verbose = True
    litellm._turn_on_debug()

    if is_async:
        client = AsyncHTTPHandler()
        mock_method = AsyncMock()
        patch_target = client.post
    else:
        client = HTTPHandler()
        mock_method = MagicMock()
        patch_target = client.post

    with patch.object(client, "post", new=mock_method):
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

        if is_async:
            mock_method.return_value = mock_response
        else:
            mock_method.return_value = mock_response

        try:
            if is_async:
                response = await litellm.arerank(
                    model="litellm_proxy/rerank-english-v2.0",
                    query="What is machine learning?",
                    documents=[
                        "Machine learning is a field of study in artificial intelligence",
                        "Biology is the study of living organisms",
                    ],
                    client=client,
                    api_base="my-custom-api-base",
                )
            else:
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
        except Exception as e:
            print(e)

        # Verify the request
        mock_method.assert_called_once()
        call_args = mock_method.call_args
        print("call_args=", call_args)

        # Check that the URL is correct
        assert "my-custom-api-base/v1/rerank" == call_args.kwargs["url"]

        # Check that the request body contains the expected data
        request_body = json.loads(call_args.kwargs["data"])
        assert request_body["query"] == "What is machine learning?"
        assert request_body["model"] == "rerank-english-v2.0"
        assert len(request_body["documents"]) == 2


def test_litellm_gateway_from_sdk_with_response_cost_in_additional_headers():
    litellm.set_verbose = True
    litellm._turn_on_debug()

    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    # Create mock response object
    mock_response = MagicMock()
    mock_response.headers = {"x-litellm-response-cost": "120"}
    mock_response.parse.return_value = litellm.ModelResponse(
        **{
            "id": "chatcmpl-BEkxQvRGp9VAushfAsOZCbhMFLsoy",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": "Hello! How can I assist you today?",
                        "refusal": None,
                        "role": "assistant",
                        "annotations": [],
                        "audio": None,
                        "function_call": None,
                        "tool_calls": None,
                    },
                }
            ],
            "created": 1742856796,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion",
            "service_tier": "default",
            "system_fingerprint": "fp_6ec83003ad",
            "usage": {
                "completion_tokens": 10,
                "prompt_tokens": 9,
                "total_tokens": 19,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            },
        }
    )

    with patch.object(
        openai_client.chat.completions.with_raw_response,
        "create",
        return_value=mock_response,
    ) as mock_call:
        response = litellm.completion(
            model="litellm_proxy/gpt-4o",
            messages=[{"role": "user", "content": "Hello world"}],
            api_base="http://0.0.0.0:4000",
            api_key="sk-PIp1h0RekR",
            client=openai_client,
        )

        # Assert the headers were properly passed through
        print(f"additional_headers: {response._hidden_params['additional_headers']}")
        assert (
            response._hidden_params["additional_headers"][
                "llm_provider-x-litellm-response-cost"
            ]
            == "120"
        )

        assert response._hidden_params["response_cost"] == 120


def test_litellm_gateway_from_sdk_with_thinking_param():
    try: 
        response = litellm.completion(
            model="litellm_proxy/anthropic.claude-3-7-sonnet-20250219-v1:0",
            messages=[{"role": "user", "content": "Hello world"}],
            api_base="http://0.0.0.0:4000",
            api_key="sk-PIp1h0RekR",
            # client=openai_client,
            thinking={"type": "enabled", "max_budget": 100},
        )
        pytest.fail("Expected an error to be raised")
    except Exception as e:
        assert "Connection error." in str(e)
    
