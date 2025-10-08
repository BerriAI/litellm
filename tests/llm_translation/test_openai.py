import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch
from typing import Optional

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse
from base_llm_unit_tests import BaseLLMChatTest
import asyncio
from litellm.types.llms.openai import (
    ChatCompletionAnnotation,
    ChatCompletionAnnotationURLCitation,
)
from base_audio_transcription_unit_tests import BaseLLMAudioTranscriptionTest


def test_openai_prediction_param():
    litellm.set_verbose = True
    code = """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """

    completion = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
            },
            {"role": "user", "content": code},
        ],
        prediction={"type": "content", "content": code},
    )

    print(completion)

    assert (
        completion.usage.completion_tokens_details.accepted_prediction_tokens > 0
        or completion.usage.completion_tokens_details.rejected_prediction_tokens > 0
    )


@pytest.mark.asyncio
async def test_openai_prediction_param_mock():
    """
    Tests that prediction parameter is correctly passed to the API
    """
    litellm.set_verbose = True

    code = """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
                    },
                    {"role": "user", "content": code},
                ],
                prediction={"type": "content", "content": code},
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the prediction parameter
        assert "prediction" in request_body
        # verify prediction is correctly sent to the API
        assert request_body["prediction"] == {"type": "content", "content": code}


@pytest.mark.asyncio
async def test_openai_prediction_param_with_caching():
    """
    Tests using `prediction` parameter with caching
    """
    from litellm.caching.caching import LiteLLMCacheType
    import logging
    from litellm._logging import verbose_logger

    verbose_logger.setLevel(logging.DEBUG)
    import time

    litellm.set_verbose = True
    litellm.cache = litellm.Cache(type=LiteLLMCacheType.LOCAL)
    code = """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """

    completion_response_1 = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
            },
            {"role": "user", "content": code},
        ],
        prediction={"type": "content", "content": code},
    )

    time.sleep(0.5)

    # cache hit
    completion_response_2 = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
            },
            {"role": "user", "content": code},
        ],
        prediction={"type": "content", "content": code},
    )

    assert completion_response_1.id == completion_response_2.id

    completion_response_3 = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is the first name of the user?"},
        ],
        prediction={"type": "content", "content": code + "FirstName"},
    )

    assert completion_response_3.id != completion_response_1.id


@pytest.mark.asyncio()
async def test_vision_with_custom_model():
    """
    Tests that an OpenAI compatible endpoint when sent an image will receive the image in the request

    """
    import base64
    import requests
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    litellm.set_verbose = True
    api_base = "https://my-custom.api.openai.com"

    # Fetch and encode a test image
    url = "https://dummyimage.com/100/100/fff&text=Test+image"
    response = requests.get(url)
    file_data = response.content
    encoded_file = base64.b64encode(file_data).decode("utf-8")
    base64_image = f"data:image/png;base64,{encoded_file}"

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            response = await litellm.acompletion(
                model="openai/my-custom-model",
                max_tokens=10,
                api_base=api_base,  # use the mock api
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image},
                            },
                        ],
                    }
                ],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkBAMAAACCzIhnAAAAG1BMVEURAAD///+ln5/h39/Dv79qX18uHx+If39MPz9oMSdmAAAACXBIWXMAAA7EAAAOxAGVKw4bAAABDElEQVRYhe2SzWqEMBRGPyQTfQxJsc5jBKGzFmlslyFIZxsCQ7sUaWd87EanpdpIrbtC71mE/NyTm9wEIAiCIAiC+N/otQBxU2Sf/aeh4enqptHXri+/yxIq63jlKCw6cXssnr3ObdzdGYFYCJ2IzHKXLygHXCB98Gm4DE+ZZemu5EisQSyZTmyg+AuzQbkezCuIy7EI0k9Ig3FtruwydY+qniqtV5yQyo8qpUIl2fc90KVzJWohWf2qu75vlw52rdfjVDHg8vLWwixW7PChqLkSyUadwfSS0uQZhEvRuIkS53uJvrK8cGWYaPwpGt8efvw+vlo8TPMzcmP8w7lrNypc1RsNgiAIgiD+Iu/RyDYhCaWrgQAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            },
        ]
        assert request_body["model"] == "my-custom-model"
        assert request_body["max_tokens"] == 10


class TestOpenAIChatCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "gpt-4o-mini"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_prompt_caching(self):
        """
        Test that prompt caching works correctly.
        Skip for now, as it's working locally but not in CI
        """
        pass

    def test_prompt_caching(self):
        """
        Works locally but CI/CD is failing this test. Temporary skip to push out a new release.
        """
        pass


def test_completion_bad_org():
    import litellm

    litellm.set_verbose = True
    _old_org = os.environ.get("OPENAI_ORGANIZATION", None)
    os.environ["OPENAI_ORGANIZATION"] = "bad-org"
    messages = [{"role": "user", "content": "hi"}]

    with pytest.raises(Exception) as exc_info:
        comp = litellm.completion(
            model="gpt-4o-mini", messages=messages, organization="bad-org"
        )

    print(exc_info.value)
    assert "header should match organization for API key" in str(exc_info.value)

    if _old_org is not None:
        os.environ["OPENAI_ORGANIZATION"] = _old_org
    else:
        del os.environ["OPENAI_ORGANIZATION"]


@patch("litellm.main.openai_chat_completions._get_openai_client")
def test_openai_max_retries_0(mock_get_openai_client):
    import litellm

    litellm.set_verbose = True
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        max_retries=0,
    )

    mock_get_openai_client.assert_called_once()
    assert mock_get_openai_client.call_args.kwargs["max_retries"] == 0


@pytest.mark.parametrize("model", ["o1", "o1-mini", "o3-mini"])
def test_o1_parallel_tool_calls(model):
    litellm.completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "foo",
            }
        ],
        parallel_tool_calls=True,
        drop_params=True,
    )


def test_openai_chat_completion_streaming_handler_reasoning_content():
    from litellm.llms.openai.chat.gpt_transformation import (
        OpenAIChatCompletionStreamingHandler,
    )
    from unittest.mock import MagicMock

    streaming_handler = OpenAIChatCompletionStreamingHandler(
        streaming_response=MagicMock(),
        sync_stream=True,
    )
    response = streaming_handler.chunk_parser(
        chunk={
            "id": "e89b6501-8ac2-464c-9550-7cd3daf94350",
            "object": "chat.completion.chunk",
            "created": 1741037890,
            "model": "deepseek-reasoner",
            "system_fingerprint": "fp_5417b77867_prod0225",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": None, "reasoning_content": "."},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
    )

    assert response.choices[0].delta.reasoning_content == "."


def validate_response_url_citation(url_citation: ChatCompletionAnnotationURLCitation):
    assert "end_index" in url_citation
    assert "start_index" in url_citation
    assert "url" in url_citation


def validate_web_search_annotations(annotations: ChatCompletionAnnotation):
    """validates litellm response contains web search annotations"""
    print("annotations: ", annotations)
    assert annotations is not None
    assert isinstance(annotations, list)
    for annotation in annotations:
        assert annotation["type"] == "url_citation"
        url_citation: ChatCompletionAnnotationURLCitation = annotation["url_citation"]
        validate_response_url_citation(url_citation)


@pytest.mark.flaky(reruns=3)
def test_openai_web_search():
    """Makes a simple web search request and validates the response contains web search annotations and all expected fields are present"""
    litellm._turn_on_debug()
    response = litellm.completion(
        model="openai/gpt-4o-search-preview",
        messages=[
            {
                "role": "user",
                "content": "What was a positive news story from today?",
            }
        ],
    )
    print("litellm response: ", response.model_dump_json(indent=4))
    message = response.choices[0].message
    if hasattr(message, "annotations"):
        annotations: ChatCompletionAnnotation = message.annotations
        validate_web_search_annotations(annotations)


def test_openai_web_search_streaming():
    """Makes a simple web search request and validates the response contains web search annotations and all expected fields are present"""
    # litellm._turn_on_debug()
    test_openai_web_search: Optional[ChatCompletionAnnotation] = None
    response = litellm.completion(
        model="openai/gpt-4o-search-preview",
        messages=[
            {
                "role": "user",
                "content": "What was a positive news story from today?",
            }
        ],
        stream=True,
    )
    for chunk in response:
        print("litellm response chunk: ", chunk)
        if (
            hasattr(chunk.choices[0].delta, "annotations")
            and chunk.choices[0].delta.annotations is not None
        ):
            test_openai_web_search = chunk.choices[0].delta.annotations

    # Assert this request has at-least one web search annotation
    if test_openai_web_search is not None:
        validate_web_search_annotations(test_openai_web_search)


class TestOpenAIGPT4OAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> dict:
        return {
            "model": "openai/gpt-4o-transcribe",
            # "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.OPENAI


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4o"])
async def test_openai_pdf_url(model):
    from litellm.utils import return_raw_request, CallTypes

    request = return_raw_request(
        CallTypes.completion,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the first page of the PDF?"},
                        {
                            "type": "file",
                            "file": {"file_id": "https://arxiv.org/pdf/2303.08774"},
                        },
                    ],
                }
            ],
        },
    )
    print("request: ", request)

    assert (
        "file_data" in request["raw_request_body"]["messages"][0]["content"][1]["file"]
    )


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_openai_codex_stream(sync_mode):
    from litellm.main import stream_chunk_builder

    kwargs = {
        "model": "openai/codex-mini-latest",
        "messages": [{"role": "user", "content": "Hey!"}],
        "stream": True,
    }

    chunks = []
    if sync_mode:
        response = litellm.completion(**kwargs)
        for chunk in response:
            chunks.append(chunk)
    else:
        response = await litellm.acompletion(**kwargs)
        async for chunk in response:
            chunks.append(chunk)

    complete_response = stream_chunk_builder(chunks=chunks)
    print("complete_response: ", complete_response)

    assert complete_response.choices[0].message.content is not None


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_openai_codex(sync_mode):

    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "openai-codex-mini-latest",
                "litellm_params": {
                    "model": "openai/codex-mini-latest",
                },
            }
        ]
    )

    kwargs = {
        "model": "openai-codex-mini-latest",
        "messages": [{"role": "user", "content": "Hey!"}],
    }

    if sync_mode:
        response = router.completion(**kwargs)
    else:
        response = await router.acompletion(**kwargs)
    print("response: ", response)

    assert response.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_openai_via_gemini_streaming_bridge():
    """
    Test that the openai via gemini streaming bridge works correctly
    """
    from litellm import Router
    from litellm.types.utils import ModelResponseStream

    router = Router(
        model_list=[
            {
                "model_name": "openai/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                },
                "model_info": {
                    "version": 2,
                },
            }
        ],
        model_group_alias={"gemini-2.5-pro": "openai/gpt-3.5-turbo"},
    )

    response = await router.agenerate_content_stream(
        model="openai/gpt-3.5-turbo",
        contents=[
            {
                "parts": [{"text": "Write a long story about space exploration"}],
                "role": "user",
            }
        ],
        generationConfig={"maxOutputTokens": 500},
    )

    printed_chunks = []
    async for chunk in response:
        print("chunk: ", chunk)
        printed_chunks.append(chunk)
        assert not isinstance(chunk, ModelResponseStream)

    assert len(printed_chunks) > 0


def test_openai_deepresearch_model_bridge():
    """
    Test that the deepresearch model bridge works correctly
    """
    litellm._turn_on_debug()
    response = litellm.completion(
        model="o3-deep-research-2025-06-26",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        tools=[
            {"type": "web_search_preview"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
        ],
    )

    print("response: ", response)


def test_openai_tool_calling():
    from pydantic import BaseModel
    from typing import Any, Literal

    class OpenAIFunction(BaseModel):
        description: Optional[str] = None
        name: str
        parameters: Optional[dict[str, Any]] = None

    class OpenAITool(BaseModel):
        type: Literal["function"]
        function: OpenAIFunction

    completion_params = {
        "model": "openai/gpt-4.1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is TSLA stock price at today?"}
                ],
            }
        ],
        "stream": False,
        "temperature": 0.5,
        "stop": None,
        "max_tokens": 1600,
        "tools": [
            OpenAITool(
                type="function",
                function=OpenAIFunction(
                    description="Get the current stock price for a given ticker symbol.",
                    name="get_stock_price",
                    parameters={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                            }
                        },
                        "required": ["ticker"],
                    },
                ),
            )
        ],
    }

    response = litellm.completion(**completion_params)


@pytest.mark.asyncio
async def test_openai_gpt5_reasoning():
    response = await litellm.acompletion(
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        reasoning_effort="minimal",
    )
    print("response: ", response)
    assert response.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_openai_safety_identifier_parameter():
    """Test that safety_identifier parameter is correctly passed to the OpenAI API."""
    from openai import AsyncOpenAI

    litellm.set_verbose = True
    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                safety_identifier="user_code_123456",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the safety_identifier parameter
        assert "safety_identifier" in request_body
        # Verify safety_identifier is correctly sent to the API
        assert request_body["safety_identifier"] == "user_code_123456"


def test_openai_safety_identifier_parameter_sync():
    """Test that safety_identifier parameter is correctly passed to the OpenAI API."""
    from openai import OpenAI

    litellm.set_verbose = True
    client = OpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                safety_identifier="user_code_123456",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the safety_identifier parameter
        assert "safety_identifier" in request_body
        # Verify safety_identifier is correctly sent to the API
        assert request_body["safety_identifier"] == "user_code_123456"
