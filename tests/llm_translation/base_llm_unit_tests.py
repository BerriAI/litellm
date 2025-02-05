import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os
import uuid

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import (
    CustomStreamWrapper,
    get_supported_openai_params,
    get_optional_params,
)
from typing import Union

# test_example.py
from abc import ABC, abstractmethod
from openai import OpenAI


def _usage_format_tests(usage: litellm.Usage):
    """
    OpenAI prompt caching
    - prompt_tokens = sum of non-cache hit tokens + cache-hit tokens
    - total_tokens = prompt_tokens + completion_tokens

    Example
    ```
    "usage": {
        "prompt_tokens": 2006,
        "completion_tokens": 300,
        "total_tokens": 2306,
        "prompt_tokens_details": {
            "cached_tokens": 1920
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0
        }
        # ANTHROPIC_ONLY #
        "cache_creation_input_tokens": 0
    }
    ```
    """
    print(f"usage={usage}")
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    assert usage.prompt_tokens > usage.prompt_tokens_details.cached_tokens


class BaseLLMChatTest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """

    @property
    def completion_function(self):
        return litellm.completion

    @property
    def async_completion_function(self):
        return litellm.acompletion

    @abstractmethod
    def get_base_completion_call_args(self) -> dict:
        """Must return the base completion call args"""
        pass

    def test_developer_role_translation(self):
        """
        Test that the developer role is translated correctly for non-OpenAI providers.

        Translate `developer` role to `system` role for non-OpenAI providers.
        """
        base_completion_call_args = self.get_base_completion_call_args()
        messages = [
            {
                "role": "developer",
                "content": "Be a good bot!",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}],
            },
        ]
        try:
            response = self.completion_function(
                **base_completion_call_args,
                messages=messages,
            )
            assert response is not None
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")

        assert response.choices[0].message.content is not None

    def test_content_list_handling(self):
        """Check if content list is supported by LLM API"""
        base_completion_call_args = self.get_base_completion_call_args()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}],
            }
        ]
        try:
            response = self.completion_function(
                **base_completion_call_args,
                messages=messages,
            )
            assert response is not None
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")

        # for OpenAI the content contains the JSON schema, so we need to assert that the content is not None
        assert response.choices[0].message.content is not None

    def test_streaming(self):
        """Check if litellm handles streaming correctly"""
        base_completion_call_args = self.get_base_completion_call_args()
        litellm.set_verbose = True
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}],
            }
        ]
        try:
            response = self.completion_function(
                **base_completion_call_args,
                messages=messages,
                stream=True,
            )
            assert response is not None
            assert isinstance(response, CustomStreamWrapper)
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")

        # for OpenAI the content contains the JSON schema, so we need to assert that the content is not None
        chunks = []
        for chunk in response:
            print(chunk)
            chunks.append(chunk)

        resp = litellm.stream_chunk_builder(chunks=chunks)
        print(resp)

        # assert resp.usage.prompt_tokens > 0
        # assert resp.usage.completion_tokens > 0
        # assert resp.usage.total_tokens > 0

    def test_pydantic_model_input(self):
        litellm.set_verbose = True

        from litellm import completion, Message

        base_completion_call_args = self.get_base_completion_call_args()
        messages = [Message(content="Hello, how are you?", role="user")]

        self.completion_function(**base_completion_call_args, messages=messages)

    @pytest.mark.parametrize("image_url", ["str", "dict"])
    def test_pdf_handling(self, pdf_messages, image_url):
        from litellm.utils import supports_pdf_input

        if image_url == "str":
            image_url = pdf_messages
        elif image_url == "dict":
            image_url = {"url": pdf_messages}

        image_content = [
            {"type": "text", "text": "What's this file about?"},
            {
                "type": "image_url",
                "image_url": image_url,
            },
        ]

        image_messages = [{"role": "user", "content": image_content}]

        base_completion_call_args = self.get_base_completion_call_args()

        if not supports_pdf_input(base_completion_call_args["model"], None):
            pytest.skip("Model does not support image input")

        response = self.completion_function(
            **base_completion_call_args,
            messages=image_messages,
        )
        assert response is not None

    def test_message_with_name(self):
        litellm.set_verbose = True
        base_completion_call_args = self.get_base_completion_call_args()
        messages = [
            {"role": "user", "content": "Hello", "name": "test_name"},
        ]
        response = self.completion_function(
            **base_completion_call_args, messages=messages
        )
        assert response is not None

    @pytest.mark.parametrize(
        "response_format",
        [
            {"type": "json_object"},
            {"type": "text"},
        ],
    )
    @pytest.mark.flaky(retries=6, delay=1)
    def test_json_response_format(self, response_format):
        """
        Test that the JSON response format is supported by the LLM API
        """
        from litellm.utils import supports_response_schema

        base_completion_call_args = self.get_base_completion_call_args()
        litellm.set_verbose = True

        if not supports_response_schema(base_completion_call_args["model"], None):
            pytest.skip("Model does not support response schema")

        messages = [
            {
                "role": "system",
                "content": "Your output should be a JSON object with no additional properties.  ",
            },
            {
                "role": "user",
                "content": "Respond with this in json. city=San Francisco, state=CA, weather=sunny, temp=60",
            },
        ]

        response = self.completion_function(
            **base_completion_call_args,
            messages=messages,
            response_format=response_format,
        )

        print(response)

        # OpenAI guarantees that the JSON schema is returned in the content
        # relevant issue: https://github.com/BerriAI/litellm/issues/6741
        assert response.choices[0].message.content is not None

    @pytest.mark.flaky(retries=6, delay=1)
    def test_json_response_pydantic_obj(self):
        litellm.set_verbose = True
        from pydantic import BaseModel
        from litellm.utils import supports_response_schema

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        class TestModel(BaseModel):
            first_response: str

        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_response_schema(base_completion_call_args["model"], None):
            pytest.skip("Model does not support response schema")

        try:
            res = self.completion_function(
                **base_completion_call_args,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "What is the capital of France?",
                    },
                ],
                response_format=TestModel,
                timeout=5,
            )
            assert res is not None

            print(res.choices[0].message)

            assert res.choices[0].message.content is not None
            assert res.choices[0].message.tool_calls is None
        except litellm.Timeout:
            pytest.skip("Model took too long to respond")
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")

    @pytest.mark.flaky(retries=6, delay=1)
    def test_json_response_pydantic_obj_nested_obj(self):
        litellm.set_verbose = True
        from pydantic import BaseModel
        from litellm.utils import supports_response_schema

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

    @pytest.mark.flaky(retries=6, delay=1)
    def test_json_response_nested_pydantic_obj(self):
        from pydantic import BaseModel
        from litellm.utils import supports_response_schema

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]

        class EventsList(BaseModel):
            events: list[CalendarEvent]

        messages = [
            {"role": "user", "content": "List 5 important events in the XIX century"}
        ]

        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_response_schema(base_completion_call_args["model"], None):
            pytest.skip(
                f"Model={base_completion_call_args['model']} does not support response schema"
            )

        try:
            res = self.completion_function(
                **base_completion_call_args,
                messages=messages,
                response_format=EventsList,
                timeout=60,
            )
            assert res is not None

            print(res.choices[0].message)

            assert res.choices[0].message.content is not None
            assert res.choices[0].message.tool_calls is None
        except litellm.Timeout:
            pytest.skip("Model took too long to respond")
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")

    @pytest.mark.flaky(retries=6, delay=1)
    def test_json_response_nested_json_schema(self):
        """
        PROD Test: ensure nested json schema sent to proxy works as expected.
        """
        from pydantic import BaseModel
        from litellm.utils import supports_response_schema
        from litellm.llms.base_llm.base_utils import type_to_response_format_param

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]

        class EventsList(BaseModel):
            events: list[CalendarEvent]

        response_format = type_to_response_format_param(EventsList)

        messages = [
            {"role": "user", "content": "List 5 important events in the XIX century"}
        ]

        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_response_schema(base_completion_call_args["model"], None):
            pytest.skip(
                f"Model={base_completion_call_args['model']} does not support response schema"
            )

        try:
            res = self.completion_function(
                **base_completion_call_args,
                messages=messages,
                response_format=response_format,
                timeout=60,
            )
            assert res is not None

            print(res.choices[0].message)

            assert res.choices[0].message.content is not None
            assert res.choices[0].message.tool_calls is None
        except litellm.Timeout:
            pytest.skip("Model took too long to respond")
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")

    @pytest.mark.flaky(retries=6, delay=1)
    def test_json_response_format_stream(self):
        """
        Test that the JSON response format with streaming is supported by the LLM API
        """
        from litellm.utils import supports_response_schema

        base_completion_call_args = self.get_base_completion_call_args()
        litellm.set_verbose = True

        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_response_schema(base_completion_call_args["model"], None):
            pytest.skip("Model does not support response schema")

        messages = [
            {
                "role": "system",
                "content": "Your output should be a JSON object with no additional properties.  ",
            },
            {
                "role": "user",
                "content": "Respond with this in json. city=San Francisco, state=CA, weather=sunny, temp=60",
            },
        ]

        try:
            response = self.completion_function(
                **base_completion_call_args,
                messages=messages,
                response_format={"type": "json_object"},
                stream=True,
            )
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")

        print(response)

        content = ""
        for chunk in response:
            content += chunk.choices[0].delta.content or ""

        print(f"content={content}<END>")

        # OpenAI guarantees that the JSON schema is returned in the content
        # relevant issue: https://github.com/BerriAI/litellm/issues/6741
        # we need to assert that the JSON schema was returned in the content, (for Anthropic we were returning it as part of the tool call)
        assert content is not None
        assert len(content) > 0

    @pytest.fixture
    def tool_call_no_arguments(self):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2c384bc6-de46-4f29-8adc-60dd5805d305",
                    "function": {"name": "Get-FAQ", "arguments": "{}"},
                    "type": "function",
                }
            ],
        }

    @abstractmethod
    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    @pytest.mark.parametrize("detail", [None, "low", "high"])
    @pytest.mark.parametrize(
        "image_url",
        [
            "http://img1.etsystatic.com/260/0/7813604/il_fullxfull.4226713999_q86e.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        ],
    )
    @pytest.mark.flaky(retries=4, delay=2)
    def test_image_url(self, detail, image_url):
        litellm.set_verbose = True
        from litellm.utils import supports_vision

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_vision(base_completion_call_args["model"], None):
            pytest.skip("Model does not support image input")
        elif "http://" in image_url and "fireworks_ai" in base_completion_call_args.get(
            "model"
        ):
            pytest.skip("Model does not support http:// input")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ]

        if detail is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://www.gstatic.com/webp/gallery/1.webp",
                                "detail": detail,
                            },
                        },
                    ],
                }
            ]
        try:
            response = self.completion_function(
                **base_completion_call_args, messages=messages
            )
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")

        assert response is not None

    @pytest.mark.flaky(retries=4, delay=1)
    def test_prompt_caching(self):
        litellm.set_verbose = True
        from litellm.utils import supports_prompt_caching

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_prompt_caching(base_completion_call_args["model"], None):
            print("Model does not support prompt caching")
            pytest.skip("Model does not support prompt caching")

        uuid_str = str(uuid.uuid4())
        messages = [
            # System Message
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the full text of a complex legal agreement {}".format(
                            uuid_str
                        )
                        * 400,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What are the key terms and conditions in this agreement?",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
            },
            # The final turn is marked with cache-control, for continuing in followups.
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What are the key terms and conditions in this agreement?",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ]

        try:
            ## call 1
            response = self.completion_function(
                **base_completion_call_args,
                messages=messages,
                max_tokens=10,
            )

            initial_cost = response._hidden_params["response_cost"]
            ## call 2
            response = self.completion_function(
                **base_completion_call_args,
                messages=messages,
                max_tokens=10,
            )

            cached_cost = response._hidden_params["response_cost"]

            assert (
                cached_cost <= initial_cost
            ), "Cached cost={} should be less than initial cost={}".format(
                cached_cost, initial_cost
            )

            _usage_format_tests(response.usage)

            print("response=", response)
            print("response.usage=", response.usage)

            _usage_format_tests(response.usage)

            assert "prompt_tokens_details" in response.usage
            assert response.usage.prompt_tokens_details.cached_tokens > 0
        except litellm.InternalServerError:
            pass

    @pytest.fixture
    def pdf_messages(self):
        import base64

        import requests

        # URL of the file
        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

        response = requests.get(url)
        file_data = response.content

        encoded_file = base64.b64encode(file_data).decode("utf-8")
        url = f"data:application/pdf;base64,{encoded_file}"

        return url

    @pytest.mark.asyncio
    async def test_completion_cost(self):
        from litellm import completion_cost

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm.set_verbose = True
        response = await self.async_completion_function(
            **self.get_base_completion_call_args(),
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )
        print(response._hidden_params)
        cost = completion_cost(response)

        assert cost > 0


class BaseOSeriesModelsTest(ABC):  # test across azure/openai
    @abstractmethod
    def get_base_completion_call_args(self):
        pass

    @abstractmethod
    def get_client(self) -> OpenAI:
        pass

    def test_reasoning_effort(self):
        """Test that reasoning_effort is passed correctly to the model"""

        from litellm import completion

        client = self.get_client()

        completion_args = self.get_base_completion_call_args()

        with patch.object(
            client.chat.completions.with_raw_response, "create"
        ) as mock_client:
            try:
                completion(
                    **completion_args,
                    reasoning_effort="low",
                    messages=[{"role": "user", "content": "Hello!"}],
                    client=client,
                )
            except Exception as e:
                print(f"Error: {e}")

            mock_client.assert_called_once()
            request_body = mock_client.call_args.kwargs
            print("request_body: ", request_body)
            assert request_body["reasoning_effort"] == "low"

    def test_developer_role_translation(self):
        """Test that developer role is translated correctly to system role for non-OpenAI providers"""
        from litellm import completion

        client = self.get_client()

        completion_args = self.get_base_completion_call_args()

        with patch.object(
            client.chat.completions.with_raw_response, "create"
        ) as mock_client:
            try:
                completion(
                    **completion_args,
                    reasoning_effort="low",
                    messages=[
                        {"role": "developer", "content": "Be a good bot!"},
                        {"role": "user", "content": "Hello!"},
                    ],
                    client=client,
                )
            except Exception as e:
                print(f"Error: {e}")

            mock_client.assert_called_once()
            request_body = mock_client.call_args.kwargs
            print("request_body: ", request_body)
            assert (
                request_body["messages"][0]["role"] == "developer"
            ), "Got={} instead of system".format(request_body["messages"][0]["role"])
            assert request_body["messages"][0]["content"] == "Be a good bot!"

    def test_completion_o_series_models_temperature(self):
        """
        Test that temperature is not passed to O-series models
        """
        try:
            from litellm import completion

            client = self.get_client()

            completion_args = self.get_base_completion_call_args()

            with patch.object(
                client.chat.completions.with_raw_response, "create"
            ) as mock_client:
                try:
                    completion(
                        **completion_args,
                        temperature=0.0,
                        messages=[
                            {
                                "role": "user",
                                "content": "Hello, world!",
                            }
                        ],
                        drop_params=True,
                        client=client,
                    )
                except Exception as e:
                    print(f"Error: {e}")

            mock_client.assert_called_once()
            request_body = mock_client.call_args.kwargs
            print("request_body: ", request_body)
            assert (
                "temperature" not in request_body
            ), "temperature should not be in the request body"
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")
