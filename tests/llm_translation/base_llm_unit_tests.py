import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os
from litellm._uuid import uuid
import time
import base64
import inspect

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
    ProviderConfigManager,
)
from litellm.main import stream_chunk_builder
from typing import Union
from litellm.types.utils import Usage, ModelResponse

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

    if usage.prompt_tokens_details is not None:
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

    def get_base_completion_call_args_with_reasoning_model(self) -> dict:
        """Must return the base completion call args with reasoning_effort"""
        return {}

    @pytest.fixture(autouse=True)
    def _handle_rate_limits(self):
        """Fixture to handle rate limit errors for all test methods"""
        try:
            yield
        except litellm.RateLimitError:
            pytest.skip("Rate limit exceeded")
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")

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

    def test_tool_call_with_property_type_array(self):
        litellm._turn_on_debug()
        from litellm.utils import supports_function_calling

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_function_calling(base_completion_call_args["model"], None):
            print("Model does not support function calling")
            pytest.skip("Model does not support function calling")
        base_completion_call_args = self.get_base_completion_call_args()
        response = self.completion_function(
            **base_completion_call_args,
            messages=[
                {
                    "role": "user",
                    "content": "Tell me if the shoe brand Air Jordan has more models than the shoe brand Nike.",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "shoe_get_id",
                        "description": "Get information about a show by its ID or name",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "shoe_id": {
                                    "type": ["string", "number"],
                                    "description": "The shoe ID or name",
                                }
                            },
                            "required": ["shoe_id"],
                            "additionalProperties": False,
                            "$schema": "http://json-schema.org/draft-07/schema#",
                        },
                    },
                },
            ],
        )
        print(response)
        print(json.dumps(response, indent=4, default=str))

    @pytest.mark.flaky(retries=3, delay=1)
    def test_tool_call_with_empty_enum_property(self):
        litellm._turn_on_debug()
        from litellm.utils import supports_function_calling

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_function_calling(base_completion_call_args["model"], None):
            print("Model does not support function calling")
            pytest.skip("Model does not support function calling")
        base_completion_call_args = self.get_base_completion_call_args()
        response = self.completion_function(
            **base_completion_call_args,
            messages=[
                {
                    "role": "user",
                    "content": "Search for the latest iPhone models and tell me which storage options are available.",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "litellm_product_search",
                        "description": "Search for product information and specifications.\n\nSupports filtering by category, brand, price range, and availability.\nCan retrieve detailed product specifications, pricing, and stock information.\nSupports different search modes and result formatting options.\n",
                        "parameters": {
                            "properties": {
                                "search_mode": {
                                    "default": "",
                                    "description": "The search strategy to use for finding products.",
                                    "enum": [
                                        "",
                                        "product_search",
                                        "product_search_with_filters",
                                        "product_search_with_sorting",
                                        "product_search_with_pagination",
                                        "product_search_with_aggregation",
                                    ],
                                    "title": "Search Mode",
                                    "type": "string",
                                },
                            },
                            "required": ["search_mode"],
                            "title": "product_search_arguments",
                            "type": "object",
                        },
                    },
                }
            ],
        )
        print(response)
        print(json.dumps(response, indent=4, default=str))

    def test_streaming(self):
        """Check if litellm handles streaming correctly"""
        from litellm.types.utils import ModelResponseStream
        from typing import Optional

        base_completion_call_args = self.get_base_completion_call_args()
        # litellm.set_verbose = True
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
        created_at: Optional[int] = None
        for chunk in response:
            print(chunk)
            chunks.append(chunk)
            if isinstance(chunk, ModelResponseStream):
                if created_at is None:
                    created_at = chunk.created
                assert chunk.created == created_at

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

    def test_web_search(self):
        from litellm.utils import supports_web_search

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm._turn_on_debug()

        base_completion_call_args = self.get_base_completion_call_args()

        if not supports_web_search(base_completion_call_args["model"], None):
            pytest.skip("Model does not support web search")

        response = self.completion_function(
            **base_completion_call_args,
            messages=[
                {"role": "user", "content": "What's the weather like in Boston today?"}
            ],
            web_search_options={},
            max_tokens=100,
        )

        assert response is not None

        print(f"response={response}")

    def test_url_context(self):
        from litellm.utils import supports_url_context

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm._turn_on_debug()

        base_completion_call_args = self.get_base_completion_call_args()

        if not supports_url_context(base_completion_call_args["model"], None):
            pytest.skip("Model does not support url context")

        response = self.completion_function(
            **base_completion_call_args,
            messages=[
                {
                    "role": "user",
                    "content": "Summarize the content of this URL: https://en.wikipedia.org/wiki/Artificial_intelligence",
                }
            ],
            tools=[{"urlContext": {}}],
        )

        assert response is not None
        print(f"response={response}")

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_pdf_handling(self, pdf_messages, sync_mode):
        from litellm.utils import supports_pdf_input

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm._turn_on_debug()

        image_content = [
            {"type": "text", "text": "What's this file about?"},
            {
                "type": "file",
                "file": {
                    "file_data": pdf_messages,
                },
            },
        ]

        image_messages = [{"role": "user", "content": image_content}]

        base_completion_call_args = self.get_base_completion_call_args()

        if not supports_pdf_input(base_completion_call_args["model"], None):
            pytest.skip("Model does not support image input")

        if sync_mode:
            response = self.completion_function(
                **base_completion_call_args,
                messages=image_messages,
            )
        else:
            response = await self.async_completion_function(
                **base_completion_call_args,
                messages=image_messages,
            )

        assert response is not None

    @pytest.mark.asyncio
    async def test_async_pdf_handling_with_file_id(self):
        from litellm.utils import supports_pdf_input

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm._turn_on_debug()

        image_content = [
            {"type": "text", "text": "What's this file about?"},
            {
                "type": "file",
                "file": {
                    "file_id": "https://upload.wikimedia.org/wikipedia/commons/2/20/Re_example.pdf"
                },
            },
        ]

        image_messages = [{"role": "user", "content": image_content}]

        base_completion_call_args = self.get_base_completion_call_args()

        if not supports_pdf_input(base_completion_call_args["model"], None):
            pytest.skip("Model does not support image input")

        response = await self.async_completion_function(
            **base_completion_call_args,
            messages=image_messages,
        )

        assert response is not None

    def test_file_data_unit_test(self, pdf_messages):
        from litellm.utils import supports_pdf_input, return_raw_request
        from litellm.types.utils import CallTypes
        from litellm.litellm_core_utils.prompt_templates.factory import (
            convert_to_anthropic_image_obj,
        )

        media_chunk = convert_to_anthropic_image_obj(
            openai_image_url=pdf_messages,
            format=None,
        )

        file_content = [
            {"type": "text", "text": "What's this file about?"},
            {
                "type": "file",
                "file": {
                    "file_data": pdf_messages,
                },
            },
        ]

        image_messages = [{"role": "user", "content": file_content}]

        base_completion_call_args = self.get_base_completion_call_args()

        if not supports_pdf_input(base_completion_call_args["model"], None):
            pytest.skip("Model does not support image input")

        raw_request = return_raw_request(
            endpoint=CallTypes.completion,
            kwargs={**base_completion_call_args, "messages": image_messages},
        )

        print("RAW REQUEST", raw_request)

        assert media_chunk["data"] in json.dumps(raw_request)

    def test_message_with_name(self):
        try:
            litellm.set_verbose = True
            base_completion_call_args = self.get_base_completion_call_args()
            messages = [
                {"role": "user", "content": "Hello", "name": "test_name"},
            ]
            response = self.completion_function(
                **base_completion_call_args, messages=messages
            )
            assert response is not None
        except litellm.RateLimitError:
            pass

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

        print(f"response={response}")

        # OpenAI guarantees that the JSON schema is returned in the content
        # relevant issue: https://github.com/BerriAI/litellm/issues/6741
        assert response.choices[0].message.content is not None

    @pytest.mark.parametrize(
        "response_format",
        [
            {"type": "text"},
        ],
    )
    @pytest.mark.flaky(retries=6, delay=1)
    def test_response_format_type_text_with_tool_calls_no_tool_choice(
        self, response_format
    ):
        base_completion_call_args = self.get_base_completion_call_args()
        messages = [
            {"role": "user", "content": "What's the weather like in Boston today?"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        try:
            print(f"MAKING LLM CALL")
            response = self.completion_function(
                **base_completion_call_args,
                messages=messages,
                response_format=response_format,
                tools=tools,
                drop_params=True,
            )
            print(f"RESPONSE={response}")
        except litellm.ContextWindowExceededError:
            pytest.skip("Model exceeded context window")
        assert response is not None

    def test_response_format_type_text(self):
        """
        Test that the response format type text does not lead to tool calls
        """
        from litellm import LlmProviders

        base_completion_call_args = self.get_base_completion_call_args()
        litellm.set_verbose = True

        _, provider, _, _ = litellm.get_llm_provider(
            model=base_completion_call_args["model"]
        )

        provider_config = ProviderConfigManager.get_provider_chat_config(
            base_completion_call_args["model"], LlmProviders(provider)
        )

        print(f"provider_config={provider_config}")

        translated_params = provider_config.map_openai_params(
            non_default_params={"response_format": {"type": "text"}},
            optional_params={},
            model=base_completion_call_args["model"],
            drop_params=False,
        )

        assert "tool_choice" not in translated_params
        assert (
            "tools" not in translated_params
        ), f"Got tools={translated_params['tools']}, expected no tools"

        print(f"translated_params={translated_params}")

    @pytest.mark.flaky(retries=6, delay=1)
    def test_json_response_pydantic_obj(self):
        litellm._turn_on_debug()
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
        litellm._turn_on_debug()
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
    def test_audio_input(self):
        """
        Test that audio input is supported by the LLM API
        """
        from litellm.utils import supports_audio_input

        litellm._turn_on_debug()
        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_audio_input(base_completion_call_args["model"], None):
            pytest.skip(
                f"Model={base_completion_call_args['model']} does not support audio input"
            )

        url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
        response = httpx.get(url)
        response.raise_for_status()
        wav_data = response.content
        encoded_string = base64.b64encode(wav_data).decode("utf-8")

        completion = self.completion_function(
            **base_completion_call_args,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this recording?"},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": encoded_string, "format": "wav"},
                        },
                    ],
                },
            ],
        )

        print(completion.choices[0].message)

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

    def test_image_url_string(self):
        litellm.set_verbose = True
        from litellm.utils import supports_vision

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_vision(base_completion_call_args["model"], None):
            pytest.skip("Model does not support image input")
        elif "http://" in image_url and "fireworks_ai" in base_completion_call_args.get(
            "model"
        ):
            pytest.skip("Model does not support http:// input")

        image_url_param = image_url
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": image_url_param,
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
        print("test_prompt_caching")
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

            print("response=", response)

            initial_cost = response._hidden_params["response_cost"]
            ## call 2
            response = self.completion_function(
                **base_completion_call_args,
                messages=messages,
                max_tokens=10,
            )

            time.sleep(1)

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
            if response.usage.prompt_tokens_details is not None:
                assert (
                    response.usage.prompt_tokens_details.cached_tokens > 0
                ), f"cached_tokens={response.usage.prompt_tokens_details.cached_tokens} should be greater than 0. Got usage={response.usage}"
        except litellm.InternalServerError as e:
            print("InternalServerError", e)

    @pytest.fixture
    def pdf_messages(self):
        import base64
        import os

        # Use local PDF file instead of external URL to avoid flaky tests
        test_dir = os.path.dirname(__file__)
        pdf_path = os.path.join(test_dir, "fixtures", "dummy.pdf")
        
        with open(pdf_path, "rb") as f:
            file_data = f.read()

        encoded_file = base64.b64encode(file_data).decode("utf-8")
        url = f"data:application/pdf;base64,{encoded_file}"

        return url

    @pytest.mark.flaky(retries=3, delay=1)
    def test_empty_tools(self):
        """
        Related Issue: https://github.com/BerriAI/litellm/issues/9080
        """
        try:
            from litellm import completion, ModelResponse

            litellm.set_verbose = True
            litellm._turn_on_debug()
            from litellm.utils import supports_function_calling

            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
            litellm.model_cost = litellm.get_model_cost_map(url="")

            base_completion_call_args = self.get_base_completion_call_args()
            if not supports_function_calling(base_completion_call_args["model"], None):
                print("Model does not support function calling")
                pytest.skip("Model does not support function calling")

            response = completion(
                **base_completion_call_args,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                tools=[],
            )  # just make sure call doesn't fail
            print("response: ", response)
            assert response is not None
        except litellm.ContentPolicyViolationError:
            pass
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")
        except litellm.RateLimitError:
            pass
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")

    @pytest.mark.flaky(retries=3, delay=1)
    def test_basic_tool_calling(self):
        try:
            from litellm import completion, ModelResponse

            litellm.set_verbose = True
            litellm._turn_on_debug()
            from litellm.utils import supports_function_calling

            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
            litellm.model_cost = litellm.get_model_cost_map(url="")

            base_completion_call_args = self.get_base_completion_call_args()
            if not supports_function_calling(base_completion_call_args["model"], None):
                print("Model does not support function calling")
                pytest.skip("Model does not support function calling")

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ]
            messages = [
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today in fahrenheit?",
                }
            ]
            request_args = {
                "messages": messages,
                "tools": tools,
            }
            request_args.update(self.get_base_completion_call_args())
            response: ModelResponse = completion(**request_args)  # type: ignore
            print(f"response: {response}")

            assert response is not None

            # if the provider did not return any tool calls do not make a subsequent llm api call
            if response.choices[0].message.content is not None:
                try:
                    json.loads(response.choices[0].message.content)
                    pytest.fail(f"Tool call returned in content instead of tool_calls")
                except Exception as e:
                    print(f"Error: {e}")
                    pass
                if (
                    "<thinking>" in response.choices[0].message.content
                    and "</thinking>" in response.choices[0].message.content
                ):
                    pytest.fail(
                        "Thinking block returned in content instead of separate reasoning_content"
                    )
            if response.choices[0].message.tool_calls is None:
                return
            # Add any assertions here to check the response

            assert isinstance(
                response.choices[0].message.tool_calls[0].function.name, str
            )
            assert isinstance(
                response.choices[0].message.tool_calls[0].function.arguments, str
            )
            assert (
                response.choices[0].finish_reason == "tool_calls"
            ), f"finish_reason: {response.choices[0].finish_reason}, expected: tool_calls"
            messages.append(
                response.choices[0].message.model_dump()
            )  # Add assistant tool invokes
            tool_result = (
                '{"location": "Boston", "temperature": "72", "unit": "fahrenheit"}'
            )
            # Add user submitted tool results in the OpenAI format
            messages.append(
                {
                    "tool_call_id": response.choices[0].message.tool_calls[0].id,
                    "role": "tool",
                    "name": response.choices[0].message.tool_calls[0].function.name,
                    "content": tool_result,
                }
            )
            # In the second response, Claude should deduce answer from tool results
            request_2_args = {
                "messages": messages,
                "tools": tools,
            }
            request_2_args.update(self.get_base_completion_call_args())
            second_response: ModelResponse = completion(**request_2_args)  # type: ignore
            print(f"second response: {second_response}")
            assert second_response is not None

            # either content or tool calls should be present
            assert (
                second_response.choices[0].message.content is not None
                or second_response.choices[0].message.tool_calls is not None
            )
        except litellm.ServiceUnavailableError:
            pytest.skip("Model is overloaded")
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")
        except litellm.RateLimitError:
            pass
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")

    @pytest.mark.flaky(retries=3, delay=1)
    @pytest.mark.asyncio
    async def test_completion_cost(self):
        from litellm import completion_cost

        litellm._turn_on_debug()

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm.set_verbose = True
        response = await self.async_completion_function(
            **self.get_base_completion_call_args(),
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )
        print(response._hidden_params["response_cost"])

        assert response._hidden_params["response_cost"] > 0

    @pytest.mark.parametrize("input_type", ["input_audio", "audio_url"])
    @pytest.mark.parametrize("format_specified", [True])
    def test_supports_audio_input(self, input_type, format_specified):
        from litellm.utils import return_raw_request, supports_audio_input
        from litellm.types.utils import CallTypes

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm.drop_params = True
        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_audio_input(base_completion_call_args["model"], None):
            print("Model does not support audio input")
            pytest.skip("Model does not support audio input")

        url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
        response = httpx.get(url)
        response.raise_for_status()
        wav_data = response.content
        audio_format = "wav"
        encoded_string = base64.b64encode(wav_data).decode("utf-8")

        audio_content = [{"type": "text", "text": "What is in this recording?"}]

        test_file_id = "gs://bucket/file.wav"

        if input_type == "input_audio":
            audio_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": encoded_string, "format": audio_format},
                }
            )
        elif input_type == "audio_url":
            audio_content.append(
                {
                    "type": "file",
                    "file": {
                        "file_id": test_file_id,
                        "filename": "my-sample-audio-file",
                    },
                }
            )

        raw_request = return_raw_request(
            endpoint=CallTypes.completion,
            kwargs={
                **base_completion_call_args,
                "modalities": ["text", "audio"],
                "audio": {"voice": "alloy", "format": audio_format},
                "messages": [
                    {
                        "role": "user",
                        "content": audio_content,
                    },
                ],
            },
        )
        print("raw_request: ", raw_request)

        if input_type == "input_audio":
            assert encoded_string in json.dumps(
                raw_request
            ), "Audio data not sent to gemini"
        elif input_type == "audio_url":
            assert test_file_id in json.dumps(
                raw_request
            ), "Audio URL not sent to gemini"

    def test_function_calling_with_tool_response(self):
        from litellm.utils import supports_function_calling
        from litellm import completion

        litellm._turn_on_debug()
        try:

            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
            litellm.model_cost = litellm.get_model_cost_map(url="")

            base_completion_call_args = self.get_base_completion_call_args()
            if not supports_function_calling(base_completion_call_args["model"], None):
                print("Model does not support function calling")
                pytest.skip("Model does not support function calling")

            def get_weather(city: str):
                return f"City: {city}, Weather: Sunny with 34 degree Celcius"

            TOOLS = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather in a city",
                        "parameters": {
                            "$id": "https://some/internal/name",
                            "$schema": "https://json-schema.org/draft-07/schema",
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The city to get the weather for",
                                }
                            },
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                }
            ]

            messages = [{"content": "How is the weather in Mumbai?", "role": "user"}]
            response, iteration = "", 0
            while True:
                if response:
                    break
                # Create a streaming response with tool calling enabled
                stream = completion(
                    **base_completion_call_args,
                    messages=messages,
                    tools=TOOLS,
                    stream=True,
                )

                final_tool_calls = {}
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    print(delta)
                    if delta.content:
                        response += delta.content
                    elif delta.tool_calls:
                        for tool_call in chunk.choices[0].delta.tool_calls or []:
                            index = tool_call.index
                            if index not in final_tool_calls:
                                final_tool_calls[index] = tool_call
                            else:
                                final_tool_calls[
                                    index
                                ].function.arguments += tool_call.function.arguments
                if final_tool_calls:
                    for tool_call in final_tool_calls.values():
                        if tool_call.function.name == "get_weather":
                            city = json.loads(tool_call.function.arguments)["city"]
                            tool_response = get_weather(city)
                            messages.append(
                                {
                                    "role": "assistant",
                                    "tool_calls": [tool_call],
                                    "content": None,
                                }
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_response,
                                }
                            )
                iteration += 1
                if iteration > 2:
                    print("Something went wrong!")
                    break

            print(response)
        except litellm.ServiceUnavailableError:
            pass

    def test_reasoning_effort(self):
        """Test that reasoning_effort is passed correctly to the model"""
        from litellm.utils import supports_reasoning
        from litellm import completion

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        base_completion_call_args = (
            self.get_base_completion_call_args_with_reasoning_model()
        )
        if len(base_completion_call_args) == 0:
            print("base_completion_call_args is empty")
            pytest.skip("Model does not support reasoning")
        if not supports_reasoning(base_completion_call_args["model"], None):
            print("Model does not support reasoning")
            pytest.skip("Model does not support reasoning")

        _, provider, _, _ = litellm.get_llm_provider(
            model=base_completion_call_args["model"]
        )

        ## CHECK PARAM MAPPING
        optional_params = get_optional_params(
            model=base_completion_call_args["model"],
            custom_llm_provider=provider,
            reasoning_effort="high",
        )
        # either accepts reasoning effort or thinking budget
        assert "reasoning_effort" in optional_params or "4096" in json.dumps(
            optional_params
        )

        try:
            litellm._turn_on_debug()
            response = completion(
                **base_completion_call_args,
                reasoning_effort="low",
                messages=[{"role": "user", "content": "Hello!"}],
            )
            print(f"response: {response}")
        except Exception as e:
            pytest.fail(f"Error: {e}")


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


class BaseAnthropicChatTest(ABC):
    """
    Ensures consistent result across anthropic model usage
    """

    @abstractmethod
    def get_base_completion_call_args(self) -> dict:
        """Must return the base completion call args"""
        pass

    @abstractmethod
    def get_base_completion_call_args_with_thinking(self) -> dict:
        """Must return the base completion call args"""
        pass

    @property
    def completion_function(self):
        return litellm.completion

    def test_anthropic_response_format_streaming_vs_non_streaming(self):
        args = {
            "messages": [
                {
                    "content": "Your goal is to summarize the previous agent's thinking process into short descriptions to let user better understand the research progress. If no information is available, just say generic phrase like 'Doing some research...' with the given output format. Make sure to adhere to the output format no matter what, even if you don't have any information or you are not allowed to respond to the given input information (then just say generic phrase like 'Doing some research...').",
                    "role": "system",
                },
                {
                    "role": "user",
                    "content": "Here is the input data (previous agent's output): \n\n Let's try to refine our search further, focusing more on the technical aspects of home automation and home energy system management:",
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "final_output",
                    "strict": True,
                    "schema": {
                        "description": 'Progress report for the thinking process\n\nThis model represents a snapshot of the agent\'s current progress during\nthe thinking process, providing a brief description of the current activity.\n\nAttributes:\n    agent_doing: Brief description of what the agent is currently doing.\n                Should be kept under 10 words. Example: "Learning about home automation"',
                        "properties": {
                            "agent_doing": {"title": "Agent Doing", "type": "string"}
                        },
                        "required": ["agent_doing"],
                        "title": "ThinkingStep",
                        "type": "object",
                        "additionalProperties": False,
                    },
                },
            },
        }

        base_completion_call_args = self.get_base_completion_call_args()

        response = self.completion_function(
            **base_completion_call_args, **args, stream=True
        )

        chunks = []
        for chunk in response:
            print(f"chunk: {chunk}")
            chunks.append(chunk)

        print(f"chunks: {chunks}")
        built_response = stream_chunk_builder(chunks=chunks)

        non_stream_response = self.completion_function(
            **base_completion_call_args, **args, stream=False
        )

        print(
            "built_response.choices[0].message.content",
            built_response.choices[0].message.content,
        )
        print(
            "non_stream_response.choices[0].message.content",
            non_stream_response.choices[0].message.content,
        )
        assert (
            json.loads(built_response.choices[0].message.content).keys()
            == json.loads(non_stream_response.choices[0].message.content).keys()
        ), f"Got={json.loads(built_response.choices[0].message.content)}, Expected={json.loads(non_stream_response.choices[0].message.content)}"

    def test_completion_thinking_with_response_format(self):
        from pydantic import BaseModel

        litellm._turn_on_debug()

        class RFormat(BaseModel):
            question: str
            answer: str

        base_completion_call_args = self.get_base_completion_call_args_with_thinking()

        messages = [{"role": "user", "content": "Generate 5 question + answer pairs"}]
        response = self.completion_function(
            **base_completion_call_args,
            messages=messages,
            response_format=RFormat,
        )

        print(response)

    def test_completion_thinking_with_max_tokens(self):
        from pydantic import BaseModel

        litellm._turn_on_debug()

        base_completion_call_args = self.get_base_completion_call_args_with_thinking()

        messages = [{"role": "user", "content": "Generate 5 question + answer pairs"}]
        response = self.completion_function(
            **base_completion_call_args,
            messages=messages,
            max_completion_tokens=20000,
        )

        print(response)

    def test_completion_thinking_without_max_tokens(self):
        from pydantic import BaseModel

        litellm._turn_on_debug()

        base_completion_call_args = self.get_base_completion_call_args_with_thinking()

        messages = [{"role": "user", "content": "Generate 5 question + answer pairs"}]
        response = self.completion_function(
            **base_completion_call_args,
            messages=messages,
        )

        print(response)

    def test_completion_with_thinking_basic(self):
        litellm._turn_on_debug()
        base_completion_call_args = self.get_base_completion_call_args_with_thinking()

        messages = [{"role": "user", "content": "Generate 5 question + answer pairs"}]
        response = self.completion_function(
            **base_completion_call_args,
            messages=messages,
        )

        print(f"response: {response}")
        assert response.choices[0].message.reasoning_content is not None
        assert isinstance(response.choices[0].message.reasoning_content, str)
        assert response.choices[0].message.thinking_blocks is not None
        assert isinstance(response.choices[0].message.thinking_blocks, list)
        assert len(response.choices[0].message.thinking_blocks) > 0

        assert response.choices[0].message.thinking_blocks[0]["signature"] is not None

    def test_anthropic_thinking_output_stream(self):
        # litellm.set_verbose = True
        try:
            base_completion_call_args = (
                self.get_base_completion_call_args_with_thinking()
            )
            resp = litellm.completion(
                **base_completion_call_args,
                messages=[{"role": "user", "content": "Tell me a joke."}],
                stream=True,
                timeout=10,
            )

            reasoning_content_exists = False
            signature_block_exists = False
            tool_call_exists = False
            for chunk in resp:
                print(f"chunk 2: {chunk}")
                if chunk.choices[0].delta.tool_calls:
                    tool_call_exists = True
                if (
                    hasattr(chunk.choices[0].delta, "thinking_blocks")
                    and chunk.choices[0].delta.thinking_blocks is not None
                    and chunk.choices[0].delta.reasoning_content is not None
                    and isinstance(chunk.choices[0].delta.thinking_blocks, list)
                    and len(chunk.choices[0].delta.thinking_blocks) > 0
                    and isinstance(chunk.choices[0].delta.reasoning_content, str)
                ):
                    reasoning_content_exists = True
                    print(chunk.choices[0].delta.thinking_blocks[0])
                    if chunk.choices[0].delta.thinking_blocks[0].get("signature"):
                        signature_block_exists = True
            assert not tool_call_exists
            assert reasoning_content_exists
            assert signature_block_exists
        except litellm.Timeout:
            pytest.skip("Model is timing out")

    def test_anthropic_reasoning_effort_thinking_translation(self):
        base_completion_call_args = self.get_base_completion_call_args_with_thinking()
        _, provider, _, _ = litellm.get_llm_provider(
            model=base_completion_call_args["model"]
        )

        optional_params = get_optional_params(
            model=base_completion_call_args.get("model"),
            custom_llm_provider=provider,
            reasoning_effort="high",
        )
        assert optional_params["thinking"] == {"type": "enabled", "budget_tokens": 4096}

        assert "reasoning_effort" not in optional_params


class BaseReasoningLLMTests(ABC):
    """
    Base class for testing reasoning llms

    - test that the responses contain reasoning_content
    - test that the usage contains reasoning_tokens
    """

    @abstractmethod
    def get_base_completion_call_args(self) -> dict:
        """Must return the base completion call args"""
        pass

    @property
    def completion_function(self):
        return litellm.completion

    def test_non_streaming_reasoning_effort(self):
        """
        Base test for non-streaming reasoning effort

        - Assert that `reasoning_content` is not None from response message
        - Assert that `reasoning_tokens` is greater than 0 from usage
        """
        litellm._turn_on_debug()
        base_completion_call_args = self.get_base_completion_call_args()
        response: ModelResponse = self.completion_function(
            **base_completion_call_args, reasoning_effort="low"
        )

        # user gets `reasoning_content` in the response message
        assert response.choices[0].message.reasoning_content is not None
        assert isinstance(response.choices[0].message.reasoning_content, str)

        # user get `reasoning_tokens`
        assert response.usage.completion_tokens_details.reasoning_tokens > 0

    def test_streaming_reasoning_effort(self):
        """
        Base test for streaming reasoning effort

        - Assert that `reasoning_content` is not None from streaming response
        - Assert that `reasoning_tokens` is greater than 0 from usage
        """
        # litellm._turn_on_debug()
        base_completion_call_args = self.get_base_completion_call_args()
        response: CustomStreamWrapper = self.completion_function(
            **base_completion_call_args,
            reasoning_effort="low",
            stream=True,
            stream_options={"include_usage": True},
        )

        resoning_content: str = ""
        usage: Usage = None
        for chunk in response:
            print(chunk)
            if hasattr(chunk.choices[0].delta, "reasoning_content"):
                resoning_content += chunk.choices[0].delta.reasoning_content
            if hasattr(chunk, "usage"):
                usage = chunk.usage

        assert resoning_content is not None
        assert len(resoning_content) > 0

        print(f"usage: {usage}")
        assert usage.completion_tokens_details.reasoning_tokens > 0
