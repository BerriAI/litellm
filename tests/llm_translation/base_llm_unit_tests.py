import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock
import os
import inspect

import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import (
    CustomStreamWrapper,
    get_supported_openai_params,
)
from typing import Union
from litellm.types.utils import Usage, ModelResponse

# test_example.py
from abc import ABC, abstractmethod
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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

    def test_web_search(self):
        from litellm.utils import supports_web_search

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm.turn_on_debug()

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

        litellm.turn_on_debug()

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

        litellm.turn_on_debug()

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

        litellm.turn_on_debug()

        image_content = [
            {"type": "text", "text": "What's this file about?"},
            {
                "type": "file",
                "file": {
                    "file_id": "https://cdn.jsdelivr.net/gh/BerriAI/litellm@d769e81c90d453240c61fc572cdb27fae06a89d0/tests/llm_translation/fixtures/dummy.pdf"
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
    def test_basic_tool_calling(self):
        try:
            from litellm import completion, ModelResponse

            litellm.set_verbose = True
            litellm.turn_on_debug()
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

    def test_function_calling_with_tool_response(self):
        from litellm.utils import supports_function_calling
        from litellm import completion

        litellm.turn_on_debug()
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

class BaseOSeriesModelsTest(ABC):  # test across azure/openai
    @abstractmethod
    def get_base_completion_call_args(self):
        pass

    @abstractmethod
    def get_client(self) -> OpenAI:
        pass

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
        litellm.turn_on_debug()
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
        # litellm.turn_on_debug()
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
