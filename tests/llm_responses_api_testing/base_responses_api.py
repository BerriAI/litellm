import httpx
import json
import pytest
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch
import os
from litellm._uuid import uuid
import time
import base64

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from abc import ABC, abstractmethod

from litellm.integrations.custom_logger import CustomLogger
import json
from litellm.types.utils import StandardLoggingPayload
from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponseAPIUsage,
    IncompleteDetails,
)
from openai.types.responses.response_create_params import (
    ResponseInputParam,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


def validate_responses_api_response(response, final_chunk: bool = False):
    """
    Validate that a response from litellm.responses() or litellm.aresponses()
    conforms to the expected ResponsesAPIResponse structure.

    Args:
        response: The response object to validate

    Raises:
        AssertionError: If the response doesn't match the expected structure
    """
    # Validate response structure
    print("response=", json.dumps(response, indent=4, default=str))
    assert isinstance(
        response, ResponsesAPIResponse
    ), "Response should be an instance of ResponsesAPIResponse"

    # Required fields
    assert "id" in response and isinstance(
        response["id"], str
    ), "Response should have a string 'id' field"
    assert "created_at" in response and isinstance(
        response["created_at"], int
    ), "Response should have an integer 'created_at' field"
    if response.get("status") == "completed":
        assert "output" in response and isinstance(
            response["output"], list
        ), "Response should have a list 'output' field"

    # Optional fields with their expected types
    optional_fields = {
        "error": (dict, type(None)),  # error can be dict or None
        "incomplete_details": (IncompleteDetails, type(None)),
        "instructions": (str, type(None)),
        "metadata": dict,
        "model": str,
        "object": str,
        "parallel_tool_calls": (bool, type(None)),
        "temperature": (int, float, type(None)),
        "tool_choice": (dict, str, type(None)),
        "tools": (list, type(None)),
        "top_p": (int, float, type(None)),
        "max_output_tokens": (int, type(None)),
        "previous_response_id": (str, type(None)),
        "reasoning": (dict, type(None)),
        "status": str,
        "text": dict,
        "truncation": (str, type(None)),
        "usage": ResponseAPIUsage,
        "user": (str, type(None)),
        "store": (bool, type(None)),
    }
    if final_chunk is False:
        optional_fields["usage"] = type(None)

    for field, expected_type in optional_fields.items():
        if field in response:
            assert isinstance(
                response[field], expected_type
            ), f"Field '{field}' should be of type {expected_type}, but got {type(response[field])}"

    # Check if output has at least one item
    if final_chunk is True and response.get("status") == "completed":
        assert (
            len(response["output"]) > 0
        ), "Response 'output' field should have at least one item"

    return True  # Return True if validation passes


class BaseResponsesAPITest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """

    @abstractmethod
    def get_base_completion_call_args(self) -> dict:
        """Must return the base completion call args"""
        pass

    def get_base_completion_reasoning_call_args(self) -> dict:
        """Must return the base completion reasoning call args"""
        return None

    def get_advanced_model_for_shell_tool(self) -> Optional[str]:
        """If specified, overrides the model used by test_responses_api_shell_tool_streaming_sees_shell_output (e.g. openai/gpt-5.2 for shell support)."""
        return None

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_openai_responses_api(self, sync_mode):
        litellm._turn_on_debug()
        litellm.set_verbose = True
        base_completion_call_args = self.get_base_completion_call_args()
        try:
            if sync_mode:
                response = litellm.responses(
                    input="Basic ping",
                    max_output_tokens=20,
                    **base_completion_call_args,
                )
            else:
                response = await litellm.aresponses(
                    input="Basic ping",
                    max_output_tokens=20,
                    **base_completion_call_args,
                )
        except litellm.InternalServerError:
            pytest.skip("Skipping test due to litellm.InternalServerError")
        print("litellm response=", json.dumps(response, indent=4, default=str))

        # Use the helper function to validate the response
        validate_responses_api_response(response, final_chunk=True)

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=2)
    async def test_basic_openai_responses_api_streaming(self, sync_mode):
        litellm._turn_on_debug()
        # Enable cost calculation for streaming usage
        litellm.include_cost_in_streaming_usage = True
        base_completion_call_args = self.get_base_completion_call_args()
        collected_content_string = ""
        response_completed_event = None
        if sync_mode:
            response = litellm.responses(
                input="Basic ping", stream=True, **base_completion_call_args
            )
            for event in response:
                print("litellm response=", json.dumps(event, indent=4, default=str))
                if event.type == "response.output_text.delta":
                    collected_content_string += event.delta
                elif event.type == "response.completed":
                    response_completed_event = event
        else:
            response = await litellm.aresponses(
                input="Basic ping", stream=True, **base_completion_call_args
            )
            async for event in response:
                print("litellm response=", json.dumps(event, indent=4, default=str))
                if event.type == "response.output_text.delta":
                    collected_content_string += event.delta
                elif event.type == "response.completed":
                    response_completed_event = event

        # assert the response completed event is not None
        assert response_completed_event is not None

        # assert the response completed event has a response
        assert response_completed_event.response is not None

        # For async agent APIs (like Manus), the response may be in 'running' state
        # without content yet - this is valid behavior
        response_status = response_completed_event.response.status
        if response_status in ["running", "pending"]:
            # Running/pending state is acceptable - task started successfully
            print(f"Response is in '{response_status}' state - async agent API behavior")
            assert response_completed_event.response.id is not None
        else:
            # For completed responses, validate content and usage
            # assert the delta chunks content had len(collected_content_string) > 0
            # this content is typically rendered on chat ui's
            assert len(collected_content_string) > 0

            # assert the response completed event includes the usage
            assert response_completed_event.response.usage is not None

            # basic test assert the usage seems reasonable
            print(
                "response_completed_event.response.usage=",
                response_completed_event.response.usage,
            )
            assert (
                response_completed_event.response.usage.input_tokens > 0
                and response_completed_event.response.usage.input_tokens < 100
            )
            assert (
                response_completed_event.response.usage.output_tokens > 0
                and response_completed_event.response.usage.output_tokens < 2000
            )
            assert (
                response_completed_event.response.usage.total_tokens > 0
                and response_completed_event.response.usage.total_tokens < 2000
            )

            # total tokens should be the sum of input and output tokens
            assert (
                response_completed_event.response.usage.total_tokens
                == response_completed_event.response.usage.input_tokens
                + response_completed_event.response.usage.output_tokens
            )

            # assert the response completed event includes cost when include_cost_in_streaming_usage is True
            assert hasattr(response_completed_event.response.usage, "cost"), "Cost should be included in streaming responses API usage object"
            assert response_completed_event.response.usage.cost > 0, "Cost should be greater than 0"
            print(f"Cost found in streaming response: {response_completed_event.response.usage.cost}")
        
        # Reset the setting
        litellm.include_cost_in_streaming_usage = False

    @pytest.mark.parametrize("sync_mode", [False, True])
    @pytest.mark.asyncio
    async def test_basic_openai_responses_delete_endpoint(self, sync_mode):
        litellm._turn_on_debug()
        litellm.set_verbose = True
        base_completion_call_args = self.get_base_completion_call_args()
        if sync_mode:
            response = litellm.responses(
                input="Basic ping", max_output_tokens=20, **base_completion_call_args
            )

            # delete the response
            if isinstance(response, ResponsesAPIResponse):
                litellm.delete_responses(
                    response_id=response.id, **base_completion_call_args
                )
            else:
                raise ValueError("response is not a ResponsesAPIResponse")
        else:
            response = await litellm.aresponses(
                input="Basic ping", max_output_tokens=20, **base_completion_call_args
            )

            # async delete the response
            if isinstance(response, ResponsesAPIResponse):
                await litellm.adelete_responses(
                    response_id=response.id, **base_completion_call_args
                )
            else:
                raise ValueError("response is not a ResponsesAPIResponse")

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.flaky(retries=3, delay=2)
    @pytest.mark.asyncio
    async def test_basic_openai_responses_streaming_delete_endpoint(self, sync_mode):
        # litellm._turn_on_debug()
        # litellm.set_verbose = True
        base_completion_call_args = self.get_base_completion_call_args()
        response_id = None
        if sync_mode:
            response_id = None
            response = litellm.responses(
                input="Basic ping",
                max_output_tokens=20,
                stream=True,
                **base_completion_call_args,
            )
            for event in response:
                print("litellm response=", json.dumps(event, indent=4, default=str))
                if "response" in event:
                    response_obj = event.get("response")
                    if response_obj is not None:
                        response_id = response_obj.get("id")
            print("got response_id=", response_id)

            # delete the response
            assert response_id is not None
            litellm.delete_responses(
                response_id=response_id, **base_completion_call_args
            )
        else:
            response = await litellm.aresponses(
                input="Basic ping",
                max_output_tokens=20,
                stream=True,
                **base_completion_call_args,
            )
            async for event in response:
                print("litellm response=", json.dumps(event, indent=4, default=str))
                if "response" in event:
                    response_obj = event.get("response")
                    if response_obj is not None:
                        response_id = response_obj.get("id")
            print("got response_id=", response_id)

            # delete the response
            assert response_id is not None
            await litellm.adelete_responses(
                response_id=response_id, **base_completion_call_args
            )

    @pytest.mark.parametrize("sync_mode", [False, True])
    @pytest.mark.flaky(retries=3, delay=2)
    @pytest.mark.asyncio
    async def test_basic_openai_responses_get_endpoint(self, sync_mode):
        litellm._turn_on_debug()
        litellm.set_verbose = True
        base_completion_call_args = self.get_base_completion_call_args()
        if sync_mode:
            response = litellm.responses(
                input="Basic ping", max_output_tokens=20, **base_completion_call_args
            )

            # get the response
            if isinstance(response, ResponsesAPIResponse):
                result = litellm.get_responses(
                    response_id=response.id, **base_completion_call_args
                )
                assert result is not None
                assert result.id == response.id
                assert result.output == response.output
            else:
                raise ValueError("response is not a ResponsesAPIResponse")
        else:
            response = await litellm.aresponses(
                input="Basic ping", max_output_tokens=20, **base_completion_call_args
            )
            # async get the response
            if isinstance(response, ResponsesAPIResponse):
                result = await litellm.aget_responses(
                    response_id=response.id, **base_completion_call_args
                )
                assert result is not None
                assert result.id == response.id
                assert result.output == response.output
            else:
                raise ValueError("response is not a ResponsesAPIResponse")

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=2)
    async def test_basic_openai_list_input_items_endpoint(self):
        """Test that calls the OpenAI List Input Items endpoint"""
        litellm._turn_on_debug()

        response = await litellm.aresponses(
            model="gpt-4o",
            input="Tell me a three sentence bedtime story about a unicorn.",
        )
        print("Initial response=", json.dumps(response, indent=4, default=str))

        response_id = response.get("id")
        assert response_id is not None, "Response should have an ID"
        print(f"Got response_id: {response_id}")

        list_items_response = await litellm.alist_input_items(
            response_id=response_id,
            limit=20,
            order="desc",
        )
        print(
            "List items response=",
            json.dumps(list_items_response, indent=4, default=str),
        )

    @pytest.mark.asyncio
    async def test_multiturn_responses_api(self):
        litellm._turn_on_debug()
        litellm.set_verbose = True
        try:
            base_completion_call_args = self.get_base_completion_call_args()
            response_1 = await litellm.aresponses(
                input="Basic ping", max_output_tokens=20, **base_completion_call_args
            )

            # follow up with a second request
            response_1_id = response_1.id
            response_2 = await litellm.aresponses(
                input="Basic ping",
                max_output_tokens=20,
                previous_response_id=response_1_id,
                **base_completion_call_args,
            )

            # assert the response is not None
            assert response_1 is not None
            assert response_2 is not None
        except litellm.InternalServerError:
            pytest.skip("Skipping test due to litellm.InternalServerError")

    @pytest.mark.asyncio
    async def test_responses_api_with_tool_calls(self):
        """Test that calls the Responses API with tool calls including function call and output"""
        litellm._turn_on_debug()
        litellm.set_verbose = True
        base_completion_call_args = self.get_base_completion_call_args()

        # Define the input with message, function call, and function call output
        input_data: ResponseInputParam = [
            {
                "type": "message",
                "role": "user",
                "content": "How is the weather in São Paulo today ?",
            },
            {
                "type": "function_call",
                "arguments": '{"location": "São Paulo, Brazil"}',
                "call_id": "fc_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "name": "get_weather",
                "id": "fc_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "fc_1fe70e2a-a596-45ef-b72c-9b8567c460e5",
                "output": "Rainy",
            },
        ]

        # Define the tools
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia",
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
            }
        ]

        try:
            # Make the responses API call
            response = await litellm.aresponses(
                input=input_data, store=False, tools=tools, **base_completion_call_args
            )
        except litellm.InternalServerError:
            pytest.skip("Skipping test due to litellm.InternalServerError")

        print("litellm response=", json.dumps(response, indent=4, default=str))

        # Validate the response structure
        validate_responses_api_response(response, final_chunk=True)

        # Additional assertions specific to tool calls
        assert response is not None
        assert "output" in response
        # For async agent APIs (like Manus), the response may be in 'running' state
        # without output yet - this is valid behavior
        if response.get("status") in ["running", "pending"]:
            print(f"Response is in '{response.get('status')}' state - async agent API behavior")
            assert response.get("id") is not None
        else:
            assert len(response["output"]) > 0

    @pytest.mark.asyncio
    async def test_responses_api_multi_turn_with_reasoning_and_structured_output(self):
        """
        Test multi-turn conversation with reasoning, structured output, and tool calls.

        This test validates:
        - First call: Model uses reasoning to process a question and makes a tool call
        - Tool call handling: Function call output is properly processed
        - Second call: Model produces structured output incorporating tool results
        - Structured output: Response conforms to defined Pydantic model schema
        """
        from pydantic import BaseModel

        litellm._turn_on_debug()
        litellm.set_verbose = True
        base_completion_call_args = self.get_base_completion_reasoning_call_args()
        if base_completion_call_args is None:
            pytest.skip("Skipping test due to no base completion reasoning call args")

        # Define tools for the conversation
        tools = [{"type": "function", "name": "get_today"}]

        # Define structured output schema
        class Output(BaseModel):
            today: str
            number_of_r: str

        # Initial conversation input
        input_messages = [
            {
                "role": "user",
                "content": "How many r in strrawberrry? While you're thinking, you should call tool get_today. Then you output the today and number of r",
            }
        ]

        # First call - should trigger reasoning and tool call
        response = await litellm.aresponses(
            input=input_messages,
            tools=tools,
            reasoning={"effort": "low", "summary": "detailed"},
            text_format=Output,
            **base_completion_call_args,
        )

        print("First call output:")
        print(json.dumps(response.output, indent=4, default=str))

        # Validate first response structure
        validate_responses_api_response(response, final_chunk=True)
        assert response.output is not None
        assert len(response.output) > 0

        # Extend input with first response output
        input_messages.extend(response.output)

        # Process any tool calls and add function outputs
        function_outputs = []
        for item in response.output:
            if hasattr(item, "type") and item.type in [
                "function_call",
                "custom_tool_call",
            ]:
                if hasattr(item, "name") and item.name == "get_today":
                    function_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": "2025-01-15",
                        }
                    )

        # Add function outputs to conversation
        input_messages.extend(function_outputs)

        print("Second call input:")
        print(json.dumps(input_messages, indent=4, default=str))

        # Second call - should produce structured output
        final_response = await litellm.aresponses(
            input=input_messages,
            tools=tools,
            reasoning={"effort": "low", "summary": "detailed"},
            text_format=Output,
            **base_completion_call_args,
        )

        print("Second call output:")
        print(json.dumps(final_response.output, indent=4, default=str))

        # Validate final response structure
        validate_responses_api_response(final_response, final_chunk=True)
        assert final_response.output is not None

    def test_openai_responses_api_dict_input_filtering(self):
        """
        Test that regular dict inputs with status fields are properly filtered
        to replicate exclude_unset=True behavior for non-Pydantic objects.
        """
        from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

        # Test input with regular dict objects (like from JSON)
        test_input = [
            {
                "role": "user",
                "content": "test"
            },
            {
                "id": "rs_123",
                "summary": [{"text": "test", "type": "summary_text"}],
                "type": "reasoning",
                "content": None,  # Should be filtered out
                "encrypted_content": None,  # Should be filtered out
                "status": None  # Should be filtered out
            },
            {
                "arguments": "{}",
                "call_id": "call_123",
                "name": "get_today",
                "type": "function_call",
                "id": "fc_123",
                "status": "completed"  # Should be preserved (not a default field)
            }
        ]

        config = OpenAIResponsesAPIConfig()
        validated_input = config._validate_input_param(test_input)

        # Verify the results
        assert len(validated_input) == 3

        # Check reasoning item (index 1)
        reasoning_item = validated_input[1]
        assert reasoning_item["type"] == "reasoning"
        assert "status" not in reasoning_item, "status field should be filtered out from reasoning item"
        assert "content" not in reasoning_item, "content field should be filtered out from reasoning item"
        assert "encrypted_content" not in reasoning_item, "encrypted_content field should be filtered out from reasoning item"
        # Note: ID auto-generation was disabled, so reasoning items may not have IDs
        # Only check for ID if it was present in the original input
        if "id" in reasoning_item:
            assert reasoning_item["id"] == "rs_123", "ID should be preserved if present"
        assert "summary" in reasoning_item, "summary field should be preserved"

        # Check function call item (index 2)
        function_call_item = validated_input[2]
        assert function_call_item["type"] == "function_call"
        assert "status" in function_call_item, "status field should be preserved in function call item"
        assert function_call_item["status"] == "completed", "status value should be preserved"

        print("✅ OpenAI Responses API dict input filtering test passed")

    @pytest.mark.parametrize("sync_mode", [False, True])
    @pytest.mark.flaky(retries=3, delay=2)
    @pytest.mark.asyncio
    async def test_basic_openai_responses_cancel_endpoint(self, sync_mode):
        try:
            litellm._turn_on_debug()
            litellm.set_verbose = True
            base_completion_call_args = self.get_base_completion_call_args()
            if sync_mode:
                response = litellm.responses(
                    input="Basic ping", max_output_tokens=20, background=True, **base_completion_call_args
                )

                # cancel the response
                if isinstance(response, ResponsesAPIResponse):
                    cancel_result = litellm.cancel_responses(
                        response_id=response.id, **base_completion_call_args
                    )
                    assert cancel_result is not None
                    assert hasattr(cancel_result, "id")
                    # The actual response structure depends on the provider implementation
                    assert isinstance(cancel_result, ResponsesAPIResponse)
                else:
                    raise ValueError("response is not a ResponsesAPIResponse")
            else:
                response = await litellm.aresponses(
                    input="Basic ping", max_output_tokens=20, background=True, **base_completion_call_args
                )

                # async cancel the response
                if isinstance(response, ResponsesAPIResponse):
                    cancel_result = await litellm.acancel_responses(
                        response_id=response.id, **base_completion_call_args
                    )
                    assert cancel_result is not None
                    assert hasattr(cancel_result, "id")
                    # The actual response structure depends on the provider implementation
                    assert isinstance(cancel_result, ResponsesAPIResponse)
                else:
                    raise ValueError("response is not a ResponsesAPIResponse")
        except Exception as e:
            if "Cannot cancel a completed response" in str(e):
                pass
            else:
                raise e

    @pytest.mark.parametrize("sync_mode", [False, True])
    @pytest.mark.asyncio
    async def test_cancel_responses_invalid_response_id(self, sync_mode):
        """Test cancel_responses with invalid response ID should raise appropriate error"""
        base_completion_call_args = self.get_base_completion_call_args()

        if sync_mode:
            with pytest.raises(Exception):
                litellm.cancel_responses(
                    response_id="invalid_response_id_12345", **base_completion_call_args
                )
        else:
            with pytest.raises(Exception):
                await litellm.acancel_responses(
                    response_id="invalid_response_id_12345", **base_completion_call_args
                )

    @pytest.mark.asyncio
    async def test_responses_api_context_management_server_side_compaction(self):
        """
        E2E test for server-side compaction (context_management) on OpenAI Responses API.
        Passes context_management with compact_threshold; validates that the request is
        accepted and returns a valid response. Compaction may not run for short inputs.
        """
        base_completion_call_args = self.get_base_completion_call_args()
        model = base_completion_call_args.get("model") or ""
        # Azure does not support compaction context_management (only clear_tool_results)
        if "azure/" in str(model):
            pytest.skip(
                "context_management compaction is not supported on Azure"
            )
        if "openai/" not in str(model):
            pytest.skip(
                "context_management server-side compaction e2e is only run for OpenAI"
            )
        context_management = [{"type": "compaction", "compact_threshold": 200000}]
        try:
            response = await litellm.aresponses(
                input="Short ping to verify context_management is accepted.",
                max_output_tokens=20,
                context_management=context_management,
                **base_completion_call_args,
            )
        except litellm.InternalServerError:
            pytest.skip("Skipping test due to litellm.InternalServerError")
        validate_responses_api_response(response, final_chunk=True)
        assert response.get("id") is not None
        assert response.get("status") is not None

    @pytest.mark.asyncio
    async def test_responses_api_shell_tool(self):
        """
        E2E test for Shell tool on OpenAI Responses API.
        Passes tools=[{"type": "shell", "environment": {"type": "container_auto"}}];
        validates that the request is accepted and returns a valid response.
        Only runs for OpenAI/Azure (Responses API with shell support).
        """
        base_completion_call_args = self.get_base_completion_call_args()
        model = self.get_advanced_model_for_shell_tool() or base_completion_call_args.get(
            "model"
        ) or ""
        if "openai/" not in str(model) and "azure/" not in str(model):
            pytest.skip(
                "Shell tool e2e is only run for OpenAI/Azure Responses API"
            )
        tools = [{"type": "shell", "environment": {"type": "container_auto"}}]
        input_msg = "List files in /mnt/data and show python --version."
        try:
            response = await litellm.aresponses(
                **{**base_completion_call_args, "model": model},
                input=input_msg,
                max_output_tokens=256,
                tools=tools,
                tool_choice="auto",
            )
        except litellm.InternalServerError:
            pytest.skip("Skipping test due to litellm.InternalServerError")
        except litellm.BadRequestError as e:
            if "shell" in str(e).lower() and "not supported" in str(e).lower():
                pytest.skip(
                    "Shell tool is not supported for this model (e.g. gpt-4o); use a model that supports shell"
                )
            raise
        validate_responses_api_response(response, final_chunk=True)
        assert response.get("id") is not None
        assert response.get("status") is not None

    @pytest.mark.asyncio
    async def test_responses_api_shell_tool_streaming_sees_shell_output(self):
        """
        E2E streaming call with Shell tool; validate we can see shell output in the stream.

        Calls aresponses(..., tools=[shell], stream=True), then iterates the stream and
        asserts at least one event is shell-related or response output contains shell_call.
        Skips when model does not support shell (e.g. gpt-4o).
        """
        base_completion_call_args = self.get_base_completion_call_args()
        model = self.get_advanced_model_for_shell_tool() or base_completion_call_args.get(
            "model"
        ) or "openai/gpt-5.2"
        if "openai/" not in str(model):
            pytest.skip(
                "Shell tool streaming e2e is only run for OpenAI/Azure Responses API"
            )
        tools = [{"type": "shell", "environment": {"type": "container_auto"}}]
        input_msg = "List files in /mnt/data and run python --version."

        stream = await litellm.aresponses(
            **{**base_completion_call_args, "model": model},
            input=input_msg,
            max_output_tokens=512,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )


        event_types_seen = []
        output_items_with_shell = []

        async for event in stream:
            print("event=", json.dumps(event, indent=4, default=str))
            event_type = getattr(event, "type", None) or (
                event.get("type") if isinstance(event, dict) else None
            )
            if event_type is not None:
                event_types_seen.append(str(event_type))
            if "shell" in str(event_type or "").lower():
                output_items_with_shell.append(event_type)
            response_obj = getattr(event, "response", None) or (
                event.get("response") if isinstance(event, dict) else None
            )
            if response_obj is not None:
                output = getattr(response_obj, "output", None) or (
                    response_obj.get("output") if isinstance(response_obj, dict) else None
                )
                if isinstance(output, list):
                    for item in output:
                        item_type = getattr(item, "type", None) or (
                            item.get("type") if isinstance(item, dict) else None
                        )
                        if item_type and "shell" in str(item_type).lower():
                            output_items_with_shell.append(item_type)

        assert len(event_types_seen) > 0, "Expected at least one stream event"
        assert len(output_items_with_shell) > 0, (
            f"Expected to see shell output in stream; event types seen: {event_types_seen!r}"
        )
