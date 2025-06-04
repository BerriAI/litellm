
import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os
import uuid
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
    ResponseTextConfig,
    ResponseAPIUsage,
    IncompleteDetails,
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
        response["created_at"], (int, float)
    ), "Response should have a numeric 'created_at' field"
    assert "output" in response and isinstance(
        response["output"], list
    ), "Response should have a list 'output' field"
    assert "parallel_tool_calls" in response and isinstance(
        response["parallel_tool_calls"], bool
    ), "Response should have a boolean 'parallel_tool_calls' field"

    # Optional fields with their expected types
    optional_fields = {
        "error": (dict, type(None)),  # error can be dict or None
        "incomplete_details": (IncompleteDetails, type(None)),
        "instructions": (str, type(None)),
        "metadata": dict,
        "model": str,
        "object": str,
        "temperature": (int, float, type(None)),
        "tool_choice": (dict, str),
        "tools": list,
        "top_p": (int, float, type(None)),
        "max_output_tokens": (int, type(None)),
        "previous_response_id": (str, type(None)),
        "reasoning": dict,
        "status": str,
        "text": ResponseTextConfig,
        "truncation": (str, type(None)),
        "usage": ResponseAPIUsage,
        "user": (str, type(None)),
    }
    if final_chunk is False:
        optional_fields["usage"] = type(None)

    for field, expected_type in optional_fields.items():
        if field in response:
            assert isinstance(
                response[field], expected_type
            ), f"Field '{field}' should be of type {expected_type}, but got {type(response[field])}"

    # Check if output has at least one item
    if final_chunk is True:
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


    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_openai_responses_api(self, sync_mode):
        litellm._turn_on_debug()
        litellm.set_verbose = True
        base_completion_call_args = self.get_base_completion_call_args()
        try: 
            if sync_mode:
                response = litellm.responses(
                    input="Basic ping", max_output_tokens=20,
                    **base_completion_call_args
                )
            else:
                response = await litellm.aresponses(
                    input="Basic ping", max_output_tokens=20,
                    **base_completion_call_args
                )
        except litellm.InternalServerError: 
            pytest.skip("Skipping test due to litellm.InternalServerError")
        print("litellm response=", json.dumps(response, indent=4, default=str))

        # Use the helper function to validate the response
        validate_responses_api_response(response, final_chunk=True)


    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_openai_responses_api_streaming(self, sync_mode):
        litellm._turn_on_debug()
        base_completion_call_args = self.get_base_completion_call_args()
        collected_content_string = ""
        response_completed_event = None
        if sync_mode:
            response = litellm.responses(
                input="Basic ping",
                stream=True,
                **base_completion_call_args
            )
            for event in response:
                print("litellm response=", json.dumps(event, indent=4, default=str))
                if event.type == "response.output_text.delta":
                    collected_content_string += event.delta
                elif event.type == "response.completed":
                    response_completed_event = event
        else:
            response = await litellm.aresponses(
                input="Basic ping",
                stream=True,
                **base_completion_call_args
            )
            async for event in response:
                print("litellm response=", json.dumps(event, indent=4, default=str))
                if event.type == "response.output_text.delta":
                    collected_content_string += event.delta
                elif event.type == "response.completed":
                    response_completed_event = event

        # assert the delta chunks content had len(collected_content_string) > 0
        # this content is typically rendered on chat ui's
        assert len(collected_content_string) > 0

        # assert the response completed event is not None
        assert response_completed_event is not None

        # assert the response completed event has a response
        assert response_completed_event.response is not None

        # assert the response completed event includes the usage
        assert response_completed_event.response.usage is not None

        # basic test assert the usage seems reasonable
        print("response_completed_event.response.usage=", response_completed_event.response.usage)
        assert response_completed_event.response.usage.input_tokens > 0 and response_completed_event.response.usage.input_tokens < 100
        assert response_completed_event.response.usage.output_tokens > 0 and response_completed_event.response.usage.output_tokens < 1000
        assert response_completed_event.response.usage.total_tokens > 0 and response_completed_event.response.usage.total_tokens < 1000

        # total tokens should be the sum of input and output tokens
        assert response_completed_event.response.usage.total_tokens == response_completed_event.response.usage.input_tokens + response_completed_event.response.usage.output_tokens



    @pytest.mark.parametrize("sync_mode", [False, True])
    @pytest.mark.asyncio
    async def test_basic_openai_responses_delete_endpoint(self, sync_mode):
        litellm._turn_on_debug()
        litellm.set_verbose = True
        base_completion_call_args = self.get_base_completion_call_args()
        if sync_mode:
            response = litellm.responses(
                input="Basic ping", max_output_tokens=20,
                **base_completion_call_args
            )

            # delete the response
            if isinstance(response, ResponsesAPIResponse):
                litellm.delete_responses(
                    response_id=response.id,
                    **base_completion_call_args
                )
            else:
                raise ValueError("response is not a ResponsesAPIResponse")
        else:
            response = await litellm.aresponses(
                input="Basic ping", max_output_tokens=20,
                **base_completion_call_args
            )

            # async delete the response
            if isinstance(response, ResponsesAPIResponse):
                await litellm.adelete_responses(
                    response_id=response.id,
                    **base_completion_call_args
                )
            else:
                raise ValueError("response is not a ResponsesAPIResponse")
    

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_openai_responses_streaming_delete_endpoint(self, sync_mode):
        #litellm._turn_on_debug()
        #litellm.set_verbose = True
        base_completion_call_args = self.get_base_completion_call_args()
        response_id = None
        if sync_mode:
            response_id = None
            response = litellm.responses(
                input="Basic ping", max_output_tokens=20,
                stream=True,
                **base_completion_call_args
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
                response_id=response_id,
                **base_completion_call_args
            )
        else:
            response = await litellm.aresponses(
                input="Basic ping", max_output_tokens=20,
                stream=True,
                **base_completion_call_args
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
                response_id=response_id,
                **base_completion_call_args
            )

    @pytest.mark.parametrize("sync_mode", [False, True])
    @pytest.mark.asyncio
    async def test_basic_openai_responses_get_endpoint(self, sync_mode):
        litellm._turn_on_debug()
        litellm.set_verbose = True
        base_completion_call_args = self.get_base_completion_call_args()
        if sync_mode:
            response = litellm.responses(
                input="Basic ping", max_output_tokens=20,
                **base_completion_call_args
            )

            # get the response
            if isinstance(response, ResponsesAPIResponse):
                result = litellm.get_responses(
                    response_id=response.id,
                    **base_completion_call_args
                )
                assert result is not None
                assert result.id == response.id
                assert result.output == response.output
            else:
                raise ValueError("response is not a ResponsesAPIResponse")
        else:
            response = await litellm.aresponses(
                input="Basic ping", max_output_tokens=20,
                **base_completion_call_args
            )
            # async get the response
            if isinstance(response, ResponsesAPIResponse):
                result = await litellm.aget_responses(
                    response_id=response.id,
                    **base_completion_call_args
                )
                assert result is not None
                assert result.id == response.id
                assert result.output == response.output
            else:
                raise ValueError("response is not a ResponsesAPIResponse")
    
    @pytest.mark.asyncio
    async def test_multiturn_responses_api(self):
        litellm._turn_on_debug()
        litellm.set_verbose = True
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
            **base_completion_call_args
        )

        # assert the response is not None
        assert response_1 is not None
        assert response_2 is not None
