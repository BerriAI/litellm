

import json
import os
import sys
from datetime import datetime
from typing import AsyncIterator, Dict, Any
import asyncio
import unittest.mock
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
import litellm
import pytest
from dotenv import load_dotenv
from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
    anthropic_messages,
)

from typing import Optional
from litellm.types.utils import StandardLoggingPayload
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.router import Router
import importlib
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
class TestCustomLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.logged_standard_logging_payload: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("inside async_log_success_event")
        self.logged_standard_logging_payload = kwargs.get("standard_logging_object")

        pass


class BaseAnthropicMessagesTest:
    """Base class for anthropic messages tests to reduce code duplication"""
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Override in subclasses to provide model-specific configuration"""
        raise NotImplementedError("Subclasses must implement model_config")
    
    @property
    def expected_model_name_in_logging(self) -> str:
        """
        This is the model name that is expected to be in the logging payload
        """
        raise NotImplementedError("Subclasses must implement expected_model_name_in_logging")
    
    def _validate_response(self, response: Any):
        """Validate non-streaming response structure"""
        # Handle type checking - response should be a dict for non-streaming
        if isinstance(response, AsyncIterator):
            pytest.fail("Expected non-streaming response but got AsyncIterator")
        
        assert isinstance(response, dict), f"Expected dict response, got {type(response)}"
        assert "id" in response
        assert "content" in response
        assert "model" in response
        assert response.get("role") == "assistant"
    
    @pytest.mark.asyncio
    async def test_non_streaming_base(self):
        """Base test for non-streaming requests"""
        litellm._turn_on_debug()
        
        request_params = self.model_config

        # Set up test parameters
        messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]
        
        # Prepare call arguments
        call_args = {
            "messages": messages,
            "max_tokens": 100,
        }


        

        # Add any additional config from subclass
        call_args.update(request_params)
        
        # Call the handler
        response = await litellm.anthropic.messages.acreate(**call_args)
        
        print(f"Non-streaming {request_params['model']} response: ", response)
        
        # Verify response
        self._validate_response(response)
        
        print(f"Non-streaming response: {json.dumps(response, indent=2, default=str)}")
        return response
    
    @pytest.mark.asyncio
    async def test_streaming_base(self):
        """Base test for streaming requests"""
        request_params = self.model_config
        # Set up test parameters
        messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]
        
        # Prepare call arguments
        call_args = {
            "messages": messages,
            "max_tokens": 100,
            "stream": True,
            "client": AsyncHTTPHandler(),
        }

        
        # Add any additional config from subclass
        call_args.update(request_params)
        
        # Call the handler
        response = await litellm.anthropic.messages.acreate(**call_args)
        
        collected_chunks = []
        if isinstance(response, AsyncIterator):
            async for chunk in response:
                print("chunk=", chunk)
                collected_chunks.append(chunk)
        
        print("collected_chunks=", collected_chunks)
        return collected_chunks


    @pytest.mark.asyncio
    async def test_response_format_consistency(self):
        """
        Test that response content blocks are consistently dicts (not Pydantic objects).
        
        This ensures that code like response["content"][0]["type"] works 
        regardless of the target provider.
        
        Issue: https://github.com/BerriAI/litellm/issues/20342
        """
        litellm._turn_on_debug()
        
        request_params = self.model_config
        
        # Set up test parameters
        messages = [{"role": "user", "content": "Say hi"}]
        
        # Prepare call arguments
        call_args = {
            "messages": messages,
            "max_tokens": 100,
        }
        
        # Add any additional config from subclass
        call_args.update(request_params)
        
        # Call the handler
        response = await litellm.anthropic.messages.acreate(**call_args)
        
        print(f"Response for {request_params['model']}: {json.dumps(response, indent=2, default=str)}")
        
        # Verify response structure
        assert "content" in response, "Response should have 'content' field"
        assert len(response["content"]) > 0, "Response content should not be empty"
        
        # Get the first content block
        block = response["content"][0]
        
        # Check that the block is a dict, not a Pydantic object
        assert isinstance(block, dict), (
            f"Content block should be a dict, but got {type(block)}. "
            f"This means response format is inconsistent across providers."
        )
        
        # Verify we can access fields using dict syntax (not object attributes)
        try:
            block_type = block["type"]
            print(f"✓ Successfully accessed block['type']: {block_type}")
        except TypeError as e:
            pytest.fail(
                f"Cannot access content block using dict syntax: {e}. "
                f"Block type: {type(block)}"
            )
        
        # Verify the block has expected structure
        assert "type" in block, "Content block should have 'type' field"
        if block["type"] == "text":
            assert "text" in block, "Text content block should have 'text' field"
        
        print(f"✓ Response format consistency test passed for {request_params['model']}")

    @pytest.mark.asyncio
    async def test_anthropic_messages_litellm_router_streaming_with_logging(self):
        """
        Test that logging and cost tracking works for anthropic_messages with streaming request
        """
        test_custom_logger = TestCustomLogger()
        litellm.callbacks = [test_custom_logger]
        litellm._turn_on_debug()
        router = Router(
            model_list=[
                {
                    "model_name": "claude-special-alias",
                    "litellm_params": {
                        **self.model_config
                    },
                }
            ]
        )

        # Set up test parameters
        messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

        # Call the handler
        response = await router.aanthropic_messages(
            messages=messages,
            model="claude-special-alias",
            max_tokens=100,
            stream=True,
        )

        response_prompt_tokens = 0
        response_completion_tokens = 0
        all_anthropic_usage_chunks = []
        buffer = ""

        async for chunk in response:
            # Decode chunk if it's bytes
            print("chunk=", chunk)

            # Handle SSE format chunks
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8")
                buffer += chunk_str
                # Extract the JSON data part from SSE format
                for line in buffer.split("\n"):
                    if line.startswith("data: "):
                        try:
                            json_data = json.loads(line[6:])  # Skip the 'data: ' prefix
                            print(
                                "\n\nJSON data:",
                                json.dumps(json_data, indent=4, default=str),
                            )

                            # Extract usage information
                            if (
                                json_data.get("type") == "message_start"
                                and "message" in json_data
                            ):
                                if "usage" in json_data["message"]:
                                    usage = json_data["message"]["usage"]
                                    all_anthropic_usage_chunks.append(usage)
                                    print(
                                        "USAGE BLOCK",
                                        json.dumps(usage, indent=4, default=str),
                                    )
                            elif "usage" in json_data:
                                usage = json_data["usage"]
                                all_anthropic_usage_chunks.append(usage)
                                print(
                                    "USAGE BLOCK", json.dumps(usage, indent=4, default=str)
                                )
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON from: {line[6:]}")
            elif hasattr(chunk, "message"):
                if chunk.message.usage:
                    print(
                        "USAGE BLOCK",
                        json.dumps(chunk.message.usage, indent=4, default=str),
                    )
                    all_anthropic_usage_chunks.append(chunk.message.usage)
            elif hasattr(chunk, "usage"):
                print("USAGE BLOCK", json.dumps(chunk.usage, indent=4, default=str))
                all_anthropic_usage_chunks.append(chunk.usage)

        print(
            "all_anthropic_usage_chunks",
            json.dumps(all_anthropic_usage_chunks, indent=4, default=str),
        )

        # Extract token counts from usage data
        if all_anthropic_usage_chunks:
            response_prompt_tokens = max(
                [usage.get("input_tokens", 0) for usage in all_anthropic_usage_chunks]
            )
            response_completion_tokens = max(
                [usage.get("output_tokens", 0) for usage in all_anthropic_usage_chunks]
            )

        print("input_tokens_anthropic_api", response_prompt_tokens)
        print("output_tokens_anthropic_api", response_completion_tokens)

        await asyncio.sleep(4)

        print(
            "logged_standard_logging_payload",
            json.dumps(
                test_custom_logger.logged_standard_logging_payload, indent=4, default=str
            ),
        )

        assert test_custom_logger.logged_standard_logging_payload is not None, "Logging payload should not be None"
        assert test_custom_logger.logged_standard_logging_payload["messages"] == messages
        assert test_custom_logger.logged_standard_logging_payload["response"] is not None
        assert (
            test_custom_logger.logged_standard_logging_payload["model"]
            == self.expected_model_name_in_logging
        )

        # check logged usage + spend
        assert test_custom_logger.logged_standard_logging_payload["response_cost"] > 0
        assert (
            test_custom_logger.logged_standard_logging_payload["prompt_tokens"]
            == response_prompt_tokens
        )
        assert (
            test_custom_logger.logged_standard_logging_payload["completion_tokens"]
            == response_completion_tokens
        )