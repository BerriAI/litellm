

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


class BaseAnthropicMessagesTest:
    """Base class for anthropic messages tests to reduce code duplication"""
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Override in subclasses to provide model-specific configuration"""
        raise NotImplementedError("Subclasses must implement model_config")
    
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
    
    async def _test_non_streaming_base(self):
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
    
    async def _test_streaming_base(self):
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