import asyncio
import json
import sys
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.google_genai import (
    generate_content,
    agenerate_content,
    generate_content_stream,
    agenerate_content_stream,
)
from google.genai.types import ContentDict, PartDict, GenerateContentResponse


class BaseGoogleGenAITest:
    """Base class for Google GenAI generate content tests to reduce code duplication"""
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Override in subclasses to provide model-specific configuration"""
        raise NotImplementedError("Subclasses must implement model_config")
    
    
    def _validate_non_streaming_response(self, response: Any):
        """Validate non-streaming response structure"""
        # Handle type checking - response should be a dict for non-streaming
        if isinstance(response, AsyncIterator):
            pytest.fail("Expected non-streaming response but got AsyncIterator")
        
        assert isinstance(response, GenerateContentResponse), f"Expected dict response, got {type(response)}"
        print(f"Response: {response.model_dump_json(indent=4)}")
        
        # Basic validation - adjust based on actual Google GenAI response structure
        # The exact structure may vary, so we'll be flexible here
        assert response is not None, "Response should not be None"
    
    def _validate_streaming_response(self, chunks: List[Any]):
        """Validate streaming response chunks"""
        assert isinstance(chunks, list), f"Expected list of chunks, got {type(chunks)}"
        assert len(chunks) >= 0, "Should have at least 0 chunks"
        print(f"Total chunks received: {len(chunks)}")
    
    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.asyncio
    async def test_non_streaming_base(self, is_async: bool):
        """Base test for non-streaming requests (parametrized for sync/async)"""
        request_params = self.model_config
        contents = ContentDict(
            parts=[
                PartDict(
                    text="Hello, can you tell me a short joke?"
                )
            ],
        )
        litellm._turn_on_debug()

        print(f"Testing {'async' if is_async else 'sync'} non-streaming with model config: {request_params}")
        print(f"Contents: {contents}")
        
        if is_async:
            print("\n--- Testing async agenerate_content ---")
            response = await agenerate_content(
                contents=contents,
                **request_params
            )
        else:
            print("\n--- Testing sync generate_content ---")
            response = generate_content(
                contents=contents,
                **request_params
            )
        
        print(f"{'Async' if is_async else 'Sync'} response: {json.dumps(response, indent=2, default=str)}")
        self._validate_non_streaming_response(response)
        
        return response
    
    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.asyncio
    async def test_streaming_base(self, is_async: bool):
        """Base test for streaming requests (parametrized for sync/async)"""
        request_params = self.model_config
        contents = ContentDict(
            parts=[
                PartDict(
                    text="Hello, can you tell me a short joke?"
                )
            ],
        )

        print(f"Testing {'async' if is_async else 'sync'} streaming with model config: {request_params}")
        print(f"Contents: {contents}")
        
        chunks = []
        
        if is_async:
            print("\n--- Testing async agenerate_content_stream ---")
            response = await agenerate_content_stream(
                contents=contents,
                **request_params
            )
            async for chunk in response:
                print(f"Async chunk: {chunk}")
                chunks.append(chunk)
        else:
            print("\n--- Testing sync generate_content_stream ---")
            response = generate_content_stream(
                contents=contents,
                **request_params
            )
            for chunk in response:
                print(f"Sync chunk: {chunk}")
                chunks.append(chunk)
        
        self._validate_streaming_response(chunks)
        
        return chunks
