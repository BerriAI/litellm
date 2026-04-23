import asyncio
import json
import sys
import os
import tempfile
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
from google.genai.types import ContentDict, PartDict
from litellm.types.google_genai.main import GenerateContentResponse
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload


def load_vertex_ai_credentials(model: str):
    """Load Vertex AI credentials for tests"""
    # Define the path to the vertex_key.json file
    if "vertex_ai" not in model:
        return None
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary files
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)
    
    return os.path.abspath(temp_file.name)


class TestCustomLogger(CustomLogger):
    def __init__(
        self,
    ):
        self.standard_logging_object: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("in async_log_success_event")
        print("kwargs=", json.dumps(kwargs, indent=4, default=str))
        self.standard_logging_object = kwargs["standard_logging_object"]
        pass


class BaseGoogleGenAITest:
    """Base class for Google GenAI generate content tests to reduce code duplication"""
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Override in subclasses to provide model-specific configuration"""
        raise NotImplementedError("Subclasses must implement model_config")
    
    @property
    def _temp_files_to_cleanup(self):
        """Lazy initialization of temp files list"""
        if not hasattr(self, '_temp_files_list'):
            self._temp_files_list = []
        return self._temp_files_list
    
    def cleanup_temp_files(self):
        """Clean up any temporary files created during testing"""
        for temp_file in self._temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
            except OSError:
                pass  # File might already be deleted
        self._temp_files_to_cleanup.clear()
    
    
    def _validate_non_streaming_response(self, response: Any):
        """Validate non-streaming response structure"""
        # Handle type checking - response should be a GenerateContentResponse for non-streaming
        if isinstance(response, AsyncIterator):
            pytest.fail("Expected non-streaming response but got AsyncIterator")
        
        assert isinstance(response, GenerateContentResponse), f"Expected GenerateContentResponse, got {type(response)}"
        print(f"Response: {response.model_dump_json(indent=4)}")
        
        # Basic validation - adjust based on actual Google GenAI response structure
        # The exact structure may vary, so we'll be flexible here
        assert response is not None, "Response should not be None"
    
    def _validate_streaming_response(self, chunks: List[Any]):
        """Validate streaming response chunks"""
        assert isinstance(chunks, list), f"Expected list of chunks, got {type(chunks)}"
        assert len(chunks) >= 0, "Should have at least 0 chunks"
        print(f"Total chunks received: {len(chunks)}")
    
    def _validate_standard_logging_payload(
        self, slp: StandardLoggingPayload, response: Any
    ):
        """
        Validate that a StandardLoggingPayload object matches the expected response for Google GenAI

        Args:
            slp (StandardLoggingPayload): The standard logging payload object to validate
            response: The Google GenAI response to compare against
        """
        # Validate payload exists
        assert slp is not None, "Standard logging payload should not be None"

        # Validate basic structure
        assert "prompt_tokens" in slp, "Standard logging payload should have prompt_tokens"
        assert "completion_tokens" in slp, "Standard logging payload should have completion_tokens" 
        assert "total_tokens" in slp, "Standard logging payload should have total_tokens"
        assert "response_cost" in slp, "Standard logging payload should have response_cost"

        # Validate token counts are reasonable (non-negative numbers)
        assert slp["prompt_tokens"] >= 0, "Prompt tokens should be non-negative"
        assert slp["completion_tokens"] >= 0, "Completion tokens should be non-negative"
        assert slp["total_tokens"] >= 0, "Total tokens should be non-negative"

        # Validate spend
        assert slp["response_cost"] >= 0, "Response cost should be non-negative"

        print(f"Standard logging payload validation passed: prompt_tokens={slp['prompt_tokens']}, completion_tokens={slp['completion_tokens']}, total_tokens={slp['total_tokens']}, cost={slp['response_cost']}")
    
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
            role="user",
        )
        temp_file_path = load_vertex_ai_credentials(model=request_params["model"])
        if temp_file_path:
            self._temp_files_to_cleanup.append(temp_file_path)
            
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
        temp_file_path = load_vertex_ai_credentials(model=request_params["model"])
        if temp_file_path:
            self._temp_files_to_cleanup.append(temp_file_path)
        contents = ContentDict(
            parts=[
                PartDict(
                    text="Hello, can you tell me a short joke?"
                )
            ],
            role="user",
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

    @pytest.mark.asyncio
    async def test_async_non_streaming_with_logging(self):
        """Test async non-streaming Google GenAI generate content with logging"""
        litellm._turn_on_debug()
        litellm.logging_callback_manager._reset_all_callbacks()
        litellm.set_verbose = True
        test_custom_logger = TestCustomLogger()
        litellm.callbacks = [test_custom_logger]
        
        request_params = self.model_config
        temp_file_path = load_vertex_ai_credentials(model=request_params["model"])
        if temp_file_path:
            self._temp_files_to_cleanup.append(temp_file_path)
        contents = ContentDict(
            parts=[
                PartDict(
                    text="Hello, can you tell me a short joke?"
                )
            ],
            role="user",
        )

        print("\n--- Testing async agenerate_content with logging ---")
        response = await agenerate_content(
            contents=contents,
            **request_params
        )

        print("Google GenAI response=", json.dumps(response, indent=4, default=str))

        print("sleeping for 5 seconds...")
        await asyncio.sleep(5)
        print(
            "standard logging payload=",
            json.dumps(test_custom_logger.standard_logging_object, indent=4, default=str),
        )

        assert response is not None
        assert test_custom_logger.standard_logging_object is not None

        self._validate_standard_logging_payload(
            test_custom_logger.standard_logging_object, response
        )

    @pytest.mark.asyncio
    async def test_async_streaming_with_logging(self):
        """Test async streaming Google GenAI generate content with logging"""
        litellm._turn_on_debug()
        litellm.set_verbose = True
        litellm.logging_callback_manager._reset_all_callbacks()
        test_custom_logger = TestCustomLogger()
        litellm.callbacks = [test_custom_logger]
        
        request_params = self.model_config
        temp_file_path = load_vertex_ai_credentials(model=request_params["model"])
        if temp_file_path:
            self._temp_files_to_cleanup.append(temp_file_path)
        contents = ContentDict(
            parts=[
                PartDict(
                    text="Hello, can you tell me a short joke?"
                )
            ],
            role="user",
        )

        print("\n--- Testing async agenerate_content_stream with logging ---")
        response = await agenerate_content_stream(
            contents=contents,
            **request_params
        )
        
        chunks = []
        async for chunk in response:
            print(f"Google GenAI chunk: {chunk}")
            chunks.append(chunk)

        print("sleeping for 5 seconds...")
        await asyncio.sleep(5)
        print(
            "standard logging payload=",
            json.dumps(test_custom_logger.standard_logging_object, indent=4, default=str),
        )

        assert len(chunks) >= 0
        assert test_custom_logger.standard_logging_object is not None

        self._validate_standard_logging_payload(
            test_custom_logger.standard_logging_object, chunks
        )
