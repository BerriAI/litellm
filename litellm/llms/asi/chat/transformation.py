"""
Translate from OpenAI's `/v1/chat/completions` to ASI's `/v1/chat/completions`
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast, Iterator, AsyncIterator, TypeVar, Type

import httpx
from pydantic import BaseModel

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.utils import ModelResponse, ModelResponseStream
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.asi.chat.json_extraction import extract_json


class ASIChatCompletionStreamingHandler(BaseModelResponseIterator):
    """ASI-specific streaming handler that handles ASI's streaming response format"""
    
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            # Handle ASI's streaming response format
            # ASI might not include 'created' in streaming chunks
            return ModelResponseStream(
                id=chunk.get("id", ""),  # Use empty string as fallback
                object="chat.completion.chunk",
                created=chunk.get("created", int(time.time())),  # Use current time as fallback
                model=chunk.get("model", ""),  # Use empty string as fallback
                choices=chunk.get("choices", []),
            )
        except Exception as e:
            # Log the error but don't crash
            print(f"Error parsing ASI streaming chunk: {str(e)}")
            # Return a minimal valid response
            return ModelResponseStream(
                id="",
                object="chat.completion.chunk",
                created=int(time.time()),
                model="",
                choices=[],
            )


class ASIChatConfig(OpenAIGPTConfig):
    """ASI Chat API Configuration
    
    This class extends OpenAIGPTConfig to provide ASI-specific functionality,
    particularly for JSON extraction from responses.
    """
    
    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """Get the API key for ASI"""
        return api_key or get_secret_str("ASI_API_KEY")
    
    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        """Get the API base URL for ASI"""
        return api_base or get_secret_str("ASI_API_BASE") or "https://api.asi1.ai/v1"
    
    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """Transform ASI response, handling JSON extraction if requested"""
        # First get the standard OpenAI-compatible response
        response = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        
        # Check if JSON extraction is requested
        json_requested = optional_params.get("json_response_requested", False) or json_mode
        
        if json_requested:
            if hasattr(logging_obj, "verbose") and logging_obj.verbose:
                print("ASI: JSON response format requested, applying extraction")
            
            try:
                # For ModelResponse objects, directly access the choices
                if isinstance(response, ModelResponse) and hasattr(response, "choices"):
                    choices = response.choices
                    
                    # Process each choice
                    for choice in choices:
                        # Use type-safe attribute access with proper guards
                        try:
                            # Try to extract content from message or delta
                            content = None
                            
                            # For non-streaming responses (message)
                            if hasattr(choice, "message"):
                                message = getattr(choice, "message")
                                if hasattr(message, "content"):
                                    content = getattr(message, "content")
                                    
                                    # Apply JSON extraction if content exists
                                    if content and isinstance(content, str):
                                        extracted_json = extract_json(content)
                                        if extracted_json:
                                            # Update content safely
                                            setattr(message, "content", extracted_json)
                                            if hasattr(logging_obj, "verbose") and logging_obj.verbose:
                                                print(f"ASI: Successfully extracted JSON: {extracted_json[:100]}...")
                            
                            # For streaming responses (delta)
                            elif hasattr(choice, "delta"):
                                delta = getattr(choice, "delta")
                                if hasattr(delta, "content"):
                                    content = getattr(delta, "content")
                                    
                                    # Apply JSON extraction if content exists
                                    if content and isinstance(content, str):
                                        extracted_json = extract_json(content)
                                        if extracted_json:
                                            # Update content safely
                                            setattr(delta, "content", extracted_json)
                                            if hasattr(logging_obj, "verbose") and logging_obj.verbose:
                                                print(f"ASI: Successfully extracted JSON from streaming: {extracted_json[:100]}...")
                        except Exception as attr_error:
                            # Log attribute access errors but continue processing
                            if hasattr(logging_obj, "verbose") and logging_obj.verbose:
                                print(f"ASI: Error accessing attributes: {str(attr_error)}")
                
                # For streaming responses, handle delta content
                elif isinstance(response, dict) and "choices" in response:
                    for choice in response["choices"]:
                        if isinstance(choice, dict):
                            # Handle delta for streaming
                            if "delta" in choice and isinstance(choice["delta"], dict) and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if content:
                                    extracted_json = extract_json(content)
                                    if extracted_json:
                                        choice["delta"]["content"] = extracted_json
                            
                            # Handle message for non-streaming
                            elif "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                                content = choice["message"]["content"]
                                if content:
                                    extracted_json = extract_json(content)
                                    if extracted_json:
                                        choice["message"]["content"] = extracted_json
            except Exception as e:
                # Log the error but don't fail the request
                if hasattr(logging_obj, "verbose") and logging_obj.verbose:
                    print(f"Error extracting JSON from ASI response: {str(e)}")
        
        return response
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
        """Map OpenAI parameters to ASI parameters"""
        # Check for JSON response format
        response_format = non_default_params.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            # Flag that we want JSON extraction
            optional_params["json_response_requested"] = True
            optional_params["json_mode"] = True
            
            # ASI doesn't natively support response_format, but we'll keep it for consistency
            # with the OpenAI API and handle it in our transformation layer
            if "response_format" not in optional_params:
                optional_params["response_format"] = response_format
        
        # Let the parent class handle the rest of the parameter mapping
        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
    
    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        """Return a custom streaming handler for ASI that can handle ASI's streaming format"""
        return ASIChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
