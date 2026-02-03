"""
A2A Protocol Transformation for LiteLLM
"""
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union, cast

import httpx
from pydantic import BaseModel

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse

from ..common_utils import (
    A2AError,
    convert_messages_to_prompt,
    extract_text_from_a2a_response,
)
from .streaming_iterator import A2AModelResponseIterator


class A2AConfig(BaseConfig):
    """
    Configuration for A2A (Agent-to-Agent) Protocol.
    
    Handles transformation between OpenAI and A2A JSON-RPC 2.0 formats.
    """
    
    def get_supported_openai_params(self, model: str) -> List[str]:
        """Return list of supported OpenAI parameters"""
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
        ]
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to A2A parameters.
        
        For A2A protocol, we don't need to map most parameters since
        they're handled in the transform_request method.
        """
        return optional_params
    
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate environment and set headers for A2A requests.
        
        Args:
            headers: Request headers dict
            model: Model name
            messages: Messages list
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            api_key: API key (optional for A2A)
            api_base: API base URL
        
        Returns:
            Updated headers dict
        """
        # Ensure Content-Type is set to application/json for JSON-RPC 2.0
        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        
        # Add Authorization header if API key is provided
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        
        return headers
    
    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete A2A agent endpoint URL.
        
        A2A agents use JSON-RPC 2.0 at the base URL, not specific paths.
        The method (message/send or message/stream) is specified in the
        JSON-RPC request body, not in the URL.
        
        Args:
            api_base: Base URL of the A2A agent (e.g., "http://0.0.0.0:9999")
            api_key: API key (not used for URL construction)
            model: Model name (not used for A2A, agent determined by api_base)
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            stream: Whether this is a streaming request (affects JSON-RPC method)
        
        Returns:
            Complete URL for the A2A endpoint (base URL)
        """
        if api_base is None:
            raise ValueError("api_base is required for A2A provider")
        
        # A2A uses JSON-RPC 2.0 at the base URL
        # Remove trailing slash for consistency
        return api_base.rstrip("/")
    
    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI request to A2A JSON-RPC 2.0 format.
        
        Args:
            model: Model name
            messages: List of OpenAI messages
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            headers: Request headers
        
        Returns:
            A2A JSON-RPC 2.0 request dict
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        if not messages:
            raise ValueError("At least one message is required for A2A completion")
        
        # Convert all messages to maintain conversation history
        # Use helper to format conversation with role prefixes
        full_context = convert_messages_to_prompt(messages)
        
        # Create single A2A message with full conversation context
        a2a_message = {
            "role": "user",
            "parts": [{"kind": "text", "text": full_context}],
            "messageId": str(uuid.uuid4()),
        }
        
        # Build JSON-RPC 2.0 request
        # For A2A protocol, the method is "message/send" for non-streaming
        # and "message/stream" for streaming (handled by optional_params["stream"])
        method = "message/stream" if optional_params.get("stream") else "message/send"
        
        request_data = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": {
                "message": a2a_message
            }
        }
        
        return request_data
    
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
        """
        Transform A2A JSON-RPC 2.0 response to OpenAI format.
        
        Args:
            model: Model name
            raw_response: HTTP response from A2A agent
            model_response: Model response object to populate
            logging_obj: Logging object
            request_data: Original request data
            messages: Original messages
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            encoding: Encoding object
            api_key: API key
            json_mode: JSON mode flag
        
        Returns:
            Populated ModelResponse object
        """
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise A2AError(
                status_code=raw_response.status_code,
                message=f"Failed to parse A2A response: {str(e)}",
                headers=dict(raw_response.headers),
            )
        
        # Check for JSON-RPC error
        if "error" in response_json:
            error = response_json["error"]
            raise A2AError(
                status_code=raw_response.status_code,
                message=f"A2A error: {error.get('message', 'Unknown error')}",
                headers=dict(raw_response.headers),
            )
        
        # Extract text from A2A response
        text = extract_text_from_a2a_response(response_json)
        
        # Populate model response
        model_response.choices = [
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content=text,
                    role="assistant",
                ),
            )
        ]
        
        # Set model
        model_response.model = model
        
        # Set ID from response
        model_response.id = response_json.get("id", str(uuid.uuid4()))
        
        return model_response
    
    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator, Any],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> BaseModelResponseIterator:
        """
        Get streaming iterator for A2A responses.
        
        Args:
            streaming_response: Streaming response iterator
            sync_stream: Whether this is a sync stream
            json_mode: JSON mode flag
        
        Returns:
            A2A streaming iterator
        """
        return A2AModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
    
    def _openai_message_to_a2a_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI message to A2A message format.
        
        Args:
            message: OpenAI message dict
        
        Returns:
            A2A message dict
        """
        content = message.get("content", "")
        role = message.get("role", "user")
        
        return {
            "role": role,
            "parts": [{"kind": "text", "text": str(content)}],
            "messageId": str(uuid.uuid4()),
        }
    
    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """Return appropriate error class for A2A errors"""
        # Convert headers to dict if needed
        headers_dict = dict(headers) if isinstance(headers, httpx.Headers) else headers
        return A2AError(
            status_code=status_code,
            message=error_message,
            headers=headers_dict,
        )
