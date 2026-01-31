"""
A2A Agent Chat Transformation

Transforms OpenAI chat completion API requests/responses to/from A2A protocol.

A2A Protocol Reference: https://github.com/a2aproject/A2A

OpenAI Message Format:
    {"role": "user", "content": "Hello!"}

A2A Message Format:
    {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello!"}],
        "messageId": "abc123"
    }
"""

import json
import time
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from uuid import uuid4

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    StreamingChoices,
    Usage,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class A2AAgentError(BaseLLMException):
    """Exception for A2A Agent errors."""

    pass


class A2AAgentConfig(BaseConfig):
    """
    Configuration class for A2A Agent provider.
    
    Handles transformation between OpenAI chat completion format and A2A protocol.
    
    Reference: https://github.com/a2aproject/A2A/blob/main/docs/specification.md
    """

    # Default values
    frequency_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    def __init__(
        self,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> str:
        return "a2a_agent"

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Return the list of supported OpenAI parameters.
        
        A2A agents may support a subset of these depending on their implementation.
        """
        return [
            "stream",
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to A2A compatible format.
        
        A2A doesn't have direct parameter equivalents for most LLM params,
        but we preserve them in case the underlying agent uses them.
        """
        supported_params = self.get_supported_openai_params(model)

        for param, value in non_default_params.items():
            if param in supported_params:
                optional_params[param] = value
            elif not drop_params:
                # Pass through unsupported params if not dropping
                optional_params[param] = value

        return optional_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get the API base and key for A2A agent."""
        api_base = api_base or get_secret_str("A2A_AGENT_API_BASE")
        api_key = api_key or get_secret_str("A2A_AGENT_API_KEY")
        return api_base, api_key

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
        Validate and set up the request headers for A2A.
        """
        api_base, api_key = self._get_openai_compatible_provider_info(api_base, api_key)

        if not api_base:
            raise A2AAgentError(
                status_code=400,
                message="api_base is required for A2A agent calls. Set via api_base parameter or A2A_AGENT_API_BASE env var.",
            )

        # Set up headers
        headers = headers or {}
        headers["Content-Type"] = "application/json"
        
        # Add authorization if API key is provided
        if api_key:
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
        Construct the complete URL for the A2A agent endpoint.
        
        A2A uses JSON-RPC at the agent's base URL.
        """
        api_base, _ = self._get_openai_compatible_provider_info(api_base, api_key)

        if not api_base:
            raise A2AAgentError(
                status_code=400,
                message="api_base is required for A2A agent calls.",
            )

        # Strip trailing slash
        api_base = api_base.rstrip("/")
        
        return api_base

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> A2AAgentError:
        """Return the appropriate error class for A2A errors."""
        return A2AAgentError(
            status_code=status_code,
            message=error_message,
            headers=dict(headers) if headers else None,
        )

    # ========================================================================
    # Request Transformation: OpenAI -> A2A
    # ========================================================================

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> Dict[str, Any]:
        """
        Transform OpenAI chat completion request to A2A SendMessageRequest format.
        
        OpenAI format:
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        
        A2A format (JSON-RPC):
            {
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": "request-id",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Hello"}],
                        "messageId": "msg-id"
                    }
                }
            }
        """
        stream = optional_params.get("stream", False)
        
        # Transform messages to A2A format
        a2a_message = self._transform_messages_to_a2a(messages)
        
        # Build A2A JSON-RPC request
        request_id = str(uuid4())
        method = "message/stream" if stream else "message/send"
        
        a2a_request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id,
            "params": {
                "message": a2a_message,
            },
        }

        # Add configuration if applicable
        config = {}
        if optional_params.get("blocking") is not None:
            config["blocking"] = optional_params["blocking"]
        if config:
            a2a_request["params"]["configuration"] = config

        verbose_logger.debug(f"A2A request: {json.dumps(a2a_request, indent=2)}")
        
        return a2a_request

    def _transform_messages_to_a2a(
        self, messages: List[AllMessageValues]
    ) -> Dict[str, Any]:
        """
        Transform OpenAI messages to a single A2A message.
        
        A2A typically works with single messages in a conversation context.
        We'll combine all messages into context and use the last user message.
        """
        # Find the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg
                break
        
        if last_user_message is None:
            # If no user message, use the last message
            last_user_message = messages[-1] if messages else {"role": "user", "content": ""}

        # Transform to A2A parts format
        content = last_user_message.get("content", "")
        
        # Handle content that might be a list (multimodal)
        parts = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append({
                            "kind": "text",
                            "text": item.get("text", ""),
                        })
                    elif item.get("type") == "image_url":
                        # A2A supports file parts for images
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                        parts.append({
                            "kind": "file",
                            "file": {
                                "uri": url,
                                "mimeType": "image/*",
                            },
                        })
                elif isinstance(item, str):
                    parts.append({"kind": "text", "text": item})
        else:
            parts.append({"kind": "text", "text": str(content)})

        # Build A2A message
        a2a_message = {
            "role": self._map_role_to_a2a(last_user_message.get("role", "user")),
            "parts": parts,
            "messageId": uuid4().hex,
        }

        return a2a_message

    def _map_role_to_a2a(self, openai_role: str) -> str:
        """Map OpenAI role to A2A role."""
        role_mapping = {
            "user": "user",
            "assistant": "agent",
            "system": "user",  # A2A doesn't have system role, treat as user context
            "tool": "user",
            "function": "user",
        }
        return role_mapping.get(openai_role, "user")

    # ========================================================================
    # Response Transformation: A2A -> OpenAI
    # ========================================================================

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        api_key: Optional[str],
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        json_mode: bool = False,
    ) -> ModelResponse:
        """
        Transform A2A response to OpenAI chat completion format.
        
        A2A response format:
            {
                "jsonrpc": "2.0",
                "id": "request-id",
                "result": {
                    "task": {...} or
                    "message": {
                        "role": "agent",
                        "parts": [{"kind": "text", "text": "Hello!"}],
                        "messageId": "msg-id"
                    }
                }
            }
        
        OpenAI response format:
            {
                "id": "chatcmpl-xxx",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop"
                }],
                "usage": {...}
            }
        """
        try:
            response_json = raw_response.json()
        except json.JSONDecodeError as e:
            raise A2AAgentError(
                status_code=500,
                message=f"Failed to parse A2A response: {str(e)}",
                headers=dict(raw_response.headers),
            )

        verbose_logger.debug(f"A2A response: {json.dumps(response_json, indent=2)}")

        # Check for JSON-RPC error
        if "error" in response_json:
            error = response_json["error"]
            raise A2AAgentError(
                status_code=error.get("code", 500),
                message=error.get("message", "Unknown A2A error"),
                headers=dict(raw_response.headers),
            )

        result = response_json.get("result", {})
        
        # Extract content from A2A response
        content = self._extract_content_from_a2a_result(result)
        finish_reason = self._determine_finish_reason(result)

        # Build OpenAI response
        model_response.id = response_json.get("id", f"chatcmpl-{uuid4().hex[:8]}")
        model_response.object = "chat.completion"
        model_response.created = int(time.time())
        model_response.model = model
        
        model_response.choices = [
            Choices(
                index=0,
                message=Message(
                    role="assistant",
                    content=content,
                ),
                finish_reason=finish_reason,
            )
        ]

        # Estimate token usage (A2A doesn't provide this)
        prompt_tokens = self._estimate_tokens(messages)
        completion_tokens = self._estimate_tokens_from_text(content)
        
        model_response.usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        return model_response

    def _extract_content_from_a2a_result(self, result: Dict[str, Any]) -> str:
        """Extract text content from A2A result."""
        content_parts = []
        
        # Check the kind of result
        result_kind = result.get("kind", "")
        
        # Handle direct message response (kind: "message")
        # The result itself contains role and parts when kind is "message"
        if result_kind == "message" or (result.get("role") and "parts" in result):
            parts = result.get("parts", [])
            for part in parts:
                if part.get("kind") == "text":
                    content_parts.append(part.get("text", ""))
        
        # Check for nested message response (some implementations)
        elif "message" in result:
            message = result["message"]
            parts = message.get("parts", [])
            for part in parts:
                if part.get("kind") == "text":
                    content_parts.append(part.get("text", ""))
        
        # Check for task with artifacts
        elif "task" in result or result_kind == "task":
            task = result.get("task", result) if "task" in result else result
            artifacts = task.get("artifacts", [])
            for artifact in artifacts:
                for part in artifact.get("parts", []):
                    if part.get("kind") == "text":
                        content_parts.append(part.get("text", ""))
            
            # Also check task status message
            status = task.get("status", {})
            status_message = status.get("message", {})
            for part in status_message.get("parts", []):
                if part.get("kind") == "text":
                    content_parts.append(part.get("text", ""))
        
        # Check for artifacts at root level
        elif "artifact" in result:
            artifact = result["artifact"]
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    content_parts.append(part.get("text", ""))
        
        # Fallback: check for parts directly on result
        elif "parts" in result:
            for part in result.get("parts", []):
                if part.get("kind") == "text":
                    content_parts.append(part.get("text", ""))

        return "\n".join(content_parts) if content_parts else ""

    def _determine_finish_reason(self, result: Dict[str, Any]) -> str:
        """Determine the finish reason from A2A result."""
        # Check task status
        if "task" in result:
            status = result["task"].get("status", {})
            state = status.get("state", "")
            
            state_mapping = {
                "completed": "stop",
                "failed": "stop",
                "canceled": "stop",
                "rejected": "stop",
                "input_required": "stop",  # Needs more input
                "working": "stop",
                "submitted": "stop",
            }
            return state_mapping.get(state.lower().replace("task_state_", ""), "stop")
        
        return "stop"

    def _estimate_tokens(self, messages: List[AllMessageValues]) -> int:
        """Estimate token count for messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4  # Rough estimate
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        total += len(item["text"]) // 4
        return max(total, 1)

    def _estimate_tokens_from_text(self, text: str) -> int:
        """Estimate token count from text."""
        return max(len(text) // 4, 1)

    # ========================================================================
    # Streaming Support
    # ========================================================================

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str]],
        sync_stream: bool,
        json_mode: bool = False,
    ):
        """
        Get an iterator that transforms A2A streaming events to OpenAI format.
        """
        from litellm.llms.a2a.chat.streaming import A2AStreamingIterator
        
        return A2AStreamingIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
