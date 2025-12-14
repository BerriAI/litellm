"""
Transformation for LangGraph API.

LangGraph provides streaming (/runs/stream) and non-streaming (/runs/wait) endpoints
for running agents.

Streaming endpoint: POST /runs/stream
Non-streaming endpoint: POST /runs/wait
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.langgraph.chat.sse_iterator import LangGraphSSEStreamIterator
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.utils import CustomStreamWrapper

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any
    HTTPHandler = Any
    AsyncHTTPHandler = Any
    CustomStreamWrapper = Any


class LangGraphError(BaseLLMException):
    """Exception class for LangGraph API errors."""

    pass


class LangGraphConfig(BaseConfig):
    """
    Configuration for LangGraph API.

    LangGraph is a framework for building stateful, multi-actor applications with LLMs.
    It provides a streaming and non-streaming API for running agents.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get LangGraph API base and key from params or environment.

        Returns:
            Tuple of (api_base, api_key)
        """
        from litellm.secret_managers.main import get_secret_str

        api_base = (
            api_base
            or get_secret_str("LANGGRAPH_API_BASE")
            or "http://localhost:2024"
        )

        api_key = api_key or get_secret_str("LANGGRAPH_API_KEY")

        return api_base, api_key

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        LangGraph supports minimal OpenAI params since it's an agent runtime.
        """
        return ["stream"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to LangGraph params.
        """
        return optional_params

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
        Get the complete URL for the LangGraph request.

        Streaming: /runs/stream
        Non-streaming: /runs/wait
        """
        if api_base is None:
            raise ValueError(
                "api_base is required for LangGraph. Set it via LANGGRAPH_API_BASE env var or api_base parameter."
            )

        # Remove trailing slash if present
        api_base = api_base.rstrip("/")

        # Choose endpoint based on streaming mode
        if stream:
            return f"{api_base}/runs/stream"
        else:
            return f"{api_base}/runs/wait"

    def _get_assistant_id(self, model: str, optional_params: dict) -> str:
        """
        Get the assistant ID from model or optional_params.

        model format: "langgraph/assistant_id" or just "assistant_id"
        """
        assistant_id = optional_params.get("assistant_id")
        if assistant_id:
            return assistant_id

        # Extract from model name
        if "/" in model:
            parts = model.split("/", 1)
            if len(parts) == 2:
                return parts[1]
        return model

    def _convert_messages_to_langgraph_format(
        self, messages: List[AllMessageValues]
    ) -> List[Dict[str, str]]:
        """
        Convert OpenAI-format messages to LangGraph format.

        OpenAI format: {"role": "user", "content": "..."}
        LangGraph format: {"role": "human", "content": "..."}
        """
        langgraph_messages: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Convert OpenAI roles to LangGraph roles
            if role == "user":
                langgraph_role = "human"
            elif role == "assistant":
                langgraph_role = "assistant"
            elif role == "system":
                langgraph_role = "system"
            else:
                langgraph_role = "human"

            # Handle content that might be a list
            if isinstance(content, list):
                content = convert_content_list_to_str(msg)
            
            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content)

            langgraph_messages.append({"role": langgraph_role, "content": content})

        return langgraph_messages

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request to LangGraph format.

        LangGraph request format:
        {
            "assistant_id": "agent",
            "input": {
                "messages": [{"role": "human", "content": "..."}]
            },
            "stream_mode": "messages-tuple"  # for streaming
        }
        """
        assistant_id = self._get_assistant_id(model, optional_params)
        langgraph_messages = self._convert_messages_to_langgraph_format(messages)

        payload: Dict[str, Any] = {
            "assistant_id": assistant_id,
            "input": {"messages": langgraph_messages},
        }

        # Add stream_mode for streaming requests
        stream = litellm_params.get("stream", False)
        if stream:
            stream_mode = optional_params.get("stream_mode", "messages-tuple")
            payload["stream_mode"] = stream_mode

        # Add optional config if provided
        if "config" in optional_params:
            payload["config"] = optional_params["config"]

        # Add optional metadata if provided
        if "metadata" in optional_params:
            payload["metadata"] = optional_params["metadata"]

        # Add thread_id if provided (for stateful conversations)
        if "thread_id" in optional_params:
            payload["thread_id"] = optional_params["thread_id"]

        verbose_logger.debug(f"LangGraph request payload: {payload}")
        return payload

    def _extract_content_from_response(self, response_json: dict) -> str:
        """
        Extract content from LangGraph non-streaming response.

        Response format varies, but commonly:
        {
            "messages": [...],  # or could be in different structure
            "values": {...}
        }
        """
        # Try to get the last AI message from the response
        messages = response_json.get("messages", [])
        if isinstance(messages, list) and messages:
            # Find the last AI/assistant message
            for msg in reversed(messages):
                if isinstance(msg, dict):
                    msg_type = msg.get("type", "")
                    role = msg.get("role", "")
                    if msg_type == "ai" or role == "assistant":
                        return msg.get("content", "")

        # Check values for output
        values = response_json.get("values", {})
        if isinstance(values, dict):
            output_messages = values.get("messages", [])
            if isinstance(output_messages, list) and output_messages:
                for msg in reversed(output_messages):
                    if isinstance(msg, dict):
                        msg_type = msg.get("type", "")
                        if msg_type == "ai":
                            return msg.get("content", "")

        # Fallback: try to serialize the whole response
        verbose_logger.warning(
            "Could not extract content from LangGraph response, returning raw"
        )
        return json.dumps(response_json)

    def get_streaming_response(
        self,
        model: str,
        raw_response: httpx.Response,
    ) -> LangGraphSSEStreamIterator:
        """
        Return a streaming iterator for SSE responses.
        """
        return LangGraphSSEStreamIterator(response=raw_response, model=model)

    def get_sync_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional[Union[HTTPHandler, "AsyncHTTPHandler"]] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> CustomStreamWrapper:
        """
        Get a CustomStreamWrapper for synchronous streaming.
        """
        from litellm.llms.custom_httpx.http_handler import (
            HTTPHandler,
            _get_httpx_client,
        )
        from litellm.utils import CustomStreamWrapper

        if client is None or not isinstance(client, HTTPHandler):
            client = _get_httpx_client(params={})

        verbose_logger.debug(f"Making sync streaming request to: {api_base}")

        # Make streaming request
        response = client.post(
            api_base,
            headers=headers,
            data=json.dumps(data),
            stream=True,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise LangGraphError(
                status_code=response.status_code, message=str(response.read())
            )

        # Create iterator for SSE stream
        completion_stream = self.get_streaming_response(
            model=model, raw_response=response
        )

        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )

        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return streaming_response

    async def get_async_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional["AsyncHTTPHandler"] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> CustomStreamWrapper:
        """
        Get a CustomStreamWrapper for asynchronous streaming.
        """
        from litellm.llms.custom_httpx.http_handler import (
            AsyncHTTPHandler,
            get_async_httpx_client,
        )
        from litellm.utils import CustomStreamWrapper

        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(
                llm_provider=cast(Any, "langgraph"), params={}
            )

        verbose_logger.debug(f"Making async streaming request to: {api_base}")

        # Make async streaming request
        response = await client.post(
            api_base,
            headers=headers,
            data=json.dumps(data),
            stream=True,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise LangGraphError(
                status_code=response.status_code, message=str(await response.aread())
            )

        # Create iterator for SSE stream
        completion_stream = self.get_streaming_response(
            model=model, raw_response=response
        )

        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )

        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return streaming_response

    @property
    def has_custom_stream_wrapper(self) -> bool:
        """Indicates that this config has custom streaming support."""
        return True

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        """
        LangGraph does not use a stream param in request body.
        Streaming is determined by the endpoint URL.
        """
        return False

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform the LangGraph response to LiteLLM ModelResponse format.
        """
        try:
            response_json = raw_response.json()
            verbose_logger.debug(f"LangGraph response: {response_json}")

            content = self._extract_content_from_response(response_json)

            # Create the message
            message = Message(content=content, role="assistant")

            # Create choices
            choice = Choices(finish_reason="stop", index=0, message=message)

            # Update model response
            model_response.choices = [choice]
            model_response.model = model

            # LangGraph doesn't provide token usage, so we estimate it
            try:
                from litellm.utils import token_counter

                prompt_tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
                completion_tokens = token_counter(
                    model="gpt-3.5-turbo", text=content, count_response_tokens=True
                )
                total_tokens = prompt_tokens + completion_tokens

                usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                setattr(model_response, "usage", usage)
            except Exception as e:
                verbose_logger.warning(f"Failed to calculate token usage: {str(e)}")

            return model_response

        except Exception as e:
            verbose_logger.error(f"Error processing LangGraph response: {str(e)}")
            raise LangGraphError(
                message=f"Error processing response: {str(e)}",
                status_code=raw_response.status_code,
            )

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
        Validate and set up environment for LangGraph requests.
        """
        headers["Content-Type"] = "application/json"

        # Add API key if provided
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return LangGraphError(status_code=status_code, message=error_message)

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """
        LangGraph has native streaming support, so we don't need to fake stream.
        """
        return False

