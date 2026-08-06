"""
Transformation for Vertex AI Agent Engine (Reasoning Engines)

Handles the transformation between LiteLLM's OpenAI-compatible format and
Vertex AI Reasoning Engine's API format.

API Reference:
- :query endpoint - for session management (create, get, list, delete)
- :streamQuery endpoint - for actual queries (stream_query method)
"""

import json
from typing import TYPE_CHECKING, Any, Final, Optional, Union, cast

import httpx

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.vertex_ai.agent_engine.sse_iterator import (
    VertexAgentEngineResponseIterator,
)
from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
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


class VertexAgentEngineError(BaseLLMException):
    """Exception for Vertex Agent Engine errors."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message=message, status_code=status_code)


class VertexAgentEngineConfig(BaseConfig, VertexBase):
    """
    Configuration for Vertex AI Agent Engine (Reasoning Engines).

    Model format: vertex_ai/agent_engine/<resource_id>
    Where resource_id is the numeric ID of the reasoning engine.
    """

    def __init__(self, **kwargs):
        BaseConfig.__init__(self, **kwargs)
        VertexBase.__init__(self)

    def get_supported_openai_params(self, model: str) -> list[str]:
        """Vertex Agent Engine has limited OpenAI compatible params."""
        return ["user"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """Map OpenAI params to Agent Engine params."""
        # Map 'user' to 'user_id' for session management
        if "user" in non_default_params:
            optional_params["user_id"] = non_default_params["user"]
        return optional_params

    def _parse_model_string(self, model: str) -> tuple[str, str]:
        """
        Parse model string to extract resource ID.

        Model format: agent_engine/<project_number>/<location>/<engine_id>
        Or: agent_engine/<engine_id> (uses default project/location)

        Returns: (resource_path, engine_id)
        """
        # Remove 'agent_engine/' prefix if present
        model = model.removeprefix("agent_engine/")

        # Check if it's a full resource path
        if model.startswith("projects/"):
            # Full path: projects/123/locations/us-central1/reasoningEngines/456
            return model, model.split("/")[-1]

        # Just the engine ID
        return model, model

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        Get the complete URL for the request.

        For Vertex Agent Engine:
        - Non-streaming: :query endpoint (for session management)
        - Streaming: :streamQuery endpoint (for actual queries)
        """
        resource_path, engine_id = self._parse_model_string(model)

        # Get project and location from litellm_params or environment
        vertex_project: Final = self.safe_get_vertex_ai_project(litellm_params)
        vertex_location: Final = self.safe_get_vertex_ai_location(litellm_params) or "us-central1"

        # Build the full resource path if only engine_id was provided
        if not resource_path.startswith("projects/"):
            if not vertex_project:
                raise ValueError(
                    "vertex_project is required for Vertex Agent Engine. "
                    "Set via litellm_params['vertex_project'] or VERTEXAI_PROJECT env var."
                )
            resource_path = f"projects/{vertex_project}/locations/{vertex_location}/reasoningEngines/{engine_id}"

        base_url: Final = get_vertex_base_url(vertex_location)

        # Always use :streamQuery endpoint for actual queries
        # The :query endpoint only supports session management methods
        # (create_session, get_session, list_sessions, delete_session, etc.)
        endpoint: Final = f"{base_url}/v1beta1/{resource_path}:streamQuery"

        verbose_logger.debug("Vertex Agent Engine URL: %s", endpoint)
        return endpoint

    def _get_auth_headers(
        self,
        optional_params: dict,
        litellm_params: dict,
    ) -> dict[str, str]:
        """Get authentication headers using Google Cloud credentials."""
        vertex_credentials: Final = self.safe_get_vertex_ai_credentials(litellm_params)
        vertex_project: Final = self.safe_get_vertex_ai_project(litellm_params)

        # Get access token using VertexBase
        access_token, project_id = self.get_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
        )

        verbose_logger.debug("Vertex Agent Engine: Authenticated for project %s", project_id)

        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    def _get_user_id(self, optional_params: dict) -> str:
        """Get or generate user ID for session management."""
        user_id: Final = optional_params.get("user_id") or optional_params.get("user")
        if user_id:
            return user_id
        # Generate a user ID
        return f"litellm-user-{str(uuid.uuid4())[:8]}"

    def _get_session_id(self, optional_params: dict) -> str | None:
        """Get session ID if provided."""
        return optional_params.get("session_id")

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request to Vertex Agent Engine format.

        The API expects:
        {
            "class_method": "stream_query",
            "input": {
                "message": "...",
                "user_id": "...",
                "session_id": "..." (optional)
            }
        }
        """
        # Use the last message content as the prompt
        prompt: Final = convert_content_list_to_str(messages[-1])

        # Get user_id and session_id
        user_id: Final = self._get_user_id(optional_params)
        session_id: Final = self._get_session_id(optional_params)

        # Build the input
        input_data: Final[dict[str, Any]] = {
            "message": prompt,
            "user_id": user_id,
        }

        if session_id:
            input_data["session_id"] = session_id

        # Build the request payload
        # Note: stream_query is used for both streaming and non-streaming
        # The difference is the endpoint (:streamQuery vs :query)
        payload: Final = {
            "class_method": "stream_query",
            "input": input_data,
        }

        verbose_logger.debug("Vertex Agent Engine payload: %s", payload)
        return payload

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        """Validate environment and set up authentication headers."""
        auth_headers: Final = self._get_auth_headers(optional_params, litellm_params)
        headers.update(auth_headers)
        return headers

    def _extract_text_from_response(self, response_data: dict) -> str:
        """Extract text content from the response."""
        # Try to get from content.parts
        content: Final = response_data.get("content", {})
        parts: Final = content.get("parts", [])
        for part in parts:
            if "text" in part:
                return part["text"]

        # Try actions.state_delta
        actions: Final = response_data.get("actions", {})
        state_delta: Final = actions.get("state_delta", {})
        for key, value in state_delta.items():
            if isinstance(value, str) and value:
                return value

        return ""

    def _calculate_usage(self, model: str, messages: list[AllMessageValues], content: str) -> Usage | None:
        """Calculate token usage using LiteLLM's token counter."""
        try:
            from litellm.utils import token_counter

            prompt_tokens: Final = token_counter(model="gpt-3.5-turbo", messages=messages)
            completion_tokens: Final = token_counter(model="gpt-3.5-turbo", text=content, count_response_tokens=True)
            total_tokens: Final = prompt_tokens + completion_tokens

            return Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception as e:
            verbose_logger.warning("Failed to calculate token usage: %s", e)
            return None

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        """
        Transform Vertex Agent Engine response to LiteLLM ModelResponse format.

        The response is a streaming SSE format even for non-streaming requests.
        We need to collect all the chunks and extract the final response.
        """
        try:
            content_type: Final = raw_response.headers.get("content-type", "").lower()
            verbose_logger.debug("Vertex Agent Engine response Content-Type: %s", content_type)

            # Parse the SSE response
            response_text: Final = raw_response.text
            verbose_logger.debug("Response (first 500 chars): %s", response_text[:500])

            # Extract content from SSE stream
            content = ""
            for line in response_text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        text = self._extract_text_from_response(data)
                        if text:
                            content = text  # Use the last non-empty text
                except json.JSONDecodeError:
                    continue

            # Create the message
            message: Final = Message(content=content, role="assistant")

            # Create choices
            choice: Final = Choices(finish_reason="stop", index=0, message=message)

            # Update model response
            model_response.choices = [choice]
            model_response.model = model

            # Calculate usage
            calculated_usage: Final = self._calculate_usage(model, messages, content)
            if calculated_usage:
                setattr(model_response, "usage", calculated_usage)

            return model_response

        except Exception as e:
            verbose_logger.error("Error processing Vertex Agent Engine response: %s", e)
            raise VertexAgentEngineError(
                message=f"Error processing response: {e}",
                status_code=raw_response.status_code,
            )

    def get_streaming_response(
        self,
        model: str,
        raw_response: httpx.Response,
    ) -> VertexAgentEngineResponseIterator:
        """Return a streaming iterator for SSE responses."""
        return VertexAgentEngineResponseIterator(
            streaming_response=raw_response.iter_lines(),
            sync_stream=True,
        )

    def get_sync_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Union[HTTPHandler, "AsyncHTTPHandler"] | None = None,
        json_mode: bool | None = None,
        signed_json_body: bytes | None = None,
    ) -> "CustomStreamWrapper":
        """Get a CustomStreamWrapper for synchronous streaming."""
        from litellm.llms.custom_httpx.http_handler import (
            HTTPHandler,
            _get_httpx_client,
        )
        from litellm.utils import CustomStreamWrapper

        if client is None or not isinstance(client, HTTPHandler):
            client = _get_httpx_client(params={})

        # Avoid logging sensitive api_base directly
        verbose_logger.debug("Making sync streaming request to Vertex AI endpoint.")

        # Make streaming request
        response: Final = client.post(
            api_base,
            headers=headers,
            data=json.dumps(data),
            stream=True,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise VertexAgentEngineError(status_code=response.status_code, message=str(response.read()))

        # Create iterator for SSE stream
        completion_stream: Final = self.get_streaming_response(model=model, raw_response=response)

        streaming_response: Final = CustomStreamWrapper(
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
        json_mode: bool | None = None,
        signed_json_body: bytes | None = None,
    ) -> "CustomStreamWrapper":
        """Get a CustomStreamWrapper for asynchronous streaming."""
        from litellm.llms.custom_httpx.http_handler import (
            AsyncHTTPHandler,
            get_async_httpx_client,
        )
        from litellm.utils import CustomStreamWrapper

        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(llm_provider=cast(Any, "vertex_ai"), params={})

        # Avoid logging sensitive api_base directly
        verbose_logger.debug("Making async streaming request to Vertex AI endpoint.")

        # Make async streaming request
        response: Final = await client.post(
            api_base,
            headers=headers,
            data=json.dumps(data),
            stream=True,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise VertexAgentEngineError(status_code=response.status_code, message=str(await response.aread()))

        # Create iterator for SSE stream (async)
        completion_stream: Final = VertexAgentEngineResponseIterator(
            streaming_response=response.aiter_lines(),
            sync_stream=False,
        )

        streaming_response: Final = CustomStreamWrapper(
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
        """Agent Engine does not allow passing `stream` in the request body."""
        return False

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return VertexAgentEngineError(status_code=status_code, message=error_message)

    def should_fake_stream(
        self,
        model: str | None,
        stream: bool | None,
        custom_llm_provider: str | None = None,
    ) -> bool:
        """Agent Engine always returns SSE streams, so we use real streaming."""
        return False
