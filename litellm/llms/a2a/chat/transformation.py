"""
A2A Protocol Transformation for LiteLLM
"""

import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.llms.azure_ai.common_utils import AZURE_ENTRA_LITELLM_PARAM_KEYS, get_azure_ai_agent_entra_token
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from ..common_utils import (
    A2A_TASK_FAILURE_STATES,
    A2A_TASK_INTERRUPTED_STATES,
    A2A_TASK_RECOVERABLE_STATES,
    A2A_TASK_STATE_COMPLETED,
    A2AError,
    a2a_hop_uses_entra,
    a2a_task_state,
    async_poll_a2a_task_completion,
    convert_messages_to_prompt,
    extract_text_from_a2a_response,
    poll_a2a_task_completion,
)
from .streaming_iterator import A2AModelResponseIterator

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


_REGISTRY_PARAMS_KEPT_OUT_OF_OPTIONAL_PARAMS: Final = (
    frozenset({"api_key", "api_base", "headers", "model"}) | AZURE_ENTRA_LITELLM_PARAM_KEYS
)
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))


def _card_declares_no_streaming(agent_card_params: Mapping[str, object]) -> bool:
    capabilities: Final = agent_card_params.get("capabilities")
    return isinstance(capabilities, Mapping) and not capabilities.get("streaming")


def _agent_authenticates_with_entra(agent_litellm_params: Mapping[str, object]) -> bool:
    return a2a_hop_uses_entra(agent_litellm_params, agent_litellm_params.get("custom_llm_provider"))


def _registry_api_key(agent_litellm_params: Mapping[str, object]) -> str | None:
    if _agent_authenticates_with_entra(agent_litellm_params):
        return get_azure_ai_agent_entra_token(agent_litellm_params)
    configured_api_key: Final = agent_litellm_params.get("api_key")
    return configured_api_key if isinstance(configured_api_key, str) else None


def _registry_headers(agent_litellm_params: Mapping[str, object]) -> dict[str, object] | None:
    stored_headers: Final = agent_litellm_params.get("headers")
    if not isinstance(stored_headers, Mapping):
        return None
    entra_owns_authorization: Final = _agent_authenticates_with_entra(agent_litellm_params)
    return {
        name: value
        for name, value in stored_headers.items()
        if not (entra_owns_authorization and str(name).lower() == "authorization")
    }


class A2AConfig(BaseConfig):
    """
    Configuration for A2A (Agent-to-Agent) Protocol.

    Handles transformation between OpenAI and A2A JSON-RPC 2.0 formats.
    """

    @property
    def uses_async_transform_response(self) -> bool:
        return True

    @staticmethod
    def _parse_a2a_raw_response(raw_response: httpx.Response) -> dict[str, object]:
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise A2AError(
                status_code=raw_response.status_code,
                message=f"Failed to parse A2A response: {e}",
                headers=dict(raw_response.headers),
            )
        if not isinstance(response_json, dict):
            return response_json  # type: ignore[return-value]
        if "error" in response_json:
            error: Final = _JSON_OBJECT.validate_python(response_json["error"])
            raise A2AError(
                status_code=raw_response.status_code,
                message=f"A2A error: {error.get('message', 'Unknown error')}",
                headers=dict(raw_response.headers),
            )
        return response_json

    @classmethod
    def _validated_after_task_lifecycle(
        cls,
        *,
        response_json: dict[str, object],
        raw_response: httpx.Response,
        request_data: dict,
        litellm_params: dict,
        polled: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        if polled is not None:
            response_json = dict(polled)
        validated: Final = _JSON_OBJECT.validate_python(response_json)
        result: Final = validated.get("result")
        if isinstance(result, dict):
            state: Final[str | None] = a2a_task_state(result)
            if state in A2A_TASK_FAILURE_STATES:
                raise A2AError(
                    status_code=raw_response.status_code,
                    message=f"A2A task ended with state: {state}",
                    headers=dict(raw_response.headers),
                )
            if state in A2A_TASK_INTERRUPTED_STATES:
                raise A2AError(
                    status_code=raw_response.status_code,
                    message=f"A2A task requires further action (state: {state})",
                    headers=dict(raw_response.headers),
                )
            if polled is None and state in A2A_TASK_RECOVERABLE_STATES:
                task_id: Final = result.get("id")
                api_base: Final = litellm_params.get("api_base")
                if not (isinstance(task_id, str) and isinstance(api_base, str)):
                    raise A2AError(
                        status_code=raw_response.status_code,
                        message=f"A2A task is not complete (state: {state})",
                        headers=dict(raw_response.headers),
                    )
                poll_headers: Final = litellm_params.get("headers")
                timeout: Final = float(litellm_params.get("timeout") or litellm_params.get("request_timeout") or 60)
                request_id: Final = str(validated.get("id") or request_data.get("id") or uuid.uuid4())
                polled_result: Final = poll_a2a_task_completion(
                    api_base=api_base,
                    task_id=task_id,
                    request_id=request_id,
                    headers=(
                        {str(k): str(v) for k, v in poll_headers.items()}
                        if isinstance(poll_headers, Mapping)
                        else None
                    ),
                    timeout=timeout,
                )
                return cls._validated_after_task_lifecycle(
                    response_json=dict(polled_result),
                    raw_response=raw_response,
                    request_data=request_data,
                    litellm_params=litellm_params,
                    polled=polled_result,
                )
            if state is not None and state != A2A_TASK_STATE_COMPLETED:
                raise A2AError(
                    status_code=raw_response.status_code,
                    message=f"A2A task is not complete (state: {state})",
                    headers=dict(raw_response.headers),
                )
        return validated

    @classmethod
    async def _validated_after_task_lifecycle_async(
        cls,
        *,
        response_json: dict[str, object],
        raw_response: httpx.Response,
        request_data: dict,
        litellm_params: dict,
        polled: Mapping[str, object] | None,
        async_http_client: "AsyncHTTPHandler | None",
    ) -> Mapping[str, object]:
        if polled is not None:
            response_json = dict(polled)
        validated: Final = _JSON_OBJECT.validate_python(response_json)
        result: Final = validated.get("result")
        if isinstance(result, dict):
            state: Final[str | None] = a2a_task_state(result)
            if state in A2A_TASK_FAILURE_STATES:
                raise A2AError(
                    status_code=raw_response.status_code,
                    message=f"A2A task ended with state: {state}",
                    headers=dict(raw_response.headers),
                )
            if state in A2A_TASK_INTERRUPTED_STATES:
                raise A2AError(
                    status_code=raw_response.status_code,
                    message=f"A2A task requires further action (state: {state})",
                    headers=dict(raw_response.headers),
                )
            if polled is None and state in A2A_TASK_RECOVERABLE_STATES:
                task_id: Final = result.get("id")
                api_base: Final = litellm_params.get("api_base")
                if not (isinstance(task_id, str) and isinstance(api_base, str)):
                    raise A2AError(
                        status_code=raw_response.status_code,
                        message=f"A2A task is not complete (state: {state})",
                        headers=dict(raw_response.headers),
                    )
                poll_headers: Final = litellm_params.get("headers")
                timeout: Final = float(litellm_params.get("timeout") or litellm_params.get("request_timeout") or 60)
                request_id: Final = str(validated.get("id") or request_data.get("id") or uuid.uuid4())
                polled_result: Final = await async_poll_a2a_task_completion(
                    api_base=api_base,
                    task_id=task_id,
                    request_id=request_id,
                    headers=(
                        {str(k): str(v) for k, v in poll_headers.items()}
                        if isinstance(poll_headers, Mapping)
                        else None
                    ),
                    timeout=timeout,
                    client=async_http_client,
                )
                return await cls._validated_after_task_lifecycle_async(
                    response_json=dict(polled_result),
                    raw_response=raw_response,
                    request_data=request_data,
                    litellm_params=litellm_params,
                    polled=polled_result,
                    async_http_client=async_http_client,
                )
            if state is not None and state != A2A_TASK_STATE_COMPLETED:
                raise A2AError(
                    status_code=raw_response.status_code,
                    message=f"A2A task is not complete (state: {state})",
                    headers=dict(raw_response.headers),
                )
        return validated

    @staticmethod
    def _fill_model_response(
        *,
        model: str,
        validated: Mapping[str, object],
        response_json: Mapping[str, object],
        model_response: ModelResponse,
        messages: list[AllMessageValues],
    ) -> ModelResponse:
        text: Final = extract_text_from_a2a_response(validated)
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
        model_response.model = model
        response_id: Final = validated.get("id", response_json.get("id", str(uuid.uuid4())))
        model_response.id = str(response_id)
        try:
            from litellm.utils import token_counter

            prompt_tokens: Final = token_counter(model="gpt-3.5-turbo", messages=messages)
            completion_tokens: Final = token_counter(model="gpt-3.5-turbo", text=text, count_response_tokens=True)
            setattr(
                model_response,
                "usage",
                Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        except Exception:  # noqa: BLE001
            pass
        return model_response

    @staticmethod
    def resolve_agent_config_from_registry(
        agent_name: str,
        api_base: str | None,
        api_key: str | None,
        headers: dict[str, object] | None,
        optional_params: dict[str, object],
    ) -> tuple[str | None, str | None, dict[str, object] | None]:
        """
        Resolve agent configuration from the registry for a registered agent.

        Args:
            agent_name: The model string with the provider prefix already stripped by
                get_llm_provider ("a2a/my-agent" -> "my-agent"), the name the agent was
                registered under
            api_base: Explicit api_base (takes precedence over registry)
            api_key: Explicit api_key (takes precedence over registry)
            headers: Explicit headers (takes precedence over registry)
            optional_params: Dict to merge additional litellm_params into

        Returns:
            Tuple of (api_base, api_key, headers) with registry values filled in
        """
        if not agent_name or (api_base is not None and api_key is not None and headers):
            return api_base, api_key, headers

        # Try registry lookup (only available in proxy context)
        try:
            from litellm.proxy.agent_endpoints.agent_registry import (
                global_agent_registry,
            )

            agent: Final = global_agent_registry.get_agent_by_name(agent_name)
            if agent:
                # Get api_base from agent card URL
                if api_base is None and agent.agent_card_params:
                    api_base = agent.agent_card_params.get("url")

                # Get api_key, headers, and other params from litellm_params
                if agent.litellm_params:
                    if api_key is None:
                        api_key = _registry_api_key(agent.litellm_params)

                    if not headers:
                        headers = _registry_headers(agent.litellm_params) or headers

                # Merge other litellm_params (timeout, max_retries, etc.)
                registry_params: Final = tuple(
                    (key, value)
                    for key, value in (agent.litellm_params.items() if agent.litellm_params else ())
                    if key not in _REGISTRY_PARAMS_KEPT_OUT_OF_OPTIONAL_PARAMS and key not in optional_params
                )
                streaming_fallback: Final = (
                    (("stream", False), ("fake_stream", True))
                    if optional_params.get("stream") and _card_declares_no_streaming(agent.agent_card_params)
                    else ()
                )
                optional_params.update((*registry_params, *streaming_fallback))
        except ImportError:
            pass  # Registry not available (not running in proxy context)

        return api_base, api_key, headers

    def get_supported_openai_params(self, model: str) -> list[str]:
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

        For A2A protocol, we need to map the stream parameter so
        transform_request can determine which JSON-RPC method to use.
        """
        # Map stream parameter
        for param, value in non_default_params.items():
            if param == "stream" and value is True:
                optional_params["stream"] = value

        return optional_params

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
            A new headers dict; the caller's dict is left untouched
        """
        content_type_default: Final = (
            () if "content-type" in headers or "Content-Type" in headers else (("Content-Type", "application/json"),)
        )
        bearer: Final = () if api_key is None else (("Authorization", f"Bearer {api_key}"),)
        return dict((*headers.items(), *content_type_default, *bearer))

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
        messages: list[AllMessageValues],
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
        request_id: Final = str(uuid.uuid4())

        if not messages:
            raise ValueError("At least one message is required for A2A completion")

        # Convert all messages to maintain conversation history
        # Use helper to format conversation with role prefixes
        full_context: Final = convert_messages_to_prompt(messages)

        # Create single A2A message with full conversation context
        a2a_message: Final = {
            "kind": "message",
            "role": "user",
            "parts": [{"kind": "text", "text": full_context}],
            "messageId": str(uuid.uuid4()),
        }

        # Build JSON-RPC 2.0 request
        # For A2A protocol, the method is "message/send" for non-streaming
        # and "message/stream" for streaming
        stream: Final = optional_params.get("stream", False)
        method: Final = "message/stream" if stream else "message/send"

        params: Final = (
            {"message": a2a_message} if stream else {"message": a2a_message, "configuration": {"blocking": True}}
        )
        request_data: Final = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        return request_data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
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
        response_json: Final = self._parse_a2a_raw_response(raw_response)
        validated: Final = self._validated_after_task_lifecycle(
            response_json=response_json,
            raw_response=raw_response,
            request_data=request_data,
            litellm_params=litellm_params,
            polled=None,
        )
        return self._fill_model_response(
            model=model,
            validated=validated,
            response_json=response_json if validated is response_json else validated,
            model_response=model_response,
            messages=messages,
        )

    async def async_transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
        async_http_client: object | None = None,
    ) -> ModelResponse:
        response_json: Final = self._parse_a2a_raw_response(raw_response)
        validated: Final = await self._validated_after_task_lifecycle_async(
            response_json=response_json,
            raw_response=raw_response,
            request_data=request_data,
            litellm_params=litellm_params,
            polled=None,
            async_http_client=async_http_client,  # type: ignore[arg-type]
        )
        return self._fill_model_response(
            model=model,
            validated=validated,
            response_json=response_json if validated is response_json else validated,
            model_response=model_response,
            messages=messages,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
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

    def _openai_message_to_a2a_message(self, message: Mapping[str, object]) -> dict[str, object]:
        """
        Convert OpenAI message to A2A message format.

        Args:
            message: OpenAI message dict

        Returns:
            A2A message dict
        """
        content: Final = message.get("content", "")
        role: Final = message.get("role", "user")

        return {
            "role": role,
            "parts": [{"kind": "text", "text": str(content)}],
            "messageId": str(uuid.uuid4()),
        }

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        """Return appropriate error class for A2A errors"""
        # Convert headers to dict if needed
        headers_dict: Final = dict(headers) if isinstance(headers, httpx.Headers) else headers
        return A2AError(
            status_code=status_code,
            message=error_message,
            headers=headers_dict,
        )
