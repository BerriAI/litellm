"""
Transformation for IBM watsonx.ai Orchestrate Agent API.

Model format: watsonx_agent/<agent_id>

API Reference: https://developer.watson-orchestrate.ibm.com/apis/orchestrate-agent/chat-with-agents
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.watsonx_agents import (
    WatsonxAgentChoice,
    WatsonxAgentMessage,
    WatsonxAgentResponse,
)
from litellm.types.utils import Choices, Message, ModelResponse

from ..common_utils import IBMWatsonXMixin, WatsonXAIError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any
    HTTPHandler = Any
    AsyncHTTPHandler = Any


class IBMWatsonXAgentConfig(IBMWatsonXMixin, BaseConfig):
    """Configuration for IBM watsonx.ai Orchestrate Agent API."""

    def __init__(self, **kwargs):
        BaseConfig.__init__(self, **kwargs)

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Return list of OpenAI parameters supported by watsonx agents.

        Currently, watsonx agents support a limited set of parameters.
        Most standard OpenAI completion parameters are not directly supported.
        """
        return [
            "stream",  # Streaming support
            "messages",  # Required messages parameter
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to watsonx agent parameters.

        Most OpenAI parameters don't have direct equivalents in watsonx agents,
        so we'll pass through only what's supported.
        """
        # Extract thread_id if provided
        thread_id = non_default_params.pop("thread_id", None)
        if thread_id:
            optional_params["thread_id"] = thread_id

        # Extract additional_parameters if provided
        additional_parameters = non_default_params.pop("additional_parameters", None)
        if additional_parameters:
            optional_params["additional_parameters"] = additional_parameters

        # Extract context if provided
        context = non_default_params.pop("context", None)
        if context:
            optional_params["context"] = context

        return optional_params

    def _get_agent_id(self, model: str) -> str:
        """
        Extract agent_id from model string.

        Expected format: "watsonx_agent/{agent_id}"
        Example: "watsonx_agent/abc123"
        """
        if "/" in model:
            parts = model.split("/")
            if len(parts) >= 2:
                return parts[1]
        raise ValueError(
            f"Invalid model format for watsonx agent: {model}. "
            "Expected format: 'watsonx_agent/AGENT_ID'"
        )

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
        Build the complete URL for the watsonx agent API.

        URL format: https://{api_endpoint}/api/v1/orchestrate/{agent_id}/chat/completions
        """
        base_url = self._get_base_url(api_base=api_base)
        agent_id = self._get_agent_id(model)

        url = f"{base_url.rstrip('/')}/api/v1/orchestrate/{agent_id}/chat/completions"

        verbose_logger.debug(f"Watsonx Agent API URL: {url}")
        return url

    def _transform_messages(
        self, messages: List[AllMessageValues]
    ) -> List[WatsonxAgentMessage]:
        """
        Transform OpenAI-style messages to watsonx agent format.

        Args:
            messages: List of OpenAI-style message dictionaries

        Returns:
            List of watsonx agent message dictionaries
        """
        transformed_messages: List[WatsonxAgentMessage] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Convert content to string if it's a list
            if isinstance(content, list):
                content = convert_content_list_to_str(msg)

            transformed_msg: WatsonxAgentMessage = {
                "role": role,
                "content": content,  # type: ignore
            }
            transformed_messages.append(transformed_msg)

        return transformed_messages

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform LiteLLM request to watsonx agent API format.

        Args:
            model: Model identifier
            messages: List of messages
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Request body for watsonx agent API
        """
        # Transform messages to agent format
        agent_messages = self._transform_messages(messages)

        # Build request body
        request_body: dict = {
            "messages": agent_messages,
            "additional_parameters": optional_params.get("additional_parameters", {}),
            "context": optional_params.get("context", {}),
            "stream": optional_params.get("stream", True),
        }

        verbose_logger.debug(
            f"Watsonx Agent request body: {request_body}"
        )
        return request_body

    def _transform_agent_choice_to_litellm(
        self, agent_choice: WatsonxAgentChoice, index: int
    ) -> Choices:
        """
        Transform a watsonx agent choice to LiteLLM format.

        Args:
            agent_choice: Agent response choice
            index: Choice index

        Returns:
            LiteLLM Choices object
        """
        agent_message = agent_choice.get("message") or {}
        content = agent_message.get("content", "")

        # Handle content that might be a list or dict
        if isinstance(content, (list, dict)):
            import json

            content = json.dumps(content)

        message = Message(
            content=str(content),
            role=agent_message.get("role", "assistant"),
        )

        finish_reason = agent_choice.get("finish_reason") or "stop"

        return Choices(
            finish_reason=finish_reason,
            index=index,
            message=message,
        )

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
        Transform watsonx agent API response to LiteLLM format.

        Args:
            model: Model identifier
            raw_response: Raw HTTP response
            model_response: LiteLLM ModelResponse object to populate
            logging_obj: Logging object
            request_data: Original request data
            messages: Original messages
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            encoding: Encoding
            api_key: API key
            json_mode: JSON mode flag

        Returns:
            Populated ModelResponse object
        """
        try:
            response_json: WatsonxAgentResponse = raw_response.json()
            verbose_logger.debug(f"Watsonx Agent response: {response_json}")

            # Extract response fields
            response_id = response_json.get("id", "")
            created = response_json.get("created", 0)
            response_model = response_json.get("model", model)
            thread_id = response_json.get("thread_id", "")

            # Transform choices
            agent_choices = response_json.get("choices", [])
            litellm_choices = [
                self._transform_agent_choice_to_litellm(choice, idx)
                for idx, choice in enumerate(agent_choices)
            ]

            # Update model response
            model_response.id = response_id
            model_response.created = created
            model_response.model = response_model
            model_response.choices = litellm_choices

            # Add thread_id as metadata
            if thread_id:
                model_response._hidden_params = model_response._hidden_params or {}
                model_response._hidden_params["thread_id"] = thread_id

            return model_response

        except Exception as e:
            verbose_logger.error(
                f"Error transforming watsonx agent response: {str(e)}"
            )
            raise WatsonXAIError(
                status_code=raw_response.status_code,
                message=f"Error transforming response: {str(e)}",
            )

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        """
        Validate environment and set up authentication headers.

        Args:
            headers: Request headers
            model: Model identifier
            messages: Messages
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            api_key: API key
            api_base: API base URL

        Returns:
            Updated headers with authentication
        """
        # Use the parent class method to validate and set up auth
        return super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[Dict, httpx.Headers],
    ) -> BaseLLMException:
        """
        Return appropriate error class for watsonx agent errors.

        Args:
            error_message: Error message
            status_code: HTTP status code
            headers: Response headers

        Returns:
            WatsonXAIError instance
        """
        return WatsonXAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    @staticmethod
    def completion(
        model: str,
        messages: List,
        api_base: str,
        api_key: Optional[str],
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        timeout: Union[float, int, Any],
        acompletion: bool,
        stream: Optional[bool] = False,
        headers: Optional[dict] = None,
    ) -> Any:
        """
        Dispatch method for watsonx agent completion.

        Routes to sync or async completion based on acompletion flag.

        Model format: watsonx_agent/<agent_id>

        Args:
            model: Model identifier (format: watsonx_agent/<agent_id>)
            messages: List of messages
            api_base: API base URL
            api_key: API key
            model_response: ModelResponse object to populate
            logging_obj: Logging object
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            timeout: Request timeout
            acompletion: Async completion flag
            stream: Streaming flag
            headers: Request headers

        Returns:
            ModelResponse object or async coroutine
        """
        from litellm.llms.watsonx.agents.handler import watsonx_agent_handler

        if acompletion:
            return watsonx_agent_handler.acompletion(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=api_key,
                model_response=model_response,
                logging_obj=logging_obj,
                optional_params=optional_params,
                litellm_params=litellm_params,
                timeout=timeout,
                headers=headers,
            )
        else:
            return watsonx_agent_handler.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=api_key,
                model_response=model_response,
                logging_obj=logging_obj,
                optional_params=optional_params,
                litellm_params=litellm_params,
                timeout=timeout,
                headers=headers,
            )
