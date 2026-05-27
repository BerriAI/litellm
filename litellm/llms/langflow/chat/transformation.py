"""LangFlow run API: POST {api_base}/api/v1/run/{flow_id}"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
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


class LangFlowError(BaseLLMException):
    """Exception class for LangFlow API errors."""

    pass


class LangFlowConfig(BaseConfig):
    """
    Configuration for the LangFlow API.

    LangFlow is a visual, low-code platform for building AI agents and pipelines.
    Each flow has a unique flow_id and is invoked via a simple HTTP endpoint.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        from litellm.secret_managers.main import get_secret_str

        api_base = (
            api_base or get_secret_str("LANGFLOW_API_BASE") or "http://localhost:7860"
        )
        api_key = api_key or get_secret_str("LANGFLOW_API_KEY")
        return api_base, api_key

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["stream"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    def _get_flow_id(self, model: str, optional_params: dict) -> str:
        """
        Extract the flow_id from the model string or optional_params.

        Model format: "langflow/{flow_id}"
        """
        flow_id = optional_params.get("flow_id")
        if flow_id:
            return flow_id

        if "/" in model:
            parts = model.split("/", 1)
            if len(parts) == 2:
                return parts[1]
        return model

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            raise ValueError(
                "api_base is required for LangFlow. Set it via LANGFLOW_API_BASE env var or api_base parameter."
            )

        api_base = api_base.rstrip("/")
        flow_id = self._get_flow_id(model, optional_params)
        return f"{api_base}/api/v1/run/{flow_id}"

    def _get_last_user_message(self, messages: List[AllMessageValues]) -> str:
        """Extract the text of the last user message to use as input_value."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = convert_content_list_to_str(msg)
                if not isinstance(content, str):
                    content = str(content)
                return content

        # Fallback: use last message regardless of role
        if messages:
            content = messages[-1].get("content", "")
            if isinstance(content, list):
                content = convert_content_list_to_str(messages[-1])
            if not isinstance(content, str):
                content = str(content)
            return content

        return ""

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request to LangFlow format.

        LangFlow request format:
        {
            "input_value": "<last user message>",
            "input_type": "chat",
            "output_type": "chat",
            "session_id": "<session_id>"
        }
        """
        input_value = self._get_last_user_message(messages)

        payload: Dict[str, Any] = {
            "input_value": input_value,
            "input_type": optional_params.get("input_type", "chat"),
            "output_type": optional_params.get("output_type", "chat"),
        }

        session_id = optional_params.get("session_id")
        if session_id:
            payload["session_id"] = session_id

        # Allow callers to pass tweaks (LangFlow component overrides)
        if "tweaks" in optional_params:
            payload["tweaks"] = optional_params["tweaks"]

        verbose_logger.debug(f"LangFlow request payload: {payload}")
        return payload

    def _extract_content_from_response(self, response_json: dict) -> str:
        """
        Extract the text content from a LangFlow response.

        Expected structure:
        {
          "outputs": [
            {
              "outputs": [
                {
                  "results": {
                    "message": {"text": "...", ...}
                  }
                }
              ]
            }
          ]
        }
        """
        try:
            outputs = response_json.get("outputs", [])
            if isinstance(outputs, list) and outputs:
                first_output = outputs[0]
                inner_outputs = first_output.get("outputs", [])
                if isinstance(inner_outputs, list) and inner_outputs:
                    first_inner = inner_outputs[0]

                    # Try results.message.text (most common Chat Output format)
                    results = first_inner.get("results", {})
                    if isinstance(results, dict):
                        message = results.get("message", {})
                        if isinstance(message, dict):
                            text = message.get("text", "")
                            if text:
                                return text

                    # Try outputs dict for other component types
                    outputs_dict = first_inner.get("outputs", {})
                    if isinstance(outputs_dict, dict):
                        for _key, val in outputs_dict.items():
                            if isinstance(val, dict):
                                msg = val.get("message", {})
                                if isinstance(msg, dict) and msg.get("text"):
                                    return msg["text"]
        except Exception as e:
            verbose_logger.warning(f"Could not parse LangFlow response structure: {e}")

        verbose_logger.warning(
            "Could not extract content from LangFlow response, returning raw JSON"
        )
        return json.dumps(response_json)

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
        try:
            response_json = raw_response.json()
            verbose_logger.debug(f"LangFlow response: {response_json}")

            content = self._extract_content_from_response(response_json)

            message = Message(content=content, role="assistant")
            choice = Choices(finish_reason="stop", index=0, message=message)

            model_response.choices = [choice]
            model_response.model = model

            try:
                from litellm.utils import token_counter

                prompt_tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
                completion_tokens = token_counter(
                    model="gpt-3.5-turbo", text=content, count_response_tokens=True
                )
                usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                setattr(model_response, "usage", usage)
            except Exception as e:
                verbose_logger.warning(f"Failed to calculate token usage: {e}")

            return model_response

        except Exception as e:
            verbose_logger.error(f"Error processing LangFlow response: {e}")
            raise LangFlowError(
                message=f"Error processing response: {e}",
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
        headers["Content-Type"] = "application/json"

        if api_key:
            headers["x-api-key"] = api_key

        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return LangFlowError(status_code=status_code, message=error_message)

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        return False

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        return stream is True
