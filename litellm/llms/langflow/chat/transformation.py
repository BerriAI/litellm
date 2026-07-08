"""LangFlow run API: POST {api_base}/api/v1/run/{flow_id}"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

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

        api_base = api_base or get_secret_str("LANGFLOW_API_BASE") or "http://localhost:7860"
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
        Extract flow_id from the authorized model name only.

        Model format: "langflow/{flow_id}". Request kwargs must not override
        flow_id (would allow calling another flow with the same API key).
        """
        if optional_params.get("flow_id") is not None:
            raise LangFlowError(
                status_code=400,
                message=("flow_id cannot be set via request parameters; use model langflow/{flow_id}"),
            )

        flow_id = (model.split("/", 1)[1] if "/" in model else model).strip()
        if not flow_id:
            raise LangFlowError(
                status_code=400,
                message="flow_id is required; use model langflow/{flow_id}",
            )
        return flow_id

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
        flow_id = quote(self._get_flow_id(model, optional_params), safe="")
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

    def _reject_caller_tweaks(self, params: dict) -> None:
        if params.get("tweaks") is not None:
            raise LangFlowError(
                status_code=400,
                message=(
                    "tweaks cannot be set via request parameters; they would "
                    "override the operator-configured LangFlow flow components"
                ),
            )

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
        self._reject_caller_tweaks(optional_params)

        input_value = self._get_last_user_message(messages)

        payload: Dict[str, Any] = {
            "input_value": input_value,
            "input_type": optional_params.get("input_type", "chat"),
            "output_type": optional_params.get("output_type", "chat"),
        }

        session_id = optional_params.get("session_id")
        if session_id:
            payload["session_id"] = session_id

        verbose_logger.debug(f"LangFlow request payload: {payload}")
        return payload

    def _extract_content_from_response(self, response_json: dict) -> Optional[str]:
        """
        Extract the assistant text from a LangFlow run response.

        Expected structure:
        {"outputs": [{"outputs": [{"results": {"message": {"text": "..."}}}]}]}

        Returns None when no message text is present so the caller can surface an
        explicit error instead of forwarding a raw JSON blob as the answer.
        """
        outputs = response_json.get("outputs", [])
        if not (isinstance(outputs, list) and outputs):
            return None

        first_output = outputs[0]
        if not isinstance(first_output, dict):
            return None

        inner_outputs = first_output.get("outputs", [])
        if not (isinstance(inner_outputs, list) and inner_outputs):
            return None

        first_inner = inner_outputs[0]
        if not isinstance(first_inner, dict):
            return None

        results = first_inner.get("results", {})
        if isinstance(results, dict):
            message = results.get("message", {})
            if isinstance(message, dict) and message.get("text"):
                return message["text"]

        outputs_dict = first_inner.get("outputs", {})
        if isinstance(outputs_dict, dict):
            for val in outputs_dict.values():
                if isinstance(val, dict):
                    msg = val.get("message", {})
                    if isinstance(msg, dict) and msg.get("text"):
                        return msg["text"]

        return None

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
        except Exception as e:
            raise LangFlowError(
                message=f"LangFlow returned a non-JSON response: {e}",
                status_code=raw_response.status_code,
            )

        verbose_logger.debug(f"LangFlow response: {response_json}")

        content = self._extract_content_from_response(response_json)
        if content is None:
            raise LangFlowError(
                message=(
                    "Could not extract a message from the LangFlow response; "
                    "ensure the flow ends in a Chat Output component"
                ),
                status_code=500,
            )

        message = Message(content=content, role="assistant")
        choice = Choices(finish_reason="stop", index=0, message=message)

        model_response.choices = [choice]
        model_response.model = model

        try:
            from litellm.utils import token_counter

            prompt_tokens = token_counter(model=model, messages=messages)
            completion_tokens = token_counter(model=model, text=content, count_response_tokens=True)
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            setattr(model_response, "usage", usage)
        except Exception as e:
            verbose_logger.warning(f"Failed to calculate token usage: {e}")

        return model_response

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        self._reject_caller_tweaks(request_data)
        return headers, None

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
