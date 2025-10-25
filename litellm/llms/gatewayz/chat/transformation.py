import json
import os
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, List, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage

from ..common_utils import ModelResponseIterator as GatewayzModelResponseIterator
from ..common_utils import validate_environment as gatewayz_validate_environment

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class GatewayzError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[httpx.Headers] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://api.gatewayz.com/v1/chat/completions"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class GatewayzChatConfig(BaseConfig):
    """
    Configuration class for Gatewayz's API interface.

    Gatewayz uses an OpenAI-compatible API format for chat completions.

    Args:
        temperature (float, optional): Controls randomness in generation (0.0-2.0)
        max_tokens (int, optional): Maximum tokens to generate
        top_p (float, optional): Nucleus sampling parameter
        frequency_penalty (float, optional): Reduces repetition of tokens
        presence_penalty (float, optional): Reduces repetition of topics
        stop (list, optional): Sequences where generation should stop
        stream (bool, optional): Enable streaming responses
        n (int, optional): Number of completions to generate
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list] = None
    stream: Optional[bool] = None
    n: Optional[int] = None

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[list] = None,
        stream: Optional[bool] = None,
        n: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "stream":
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            if param == "presence_penalty":
                optional_params["presence_penalty"] = value
            if param == "stop":
                optional_params["stop"] = value
            if param == "n":
                optional_params["n"] = value
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
        # Get API key from environment if not provided
        if api_key is None:
            api_key = (
                get_secret_str("GATEWAYZ_API_KEY")
                or os.getenv("GATEWAYZ_API_KEY")
                or litellm.gatewayz_key
            )

        if api_key is None:
            raise ValueError(
                "GATEWAYZ_API_KEY not found. Please set it in your environment or pass it as api_key parameter."
            )

        # Get API base URL
        if api_base is None:
            api_base = (
                os.getenv("GATEWAYZ_API_BASE")
                or litellm.gatewayz_api_base
                or "https://api.gatewayz.com"
            )

        # Store for later use
        self.api_key = api_key
        self.api_base = api_base

        return gatewayz_validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # Strip the "gatewayz/" prefix from model name
        model_name = model
        if model.startswith("gatewayz/"):
            model_name = model.split("/", 1)[1]

        # Build request payload in OpenAI format
        request_data = {
            "model": model_name,
            "messages": messages,
        }

        # Add optional parameters
        if "temperature" in optional_params:
            request_data["temperature"] = optional_params["temperature"]
        if "max_tokens" in optional_params:
            request_data["max_tokens"] = optional_params["max_tokens"]
        if "top_p" in optional_params:
            request_data["top_p"] = optional_params["top_p"]
        if "frequency_penalty" in optional_params:
            request_data["frequency_penalty"] = optional_params["frequency_penalty"]
        if "presence_penalty" in optional_params:
            request_data["presence_penalty"] = optional_params["presence_penalty"]
        if "stop" in optional_params:
            request_data["stop"] = optional_params["stop"]
        if "stream" in optional_params:
            request_data["stream"] = optional_params["stream"]
        if "n" in optional_params:
            request_data["n"] = optional_params["n"]

        return request_data

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
            raw_response_json = raw_response.json()
        except Exception:
            raise GatewayzError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        # Extract response fields
        choices = raw_response_json.get("choices", [])
        if len(choices) == 0:
            raise GatewayzError(
                message="No choices returned in response",
                status_code=raw_response.status_code,
            )

        # Get the first choice
        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

        # Set message content and finish reason
        model_response.choices[0].message.content = content  # type: ignore
        model_response.choices[0].finish_reason = finish_reason

        # Extract usage information
        usage_data = raw_response_json.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)

        # Set model metadata
        model_response.created = raw_response_json.get("created", int(time.time()))
        model_response.model = model
        model_response.id = raw_response_json.get("id", f"chatcmpl-{int(time.time())}")

        # Set usage
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        setattr(model_response, "usage", usage)

        return model_response

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for the chat completions endpoint
        """
        if api_base is None:
            api_base = (
                os.getenv("GATEWAYZ_API_BASE")
                or litellm.gatewayz_api_base
                or "https://api.gatewayz.com"
            )

        # Ensure api_base doesn't end with /
        api_base = api_base.rstrip("/")

        # Construct the full URL
        return f"{api_base}/v1/chat/completions"

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        return GatewayzModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return GatewayzError(status_code=status_code, message=error_message)
