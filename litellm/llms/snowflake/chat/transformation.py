"""
Support for Snowflake REST API 
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai_like.chat.transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class SnowflakeConfig(OpenAIGPTConfig):
    """
    source: https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex
    """

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> List:
        return ["temperature", "max_tokens", "top_p", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        If any supported_openai_params are in non_default_params, add them to optional_params, so they are used in API call

        Args:
            non_default_params (dict): Non-default parameters to filter.
            optional_params (dict): Optional parameters to update.
            model (str): Model name for parameter support check.

        Returns:
            dict: Updated optional_params with supported non-default parameters.
        """
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params

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
        response_json = raw_response.json()
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        returned_response = ModelResponse(**response_json)

        returned_response.model = "snowflake/" + (returned_response.model or "")

        if model is not None:
            returned_response._hidden_params["model"] = model
        return returned_response

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Return headers to use for Snowflake completion request

        Snowflake REST API Ref: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api#api-reference
        Expected headers:
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + <JWT>,
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT"
        }
        """

        if api_key is None:
            raise ValueError("Missing Snowflake JWT key")

        headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer " + api_key,
                "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
            }
        )
        return headers

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or f"""https://{get_secret_str("SNOWFLAKE_ACCOUNT_ID")}.snowflakecomputing.com/api/v2/cortex/inference:complete"""
            or get_secret_str("SNOWFLAKE_API_BASE")
        )
        dynamic_api_key = api_key or get_secret_str("SNOWFLAKE_JWT")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        If api_base is not provided, use the default DeepSeek /chat/completions endpoint.
        """
        if not api_base:
            api_base = f"""https://{get_secret_str("SNOWFLAKE_ACCOUNT_ID")}.snowflakecomputing.com/api/v2/cortex/inference:complete"""

        return api_base

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        stream: bool = optional_params.pop("stream", None) or False
        extra_body = optional_params.pop("extra_body", {})
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            **optional_params,
            **extra_body,
        }
