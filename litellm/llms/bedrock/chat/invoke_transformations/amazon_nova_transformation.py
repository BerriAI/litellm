"""
Handles transforming requests for `bedrock/invoke/{nova} models`

Inherits from `AmazonConverseConfig`

Nova + Invoke API Tutorial: https://docs.aws.amazon.com/nova/latest/userguide/using-invoke-api.html
"""

from typing import Any, List, Optional

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.llms.bedrock import BedrockInvokeNovaRequest
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ..converse_transformation import AmazonConverseConfig
from .base_invoke_transformation import AmazonInvokeConfig


class AmazonInvokeNovaConfig(AmazonInvokeConfig, AmazonConverseConfig):
    """
    Config for sending `nova` requests to `/bedrock/invoke/`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_supported_openai_params(self, model: str) -> list:
        return AmazonConverseConfig.get_supported_openai_params(self, model)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return AmazonConverseConfig.map_openai_params(
            self, non_default_params, optional_params, model, drop_params
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        _transformed_nova_request = AmazonConverseConfig.transform_request(
            self,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        _bedrock_invoke_nova_request = BedrockInvokeNovaRequest(
            **_transformed_nova_request
        )
        self._remove_empty_system_messages(_bedrock_invoke_nova_request)
        bedrock_invoke_nova_request = self._filter_allowed_fields(
            _bedrock_invoke_nova_request
        )
        return bedrock_invoke_nova_request

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Logging,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        return AmazonConverseConfig.transform_response(
            self,
            model,
            raw_response,
            model_response,
            logging_obj,
            request_data,
            messages,
            optional_params,
            litellm_params,
            encoding,
            api_key,
            json_mode,
        )

    def _filter_allowed_fields(
        self, bedrock_invoke_nova_request: BedrockInvokeNovaRequest
    ) -> dict:
        """
        Filter out fields that are not allowed in the `BedrockInvokeNovaRequest` dataclass.
        """
        allowed_fields = set(BedrockInvokeNovaRequest.__annotations__.keys())
        return {
            k: v for k, v in bedrock_invoke_nova_request.items() if k in allowed_fields
        }

    def _remove_empty_system_messages(
        self, bedrock_invoke_nova_request: BedrockInvokeNovaRequest
    ) -> None:
        """
        In-place remove empty `system` messages from the request.

        /bedrock/invoke/ does not allow empty `system` messages.
        """
        _system_message = bedrock_invoke_nova_request.get("system", None)
        if isinstance(_system_message, list) and len(_system_message) == 0:
            bedrock_invoke_nova_request.pop("system", None)
        return
