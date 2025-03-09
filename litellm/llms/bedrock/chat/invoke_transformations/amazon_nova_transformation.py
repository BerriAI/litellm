"""
Handles transforming requests for `bedrock/invoke/{nova} models`

Inherits from `AmazonConverseConfig`

Nova + Invoke API Tutorial: https://docs.aws.amazon.com/nova/latest/userguide/using-invoke-api.html
"""

from typing import List

import litellm
from litellm.types.llms.bedrock import BedrockInvokeNovaRequest
from litellm.types.llms.openai import AllMessageValues


class AmazonInvokeNovaConfig(litellm.AmazonConverseConfig):
    """
    Config for sending `nova` requests to `/bedrock/invoke/`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        _transformed_nova_request = super().transform_request(
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
