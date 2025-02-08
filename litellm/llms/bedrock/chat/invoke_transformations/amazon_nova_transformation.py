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
        bedrock_invoke_nova_request = dict(
            BedrockInvokeNovaRequest(**_transformed_nova_request)
        )
        return bedrock_invoke_nova_request
