"""
Translate from OpenAI's `/v1/chat/completions` to SageMaker Nova Inference endpoints.

Nova models on SageMaker use OpenAI-compatible request/response format with
additional Nova-specific parameters (top_k, reasoning_effort, etc.).

Docs: https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-sagemaker-inference-api-reference.html
"""

from typing import List

from litellm.types.llms.openai import AllMessageValues

from ..chat.transformation import SagemakerChatConfig


class SagemakerNovaConfig(SagemakerChatConfig):
    """
    Config for Amazon Nova models deployed on SageMaker Inference endpoints.

    Nova uses OpenAI-compatible format (same as sagemaker_chat / HF Messages API)
    but with additional Nova-specific parameters and requires `stream: true` in
    the request body for streaming.

    Usage:
        model="sagemaker_nova/<endpoint-name>"
    """

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        """Nova expects `stream: true` in the request body for streaming."""
        return True

    def get_supported_openai_params(self, model: str) -> List:
        """Extend parent params with Nova-specific parameters."""
        params = super().get_supported_openai_params(model)
        nova_params = [
            "top_k",
            "reasoning_effort",
            "allowed_token_ids",
            "truncate_prompt_tokens",
        ]
        for p in nova_params:
            if p not in params:
                params.append(p)
        return params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Nova SageMaker endpoints do not accept 'model' in the request body.
        Only supported fields: messages, max_tokens, max_completion_tokens,
        temperature, top_p, top_k, stream, stream_options, logprobs,
        top_logprobs, reasoning_effort, allowed_token_ids, truncate_prompt_tokens.
        """
        request_body = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        request_body.pop("model", None)
        return request_body


sagemaker_nova_config = SagemakerNovaConfig()
