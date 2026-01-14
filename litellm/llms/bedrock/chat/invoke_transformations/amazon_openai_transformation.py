"""
Transformation for Bedrock imported models that use OpenAI Chat Completions format.

Use this for models imported into Bedrock that accept the OpenAI API format.
Model format: bedrock/openai/<model-id>

Example: bedrock/openai/arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import httpx

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonBedrockOpenAIConfig(OpenAIGPTConfig, BaseAWSLLM):
    """
    Configuration for Bedrock imported models that use OpenAI Chat Completions format.
    
    This class handles the transformation of requests and responses for Bedrock
    imported models that accept the OpenAI API format directly.
    
    Inherits from OpenAIGPTConfig to leverage standard OpenAI parameter handling
    and response transformation, while adding Bedrock-specific URL generation
    and AWS request signing.
    
    Usage:
        model = "bedrock/openai/arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123"
    """

    def __init__(self, **kwargs):
        OpenAIGPTConfig.__init__(self, **kwargs)
        BaseAWSLLM.__init__(self, **kwargs)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "bedrock"

    def _get_openai_model_id(self, model: str) -> str:
        """
        Extract the actual model ID from the LiteLLM model name.
        
        Input format: bedrock/openai/<model-id>
        Returns: <model-id>
        """
        # Remove bedrock/ prefix if present
        if model.startswith("bedrock/"):
            model = model[8:]
        
        # Remove openai/ prefix
        if model.startswith("openai/"):
            model = model[7:]
        
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
        """
        Get the complete URL for the Bedrock invoke endpoint.
        
        Uses the standard Bedrock invoke endpoint format.
        """
        model_id = self._get_openai_model_id(model)
        
        # Get AWS region
        aws_region_name = self._get_aws_region_name(
            optional_params=optional_params, model=model
        )
        
        # Get runtime endpoint
        aws_bedrock_runtime_endpoint = optional_params.get(
            "aws_bedrock_runtime_endpoint", None
        )
        endpoint_url, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_region_name=aws_region_name,
        )
        
        # Build the invoke URL
        if stream:
            endpoint_url = f"{endpoint_url}/model/{model_id}/invoke-with-response-stream"
        else:
            endpoint_url = f"{endpoint_url}/model/{model_id}/invoke"
        
        return endpoint_url

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
        """
        Sign the request using AWS Signature Version 4.
        """
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
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
        Transform the request to OpenAI Chat Completions format for Bedrock imported models.
        
        Removes AWS-specific params and stream param (handled separately in URL),
        then delegates to parent class for standard OpenAI request transformation.
        """
        # Remove stream from optional_params as it's handled separately in URL
        optional_params.pop("stream", None)
        
        # Remove AWS-specific params that shouldn't be in the request body
        inference_params = {
            k: v
            for k, v in optional_params.items()
            if k not in self.aws_authentication_params
        }
        
        # Use parent class transform_request for OpenAI format
        return super().transform_request(
            model=self._get_openai_model_id(model),
            messages=messages,
            optional_params=inference_params,
            litellm_params=litellm_params,
            headers=headers,
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
        """
        Validate the environment and return headers.
        
        For Bedrock, we don't need Bearer token auth since we use AWS SigV4.
        """
        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BedrockError:
        """Return the appropriate error class for Bedrock."""
        return BedrockError(status_code=status_code, message=error_message)
