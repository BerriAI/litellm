"""
SageMaker Voyage AI Embedding Configuration

This module provides support for Voyage AI embedding models deployed on AWS SageMaker.
It inherits from the regular Voyage AI configuration but uses AWS authentication and SageMaker endpoints.

Usage:
    response = embedding(
        model="sagemaker_voyage/voyage-3",
        input=["Sample text 1", "Sample text 2"],
        input_type="query",
        aws_region_name="us-east-1"
    )
"""
from typing import Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage

from .transformation import VoyageEmbeddingConfig, VoyageError


class SageMakerVoyageError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Union[dict, httpx.Headers] = {},
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://runtime.sagemaker.{region}.amazonaws.com/"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class SageMakerVoyageEmbeddingConfig(VoyageEmbeddingConfig, BaseAWSLLM):
    """
    SageMaker Voyage AI embedding configuration that combines Voyage AI transformation logic
    with AWS authentication and SageMaker endpoint management.
    
    Reference: 
    - Voyage AI API: https://docs.voyageai.com/reference/embeddings-api
    - SageMaker Runtime: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html
    """

    # Map Voyage AI model names to their SageMaker model package resource IDs
    VOYAGE_SAGEMAKER_MODEL_MAPPING = {
        "voyage-2": "voyage-2-61d99cdc29f7359e9f70b9a6098826ae",
        "voyage-large-2": "voyage-large-2-0ff46a42571f34ffbc8182db4cfca13d",
        "voyage-large-2-instruct": "voyage-large-2-instruct-7557d7ff5d183919a4e767c35eb8b0f3",
        "voyage-code-2": "voyage-code-2-b220a35876c038039869f51e600d44b4",
        "voyage-law-2": "voyage-law-2-9eed475b45ec3034bee6cb155fdd853d",
        "voyage-finance-2": "voyage-finance-2-1adbefe1db413d249a1e44270d748140",
        "voyage-multilingual-2": "voyage-multilingual-2-9d8b876c9ef63b0e811b51f60a450e57",
        "voyage-code-3": "voyage-code-3-v1-634308625a5e3ecc8e86cbddc167c902",
        "voyage-multimodal-3": "voyage-multimodal-3-v1-0-3b30a5c60eea3f6e9d91efd1fde1df07",
        "voyage-3-large": "voyage-3-large-v1-906b498ebfaf36af87de6a73f60ecad8",
        "voyage-3": "voyage-3-008e58ecc01b306b82d088dcb115a8a2",
        "voyage-3-lite": "voyage-3-lite-4f6d6fd00bab304e89341742ec1728d4",
        "rerank-lite-1": "rerank-lite-1-3a1cfa7773203a7d8c50b27fd9ecadc1",
        "rerank-2": "rerank-2-v1-1a846a9be136312ab9d5543c3d6d3564",
        "rerank-2-lite": "rerank-2-lite-v1-a953a26b0e523237bcf7627ad4cddd8b",
    }

    def __init__(self) -> None:
        VoyageEmbeddingConfig.__init__(self)
        BaseAWSLLM.__init__(self)

    def _extract_model_name(self, model: str) -> str:
        """
        Extract the actual Voyage model name from the full model path.
        
        Args:
            model: Full model name like "sagemaker/voyage/voyage-3" or just "voyage-3"
            
        Returns:
            str: The base Voyage model name (e.g., "voyage-3")
        """
        if "/" in model:
            # Handle both new format "sagemaker/voyage/voyage-3" and old format "sagemaker_voyage/voyage-3"
            parts = model.split("/")
            if len(parts) >= 3 and parts[0] == "sagemaker" and parts[1] == "voyage":
                # New format: sagemaker/voyage/voyage-3
                return parts[2]
            else:
                # Old format or other: take the last part
                return parts[-1]
        return model

    def _get_endpoint_name(self, model: str, optional_params: dict) -> str:
        """
        Get the SageMaker endpoint name for the Voyage model.
        
        Args:
            model: The model name
            optional_params: Optional parameters that may contain custom endpoint name
            
        Returns:
            str: The SageMaker endpoint name
        """
        # Check if custom endpoint name is provided
        if "sagemaker_endpoint_name" in optional_params:
            return optional_params["sagemaker_endpoint_name"]
        
        # Extract base model name and use it as endpoint name
        base_model = self._extract_model_name(model)
        return base_model

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
        Construct the SageMaker runtime endpoint URL for Voyage AI models.
        
        Args:
            api_base: Not used for SageMaker
            api_key: Not used for SageMaker (uses AWS credentials)
            model: The model name
            optional_params: Optional parameters including AWS region
            litellm_params: LiteLLM parameters
            stream: Not used for embeddings
            
        Returns:
            str: Complete SageMaker runtime endpoint URL
        """
        # Get AWS region
        aws_region_name = self._get_aws_region_name(optional_params, model)
        
        # Get endpoint name
        endpoint_name = self._get_endpoint_name(model, optional_params)
        
        # Construct SageMaker runtime URL
        return f"https://runtime.sagemaker.{aws_region_name}.amazonaws.com/endpoints/{endpoint_name}/invocations"

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
        Validate environment and return AWS authentication headers instead of API key headers.
        
        Args:
            headers: Existing headers
            model: Model name
            messages: Input messages (not used for embeddings)
            optional_params: Optional parameters containing AWS credentials
            litellm_params: LiteLLM parameters
            api_key: Not used for SageMaker
            api_base: Not used for SageMaker
            
        Returns:
            dict: AWS authentication headers
        """
        # Get AWS credentials
        credentials = self.get_credentials(
            aws_access_key_id=optional_params.get("aws_access_key_id"),
            aws_secret_access_key=optional_params.get("aws_secret_access_key"),
            aws_session_token=optional_params.get("aws_session_token"),
            aws_region_name=optional_params.get("aws_region_name"),
            aws_session_name=optional_params.get("aws_session_name"),
            aws_profile_name=optional_params.get("aws_profile_name"),
            aws_role_name=optional_params.get("aws_role_name"),
            aws_web_identity_token=optional_params.get("aws_web_identity_token"),
            aws_sts_endpoint=optional_params.get("aws_sts_endpoint"),
            aws_external_id=optional_params.get("aws_external_id"),
        )

        # Store credentials in optional_params for use in HTTP handler
        # The actual AWS request signing will be handled by boto3/httpx in the HTTP handler
        optional_params["aws_credentials"] = credentials
        optional_params["aws_region_name"] = self._get_aws_region_name(optional_params, model)

        # Return standard headers - AWS signing will be handled by the HTTP client
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the embedding request to SageMaker format while preserving Voyage AI parameters.
        
        The SageMaker Voyage models expect the same format as the Voyage API, so we use
        the parent class transformation and ensure it's compatible with SageMaker deployment.
        
        Args:
            model: Model name
            input: Embedding input texts
            optional_params: Optional parameters including Voyage-specific params
            headers: Request headers
            
        Returns:
            dict: Request payload formatted for SageMaker Voyage deployment
        """
        # Use the parent Voyage transformation logic
        base_request = super().transform_embedding_request(
            model=self._extract_model_name(model),
            input=input,
            optional_params=optional_params,
            headers=headers,
        )
        
        # SageMaker Voyage deployments expect the same format as the Voyage API
        # The main difference is the endpoint URL and authentication method
        return base_request

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> EmbeddingResponse:
        """
        Transform the SageMaker response to the standard embedding response format.
        
        SageMaker Voyage deployments return the same format as the Voyage API,
        so we can use the parent class transformation logic.
        
        Args:
            model: Model name
            raw_response: Raw HTTP response from SageMaker
            model_response: Response object to populate
            logging_obj: Logging object
            api_key: Not used for SageMaker
            request_data: Original request data
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            
        Returns:
            EmbeddingResponse: Populated response object
        """
        try:
            # Use parent class transformation logic since SageMaker returns same format
            return super().transform_embedding_response(
                model=model,
                raw_response=raw_response,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key=api_key,
                request_data=request_data,
                optional_params=optional_params,
                litellm_params=litellm_params,
            )
        except VoyageError as e:
            # Convert VoyageError to SageMakerVoyageError
            raise SageMakerVoyageError(
                status_code=e.status_code,
                message=e.message,
                headers=e.response.headers if hasattr(e, 'response') else {},
            )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """
        Return the appropriate error class for SageMaker Voyage errors.
        
        Args:
            error_message: Error message
            status_code: HTTP status code
            headers: Response headers
            
        Returns:
            BaseLLMException: SageMaker-specific error instance
        """
        return SageMakerVoyageError(
            message=error_message, status_code=status_code, headers=headers
        )