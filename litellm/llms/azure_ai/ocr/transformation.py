"""
Azure AI OCR transformation implementation.
"""
from typing import Dict, Optional

from litellm._logging import verbose_logger
from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.secret_managers.main import get_secret_str


class AzureAIOCRConfig(MistralOCRConfig):
    """
    Azure AI OCR transformation configuration.
    
    Azure AI uses Mistral's OCR API but with a different endpoint format.
    Inherits transformation logic from MistralOCRConfig since they use the same format.
    
    Reference: Azure AI Foundry OCR documentation
    
    Important: Azure AI only supports base64 data URIs (data:image/..., data:application/pdf;base64,...).
    Regular URLs are not supported.
    """

    def __init__(self) -> None:
        super().__init__()

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Validate environment and return headers for Azure AI OCR.
        
        Azure AI uses Bearer token authentication with AZURE_AI_API_KEY.
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = get_secret_str("AZURE_AI_API_KEY")

        if api_key is None:
            raise ValueError(
                "Missing Azure AI API Key - A call is being made to Azure AI but no key is set either in the environment variables or via params"
            )

        # Validate API base is provided
        if api_base is None:
            api_base = get_secret_str("AZURE_AI_API_BASE")
        
        if api_base is None:
            raise ValueError(
                "Missing Azure AI API Base - Set AZURE_AI_API_BASE environment variable or pass api_base parameter"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **headers,
        }

        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        **kwargs,
    ) -> str:
        """
        Get complete URL for Azure AI OCR endpoint.
        
        Azure AI endpoint format: https://<api_base>/providers/mistral/azure/ocr
        
        Args:
            api_base: Azure AI API base URL
            model: Model name (not used in URL construction)
            optional_params: Optional parameters
            
        Returns: Complete URL for Azure AI OCR endpoint
        """
        if api_base is None:
            raise ValueError(
                "Missing Azure AI API Base - Set AZURE_AI_API_BASE environment variable or pass api_base parameter"
            )

        # Ensure no trailing slash
        api_base = api_base.rstrip("/")
        
        # Azure AI OCR endpoint format
        return f"{api_base}/providers/mistral/azure/ocr"

