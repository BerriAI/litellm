"""
TARS (Tetrate Agent Router Service) common utilities and model info.
"""

from typing import List, Optional

import httpx

from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str


class TarsException(BaseLLMException):
    """Exception class for TARS provider errors."""
    pass


class TarsModelInfo(BaseLLMModelInfo):
    """
    Model info for TARS (Tetrate Agent Router Service) provider.
    
    Supports dynamic model fetching from the TARS API.
    """

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """Get TARS API key from parameter or environment variable."""
        return api_key or get_secret_str("TARS_API_KEY")

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        """Get TARS API base URL from parameter or environment variable."""
        return api_base or get_secret_str("TARS_API_BASE") or "https://api.router.tetrate.ai/v1"

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        """Remove tars/ prefix from model name."""
        return model.replace("tars/", "")

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        """
        Fetch available models from TARS API.
        
        Args:
            api_key: TARS API key (optional, will use TARS_API_KEY env var if not provided)
            api_base: TARS API base URL (optional, defaults to https://api.router.tetrate.ai/v1)
            
        Returns:
            List of model names prefixed with "tars/"
        """
        api_base = self.get_api_base(api_base)
        api_key = self.get_api_key(api_key)
        
        if api_key is None:
            raise ValueError(
                "TARS_API_KEY is not set. Please set the environment variable to query TARS's /models endpoint."
            )

        try:
            # Use a fresh httpx client to avoid any global configuration issues
            url = f"{api_base}/models"
            with httpx.Client() as client:
                response = client.get(
                    url=url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10.0
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Failed to fetch models from TARS. Status code: {e.response.status_code}, Response: {e.response.text}"
            )
        except Exception as e:
            raise ValueError(f"Failed to fetch models from TARS. Error: {e}")

        models_data = response.json().get("data", [])
        
        # Extract model IDs and prefix with "tars/"
        litellm_model_names = []
        for model in models_data:
            if isinstance(model, dict) and "id" in model:
                model_id = model["id"]
                litellm_model_name = f"tars/{model_id}"
                litellm_model_names.append(litellm_model_name)
        
        return sorted(litellm_model_names)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Validate TARS environment and add authentication headers."""
        api_key = self.get_api_key(api_key)
        api_base = self.get_api_base(api_base)
        
        if api_key is None:
            raise ValueError(
                "TARS_API_KEY is not set. Please set the environment variable."
            )
        
        headers["Authorization"] = f"Bearer {api_key}"
        return headers