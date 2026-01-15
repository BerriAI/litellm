from typing import List, Literal, Optional

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


class AzureFoundryModelInfo(BaseLLMModelInfo):
    @staticmethod
    def get_azure_ai_route(model: str) -> Literal["agents", "default"]:
        """
        Get the Azure AI route for the given model.
        
        Similar to BedrockModelInfo.get_bedrock_route().
        """
        if "agents/" in model:
            return "agents"
        return "default"

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return (
                api_base
                or litellm.api_base
                or get_secret_str("AZURE_AI_API_BASE")
        )
    
    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return (
                api_key
                or litellm.api_key
                or litellm.openai_key
                or get_secret_str("AZURE_AI_API_KEY")
            )
    
    @property
    def api_version(self, api_version: Optional[str] = None) -> Optional[str]:
        api_version = (
            api_version
            or litellm.api_version
            or get_secret_str("AZURE_API_VERSION")
        )
        return api_version
    
    #########################################################
    # Not implemented methods
    #########################################################
        

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        raise NotImplementedError("Azure Foundry does not support base model")

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
        """Azure Foundry sends api key in query params"""
        raise NotImplementedError("Azure Foundry does not support environment validation")
