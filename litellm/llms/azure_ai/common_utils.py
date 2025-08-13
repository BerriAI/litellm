from typing import List, Optional, Union

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


class AzureFoundryModelInfo(BaseLLMModelInfo):
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
    
    #########################################################
    # Not implemented methods
    #########################################################
    @property
    def api_version(self) -> str:
        raise NotImplementedError("Azure Foundry does not support api version")

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
