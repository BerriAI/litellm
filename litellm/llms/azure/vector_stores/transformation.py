from typing import Optional

from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.vector_stores.transformation import OpenAIVectorStoreConfig


class AzureOpenAIVectorStoreConfig(OpenAIVectorStoreConfig):
    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        return BaseAzureLLM.get_complete_url(
            api_base=api_base,
            litellm_params=litellm_params,
            route="/openai/vector_stores"
        )

