from typing import Final

from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.vector_stores.transformation import OpenAIVectorStoreConfig
from litellm.types.router import GenericLiteLLMParams


class AzureOpenAIVectorStoreConfig(OpenAIVectorStoreConfig):
    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        return BaseAzureLLM.get_base_azure_url(
            api_base=api_base,
            litellm_params=litellm_params,
            route="/openai/vector_stores",
        )

    def get_vector_store_url(self, api_base: str, encoded_vector_store_id: str) -> str:
        path, _, query = api_base.partition("?")
        url: Final = f"{path}/{encoded_vector_store_id}"
        return f"{url}?{query}" if query else url

    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        return BaseAzureLLM.base_validate_azure_environment(headers=headers, litellm_params=litellm_params)
