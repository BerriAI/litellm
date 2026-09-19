from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.vector_stores.transformation import OpenAIVectorStoreConfig
from litellm.types.router import GenericLiteLLMParams


class AzureOpenAIVectorStoreConfig(OpenAIVectorStoreConfig):
    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        return BaseAzureLLM._get_base_azure_url(
            api_base=api_base,
            litellm_params=litellm_params,
            route="/openai/vector_stores",
        )

    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        return BaseAzureLLM._base_validate_azure_environment(headers=headers, litellm_params=litellm_params)
