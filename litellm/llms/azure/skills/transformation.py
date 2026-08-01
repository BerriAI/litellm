"""Azure OpenAI native Skills API configuration."""

from litellm.constants import AZURE_DEFAULT_RESPONSES_API_VERSION
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.skills.transformation import OpenAISkillsConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class AzureOpenAISkillsConfig(OpenAISkillsConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.AZURE

    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        return BaseAzureLLM._base_validate_azure_environment(headers=headers, litellm_params=litellm_params)

    def get_api_base(self, litellm_params: GenericLiteLLMParams) -> str | None:
        return litellm_params.api_base

    def get_complete_url(
        self,
        api_base: str | None,
        endpoint: str,
        skill_id: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> str:
        path = endpoint.strip("/")
        if skill_id is not None:
            from litellm.litellm_core_utils.url_utils import encode_url_path_segment

            path = f"{path}/{encode_url_path_segment(skill_id, field_name='skill_id')}"
        return BaseAzureLLM._get_base_azure_url(
            api_base=api_base or (litellm_params.api_base if litellm_params is not None else None),
            litellm_params=litellm_params,
            route=f"/openai/{path}",
            default_api_version=AZURE_DEFAULT_RESPONSES_API_VERSION,
        )
