import asyncio
import re
from collections.abc import Mapping
from typing import Final

import httpx

import litellm
from litellm.llms.azure_ai.common_utils import get_azure_ai_auth_headers
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig

_FOUNDRY_ROUTE_SUFFIX: Final = re.compile(r"(/api/projects/[^/]+)?(/openai/v1|/openai|/models|/v1)?$")


class AzureAIDecisionsConfig(BaseDecisionsConfig):
    """Microsoft Foundry decision models (Microsoft-Decision-1), where model is the deployment name"""

    path = "/providers/microsoft/v1/systemone"
    api_key_env = ("AZURE_AI_API_KEY",)
    api_base_env = ("AZURE_AI_API_BASE",)
    api_key_required = False

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return "Missing AZURE_AI_API_BASE - set AZURE_AI_API_BASE or pass api_base, e.g. https://<resource>.services.ai.azure.com"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, str]:
        try:
            auth_headers: Final = get_azure_ai_auth_headers(api_key, litellm_params)
        except ValueError as error:
            raise litellm.AuthenticationError(message=str(error), model=model, llm_provider="azure_ai") from error
        return {
            **super().validate_environment(
                headers=headers,
                model=model,
                api_key=api_key,
                litellm_params=litellm_params,
            ),
            **auth_headers,
        }

    async def avalidate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, str]:
        if api_key:
            return self.validate_environment(
                headers=headers,
                model=model,
                api_key=api_key,
                litellm_params=litellm_params,
            )
        return await asyncio.to_thread(
            self.validate_environment,
            headers=headers,
            model=model,
            api_key=api_key,
            litellm_params=litellm_params,
        )

    def get_complete_url(self, api_base: str, model: str) -> str:
        url: Final = httpx.URL(api_base)
        resource_path: Final = _FOUNDRY_ROUTE_SUFFIX.sub("", url.path.rstrip("/"))
        return str(url.copy_with(path=f"{resource_path}{self.path}", query=None, fragment=None))
