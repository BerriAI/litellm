import re
from typing import Final

import httpx

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig

_FOUNDRY_ROUTE_SUFFIX: Final = re.compile(r"(/api/projects/[^/]+)?(/openai/v1|/openai|/models|/v1)?$")


class AzureAIDecisionsConfig(BaseDecisionsConfig):
    """Microsoft Foundry decision models (Microsoft-Decision-1), where model is the deployment name"""

    path = "/providers/microsoft/v1/systemone"
    api_key_env = ("AZURE_AI_API_KEY",)
    api_base_env = ("AZURE_AI_API_BASE",)

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return "Missing AZURE_AI_API_BASE - set AZURE_AI_API_BASE or pass api_base, e.g. https://<resource>.services.ai.azure.com"

    def get_complete_url(self, api_base: str, model: str) -> str:
        url: Final = httpx.URL(api_base)
        resource_path: Final = _FOUNDRY_ROUTE_SUFFIX.sub("", url.path.rstrip("/"))
        return str(url.copy_with(path=f"{resource_path}{self.path}", query=None, fragment=None))
