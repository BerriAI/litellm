import asyncio
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import ValidationError

import litellm
from litellm.llms.azure_ai.common_utils import (
    AZURE_ENTRA_LITELLM_PARAM_KEYS,
    get_azure_ai_auth_headers,
)
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig

_FOUNDRY_ROUTE_SUFFIX: Final = re.compile(r"(/api/projects/[^/]+)?(/openai/v1|/openai|/models|/v1)?$")
_AZURE_AI_CREDENTIAL_PARAM_KEYS: Final = AZURE_ENTRA_LITELLM_PARAM_KEYS | frozenset({"azure_ad_token_provider"})


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
        credential_params: Final = MappingProxyType(
            {key: litellm_params[key] for key in _AZURE_AI_CREDENTIAL_PARAM_KEYS if key in litellm_params}
        )
        try:
            auth_headers: Final = get_azure_ai_auth_headers(api_key, credential_params)
        except ValidationError as error:
            field_names: Final = tuple(
                dict.fromkeys(
                    ".".join(str(field) for field in detail["loc"])
                    for detail in error.errors(include_input=False, include_context=False, include_url=False)
                )
            )
            raise litellm.BadRequestError(
                message=f"Invalid Azure AI credential parameter(s): {', '.join(f'`{name}`' for name in field_names)}",
                model=model,
                llm_provider="azure_ai",
            ) from error
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
