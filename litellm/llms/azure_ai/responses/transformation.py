from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import httpx

from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import *
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import _add_path_to_api_base

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AzureAIStudioResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        api_key = (
            (litellm_params.get("api_key") if litellm_params else None)
            or AzureFoundryModelInfo.get_api_key()
        )
        api_base = (
            (litellm_params.get("api_base") if litellm_params else None)
            or AzureFoundryModelInfo.get_api_base()
        )

        if api_base and self._should_use_api_key_header(api_base):
            headers["api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"

        headers["Content-Type"] = "application/json"
        return headers

    def _should_use_api_key_header(self, api_base: str) -> bool:
        from urllib.parse import urlparse

        parsed_url = urlparse(api_base)
        host = parsed_url.hostname
        if host and (
            host.endswith(".services.ai.azure.com")
            or host.endswith(".openai.azure.com")
        ):
            return True
        return False

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        return super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        if api_base is None:
            raise ValueError("api_base is required for Azure AI Studio Responses API")

        original_url = httpx.URL(api_base)
        api_version = litellm_params.get("api_version")

        query_params = dict(original_url.params)
        if "api-version" not in query_params and api_version:
            query_params["api-version"] = api_version

        if "services.ai.azure.com" in api_base:
            new_url = _add_path_to_api_base(
                api_base=api_base, ending_path="/models/responses"
            )
        else:
            new_url = _add_path_to_api_base(
                api_base=api_base, ending_path="/responses"
            )

        final_url = httpx.URL(new_url).copy_with(params=query_params)
        return str(final_url)
