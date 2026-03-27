from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import httpx

from litellm.constants import AZURE_DEFAULT_CONTAINERS_API_VERSION
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.containers.transformation import OpenAIContainerConfig
from litellm.types.containers.main import ContainerObject
from litellm.types.router import GenericLiteLLMParams


class AzureOpenAIContainerConfig(OpenAIContainerConfig):
    """Azure OpenAI Container Config.

    Inherits from OpenAIContainerConfig and overrides only Azure-specific methods.
    Request/response transformations are identical to OpenAI.
    """

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """Get the complete URL for Azure container API.

        Constructs Azure-specific URLs like:
        https://{resource}.openai.azure.com/openai/v1/containers?api-version=xxx
        """
        return BaseAzureLLM._get_base_azure_url(
            api_base=api_base,
            litellm_params=litellm_params,
            route="/openai/v1/containers",
            default_api_version=AZURE_DEFAULT_CONTAINERS_API_VERSION,
        )

    def validate_environment(
        self,
        headers: dict,
        api_key: Optional[str] = None,
    ) -> dict:
        """Validate and set up Azure authentication headers.

        Uses Azure api-key header (not Bearer token like OpenAI).
        """
        # Create a GenericLiteLLMParams with the api_key if provided
        litellm_params = GenericLiteLLMParams(api_key=api_key) if api_key else None

        # Azure uses BaseAzureLLM's validation which handles api-key header
        return BaseAzureLLM._base_validate_azure_environment(
            headers=headers, litellm_params=litellm_params
        )

    def transform_container_create_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ContainerObject:
        """Transform the Azure container creation response.

        Overrides OpenAI's method to use provider="azure" for cost tracking.
        """
        response_data = raw_response.json()
        container_obj = ContainerObject(**response_data)  # type: ignore[arg-type]

        container_cost = StandardBuiltInToolCostTracking.get_cost_for_code_interpreter(
            sessions=1,
            provider="azure",
        )

        if (
            not hasattr(container_obj, "_hidden_params")
            or container_obj._hidden_params is None
        ):
            container_obj._hidden_params = {}
        if "additional_headers" not in container_obj._hidden_params:
            container_obj._hidden_params["additional_headers"] = {}
        container_obj._hidden_params["additional_headers"][
            "llm_provider-x-litellm-response-cost"
        ] = container_cost

        return container_obj

    def _construct_url_with_subpath(self, api_base: str, subpath: str) -> str:
        """Construct a URL by inserting subpath before query parameters.

        Azure URLs contain ?api-version=... query params. Naively appending
        path segments via string concatenation would place them after the
        query string, producing malformed URLs. This helper parses the URL
        and appends the subpath to the path component only.
        """
        parsed = urlparse(api_base)
        new_path = f"{parsed.path.rstrip('/')}/{subpath}"
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                new_path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

    def transform_container_retrieve_request(
        self,
        container_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        url = self._construct_url_with_subpath(api_base, container_id)
        data: Dict[str, Any] = {}
        return url, data

    def transform_container_delete_request(
        self,
        container_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        url = self._construct_url_with_subpath(api_base, container_id)
        data: Dict[str, Any] = {}
        return url, data

    def transform_container_file_list_request(
        self,
        container_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        url = self._construct_url_with_subpath(api_base, f"{container_id}/files")
        params: Dict[str, Any] = {}
        if after is not None:
            params["after"] = after
        if limit is not None:
            params["limit"] = str(limit)
        if order is not None:
            params["order"] = order
        if extra_query:
            params.update(extra_query)
        return url, params

    def transform_container_file_content_request(
        self,
        container_id: str,
        file_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        url = self._construct_url_with_subpath(
            api_base, f"{container_id}/files/{file_id}/content"
        )
        params: Dict[str, Any] = {}
        return url, params
