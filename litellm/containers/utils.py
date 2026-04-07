from typing import Any, Dict, Optional, TypeVar

from litellm.llms.base_llm.containers.transformation import BaseContainerConfig
from litellm.types.containers.main import (
    ContainerCreateOptionalRequestParams,
    ContainerListOptionalRequestParams,
)

T = TypeVar("T")


class ContainerRequestUtils:
    @staticmethod
    def get_requested_container_create_optional_param(
        passed_params: dict,
    ) -> ContainerCreateOptionalRequestParams:
        """Extract only valid container creation parameters from the passed parameters."""
        container_create_optional_params = ContainerCreateOptionalRequestParams()

        valid_params = [
            "expires_after",
            "file_ids",
            "extra_headers",
            "extra_body",
        ]

        for param in valid_params:
            if param in passed_params and passed_params[param] is not None:
                container_create_optional_params[param] = passed_params[param]  # type: ignore

        return container_create_optional_params

    @staticmethod
    def get_optional_params_container_create(
        container_provider_config: BaseContainerConfig,
        container_create_optional_params: ContainerCreateOptionalRequestParams,
    ) -> Dict:
        """Get the optional parameters for container creation."""
        supported_params = container_provider_config.get_supported_openai_params()

        # Filter out unsupported parameters
        filtered_params = {
            k: v
            for k, v in container_create_optional_params.items()
            if k in supported_params
        }

        return container_provider_config.map_openai_params(
            container_create_optional_params=filtered_params,  # type: ignore
            drop_params=False,
        )

    @staticmethod
    def get_requested_container_list_optional_param(
        passed_params: dict,
    ) -> ContainerListOptionalRequestParams:
        """Extract only valid container list parameters from the passed parameters."""
        container_list_optional_params = ContainerListOptionalRequestParams()

        valid_params = [
            "after",
            "limit",
            "order",
            "extra_headers",
            "extra_query",
        ]

        for param in valid_params:
            if param in passed_params and passed_params[param] is not None:
                container_list_optional_params[param] = passed_params[param]  # type: ignore

        return container_list_optional_params

    @staticmethod
    def encode_container_id_in_response(
        response_obj: T,
        custom_llm_provider: Optional[str],
        model_id: Optional[str],
    ) -> T:
        """
        Encode container_id in response object with provider/model metadata for routing.
        
        This mirrors the responses API pattern where response IDs are encoded with
        routing metadata so follow-up calls can route to the correct provider.
        
        Args:
            response_obj: Response object with an `id` attribute (ContainerObject, DeleteContainerResult, etc.)
            custom_llm_provider: Provider name (e.g., "azure", "openai")
            model_id: Model ID from litellm_metadata
            
        Returns:
            The same response object with encoded container_id
        """
        if response_obj and hasattr(response_obj, "id"):
            from litellm.responses.utils import ResponsesAPIRequestUtils
            
            encoded_id = ResponsesAPIRequestUtils._build_container_id(
                custom_llm_provider=custom_llm_provider,
                model_id=model_id,
                container_id=response_obj.id,
            )
            response_obj.id = encoded_id
        
        return response_obj
