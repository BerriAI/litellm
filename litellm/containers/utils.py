from typing import Any, Final, TypeVar

from litellm.llms.base_llm.containers.transformation import BaseContainerConfig
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.containers.main import (
    ContainerCreateOptionalRequestParams,
    ContainerListOptionalRequestParams,
)
from litellm.types.router import GenericLiteLLMParams


def decode_managed_container_id_for_request(
    container_id: str,
    custom_llm_provider: str,
    litellm_params: GenericLiteLLMParams,
) -> tuple[str, str, GenericLiteLLMParams]:
    """Decode a LiteLLM-managed container ID for upstream API calls.

    Returns:
        (original_container_id, resolved_provider, updated_litellm_params)
    """
    decoded: Final = ResponsesAPIRequestUtils._decode_container_id(container_id)
    original_container_id: Final = decoded.get("response_id", container_id)

    decoded_provider: Final = decoded.get("custom_llm_provider")
    if decoded_provider and custom_llm_provider == "openai":
        custom_llm_provider = decoded_provider

    decoded_model_id: Final = decoded.get("model_id")
    if decoded_model_id and not litellm_params.get("model_id"):
        litellm_params["model_id"] = decoded_model_id

    return original_container_id, custom_llm_provider, litellm_params


T = TypeVar("T")


class ContainerRequestUtils:
    @staticmethod
    def get_requested_container_create_optional_param(
        passed_params: dict,
    ) -> ContainerCreateOptionalRequestParams:
        """Extract only valid container creation parameters from the passed parameters."""
        container_create_optional_params: Final = ContainerCreateOptionalRequestParams()

        valid_params: Final = [
            "expires_after",
            "file_ids",
            "extra_headers",
            "extra_body",
        ]

        for param in valid_params:
            if param in passed_params and passed_params[param] is not None:
                container_create_optional_params[param] = passed_params[param]

        return container_create_optional_params

    @staticmethod
    def get_optional_params_container_create(
        container_provider_config: BaseContainerConfig,
        container_create_optional_params: ContainerCreateOptionalRequestParams,
    ) -> dict:
        """Get the optional parameters for container creation."""
        supported_params: Final = container_provider_config.get_supported_openai_params()

        # Filter out unsupported parameters
        filtered_params: Final = {k: v for k, v in container_create_optional_params.items() if k in supported_params}

        return container_provider_config.map_openai_params(
            container_create_optional_params=filtered_params,
            drop_params=False,
        )

    @staticmethod
    def get_requested_container_list_optional_param(
        passed_params: dict,
    ) -> ContainerListOptionalRequestParams:
        """Extract only valid container list parameters from the passed parameters."""
        container_list_optional_params: Final = ContainerListOptionalRequestParams()

        valid_params: Final = [
            "after",
            "limit",
            "order",
            "extra_headers",
            "extra_query",
        ]

        for param in valid_params:
            if param in passed_params and passed_params[param] is not None:
                container_list_optional_params[param] = passed_params[param]

        return container_list_optional_params

    @staticmethod
    def encode_container_id_in_response(
        response_obj: T,
        custom_llm_provider: str | None,
        litellm_metadata: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> T:
        """
        Encode container_id in response object with provider/model metadata for routing.

        This mirrors the responses API pattern where response IDs are encoded with
        routing metadata so follow-up calls can route to the correct provider.

        Encodes when:
        1. litellm_metadata contains model_info.id (indicating router/proxy usage), OR
        2. extra_body contains target_model_names (indicating model-specific routing)

        Direct SDK calls with explicit custom_llm_provider and no routing hints return raw IDs.

        Args:
            response_obj: Response object with an `id` attribute (ContainerObject, DeleteContainerResult, etc.)
            custom_llm_provider: Provider name (e.g., "azure", "openai")
            litellm_metadata: Optional litellm_metadata dict that may contain model_info.id
            extra_body: Optional extra_body dict that may contain target_model_names

        Returns:
            The same response object with encoded container_id (if routing metadata present)
        """
        # Extract model_id from litellm_metadata
        litellm_metadata = litellm_metadata or {}
        model_info: Final[dict[str, Any]] = litellm_metadata.get("model_info", {}) or {}
        model_id = model_info.get("id")

        # Check if we should encode based on routing metadata
        should_encode = False

        # Case 1: Router/proxy usage (model_id from router)
        if model_id is not None:
            should_encode = True

        # Case 2: target_model_names in extra_body (model-specific routing)
        if extra_body and "target_model_names" in extra_body:
            should_encode = True
            # Extract model_id from target_model_names if not already set
            if model_id is None:
                target_models: Final = extra_body["target_model_names"]
                # Use first model as model_id for encoding
                if isinstance(target_models, str):
                    model_id = target_models.split(",")[0].strip()
                elif isinstance(target_models, list) and len(target_models) > 0:
                    model_id = str(target_models[0]).strip()

        # Only encode if we have routing metadata
        if should_encode and response_obj and hasattr(response_obj, "id"):
            encoded_id: Final = ResponsesAPIRequestUtils._build_container_id(
                custom_llm_provider=custom_llm_provider,
                model_id=model_id,
                container_id=response_obj.id,
            )
            response_obj.id = encoded_id

        return response_obj
