"""
Transformation for Azure AI Foundry Model Router.

The Model Router is a special Azure AI deployment that automatically routes requests
to the best available model. It has specific cost tracking requirements.
"""

from typing import Any, Final

from httpx import Response

from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig
from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse


class AzureModelRouterConfig(AzureAIStudioConfig):
    """
    Configuration for Azure AI Foundry Model Router.

    Handles:
    - Stripping model_router prefix before sending to Azure API
    - Preserving full model path in responses for cost tracking
    - Calculating flat infrastructure costs for Model Router
    """

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform request for Model Router.

        Strips the model_router/ prefix so only the deployment name is sent to Azure.
        Example: model_router/azure-model-router -> azure-model-router
        """
        from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo

        # Get base model name (strips routing prefixes like model_router/)
        base_model: Final[str] = AzureFoundryModelInfo.get_base_model(model)

        return super().transform_request(base_model, messages, optional_params, litellm_params, headers)

    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        """
        Transform response for Model Router.

        Extracts the actual model used from the Azure response (e.g., gpt-5-nano-2025-08-07)
        and returns it with the azure_ai/ prefix for proper display and cost tracking.

        Also stamps that model onto ``_hidden_params`` so downstream consumers (spend logs,
        response restamping) can read it instead of guessing the route from the model string.
        """
        from litellm.llms.azure_ai.common_utils import (
            AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY,
            AzureFoundryModelInfo,
        )
        from litellm.router_utils.add_retry_fallback_headers import (
            get_hidden_params_dict,
        )

        # Get base model for the parent call (strips routing prefixes for API compatibility)
        base_model: Final[str] = AzureFoundryModelInfo.get_base_model(model)

        # Call parent transform_response first - this will extract the actual model
        # from the raw response (e.g., "gpt-5-nano-2025-08-07")
        transformed_response: Final = super().transform_response(
            model=base_model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        selected_model: Final = transformed_response.model
        if selected_model:
            # Rebuilt rather than mutated in place: ModelResponseBase declares _hidden_params as a
            # class-level dict, so an in-place write can bleed into unrelated responses.
            transformed_response._hidden_params = {  # pyright: ignore[reportPrivateUsage]  # ModelResponse exposes no public hidden-params setter  # mutable-ok: ModelResponse requires _hidden_params to be a plain dict
                **get_hidden_params_dict(transformed_response),
                AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY: selected_model,
            }
        return transformed_response

    def calculate_additional_costs(self, model: str, prompt_tokens: int, completion_tokens: int) -> dict | None:
        """
        Calculate additional costs for Azure Model Router.

        Adds a flat infrastructure cost of $0.14 per M input tokens for using the Model Router.

        Args:
            model: The model name (should be a model router model)
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Dictionary with additional costs, or None if not applicable.
        """
        from litellm.llms.azure_ai.cost_calculator import (
            calculate_azure_model_router_flat_cost,
        )

        flat_cost: Final = calculate_azure_model_router_flat_cost(model=model, prompt_tokens=prompt_tokens)

        if flat_cost > 0:
            return {"Azure Model Router Flat Cost": flat_cost}

        return None
