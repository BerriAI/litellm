"""
Wandb Hub chat completion transformation
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionAssistantMessage
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class WandbHubChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or "https://api.inference.wandb.ai/v1"
        dynamic_api_key = api_key or get_secret_str("WANDB_API_KEY") or ""
        return api_base, dynamic_api_key

    def _validate_project_id(self, optional_params: dict) -> str:
        """
        Validate and extract project_id from optional_params.
        Project ID is required for Wandb Hub API calls.
        """
        project_id = optional_params.get("project_id") or optional_params.get("project")
        if not project_id:
            raise ValueError(
                "project_id is required for Wandb Hub. "
                "Please provide it in optional_params as 'project_id' or 'project'."
            )
        return project_id

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # Validate and extract project_id
        project_id = self._validate_project_id(optional_params)
        
        # Remove project_id from optional_params as it's not needed in the request body
        optional_params_copy = optional_params.copy()
        optional_params_copy.pop("project_id", None)
        optional_params_copy.pop("project", None)
        
        # Add project to headers (as required by Wandb API)
        headers["project"] = project_id
        
        # Use parent class to handle standard OpenAI transformation
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params_copy,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        # Use the standard OpenAI-like response transformation
        response_json = raw_response.json()
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        returned_response = ModelResponse(**response_json)
        
        # Add wandb_hub prefix to model name
        returned_response.model = "wandb_hub/" + (returned_response.model or model)
        
        return returned_response

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = True,
    ) -> dict:
        mapped_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
        
        # Handle max_completion_tokens -> max_tokens conversion
        if (
            "max_completion_tokens" in non_default_params
            and replace_max_completion_tokens_with_max_tokens
        ):
            mapped_params["max_tokens"] = non_default_params["max_completion_tokens"]
            mapped_params.pop("max_completion_tokens", None)

        return mapped_params