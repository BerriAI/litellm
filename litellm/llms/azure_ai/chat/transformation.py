from typing import List, Optional, Tuple, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _audio_or_image_in_message_content,
    convert_content_list_to_str,
)
from litellm.llms.openai.openai import OpenAIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ProviderField
from litellm.utils import _add_path_to_api_base


class AzureAIStudioConfig(OpenAIConfig):
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if api_base and "services.ai.azure.com" in api_base:
            headers["api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"

        return headers

    def get_complete_url(
        self,
        api_base: str,
        model: str,
        optional_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Constructs a complete URL for the API request.

        Args:
        - api_base: Base URL, e.g.,
            "https://litellm8397336933.services.ai.azure.com"
            OR
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
        - model: Model name.
        - optional_params: Additional query parameters, including "api_version".
        - stream: If streaming is required (optional).

        Returns:
        - A complete URL string, e.g.,
        "https://litellm8397336933.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
        """
        original_url = httpx.URL(api_base)

        # Extract api_version or use default
        api_version = cast(Optional[str], optional_params.get("api_version"))

        # Check if 'api-version' is already present
        if "api-version" not in original_url.params and api_version:
            # Add api_version to optional_params
            original_url.params["api-version"] = api_version

        # Add the path to the base URL
        if "services.ai.azure.com" in api_base:
            new_url = _add_path_to_api_base(
                api_base=api_base, ending_path="/models/chat/completions"
            )
        else:
            new_url = _add_path_to_api_base(
                api_base=api_base, ending_path="/chat/completions"
            )

        # Convert optional_params to query parameters
        query_params = original_url.params
        final_url = httpx.URL(new_url).copy_with(params=query_params)

        return str(final_url)

    def get_required_params(self) -> List[ProviderField]:
        """For a given provider, return it's required fields with a description"""
        return [
            ProviderField(
                field_name="api_key",
                field_type="string",
                field_description="Your Azure AI Studio API Key.",
                field_value="zEJ...",
            ),
            ProviderField(
                field_name="api_base",
                field_type="string",
                field_description="Your Azure AI Studio API Base.",
                field_value="https://Mistral-serverless.",
            ),
        ]

    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
    ) -> List:
        """
        - Azure AI Studio doesn't support content as a list. This handles:
            1. Transforms list content to a string.
            2. If message contains an image or audio, send as is (user-intended)
        """
        for message in messages:

            # Do nothing if the message contains an image or audio
            if _audio_or_image_in_message_content(message):
                continue

            texts = convert_content_list_to_str(message=message)
            if texts:
                message["content"] = texts
        return messages

    def _is_azure_openai_model(self, model: str, api_base: Optional[str]) -> bool:
        try:
            if "/" in model:
                model = model.split("/", 1)[1]
            if (
                model in litellm.open_ai_chat_completion_models
                or model in litellm.open_ai_text_completion_models
                or model in litellm.open_ai_embedding_models
            ):
                return True

        except Exception:
            return False
        return False

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        api_base = api_base or get_secret_str("AZURE_AI_API_BASE")
        dynamic_api_key = api_key or get_secret_str("AZURE_AI_API_KEY")

        if self._is_azure_openai_model(model=model, api_base=api_base):
            verbose_logger.debug(
                "Model={} is Azure OpenAI model. Setting custom_llm_provider='azure'.".format(
                    model
                )
            )
            custom_llm_provider = "azure"
        return api_base, dynamic_api_key, custom_llm_provider
