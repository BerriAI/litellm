from typing import Any, List, Optional, Tuple, cast
from urllib.parse import urlparse

import httpx
from httpx import Response

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _audio_or_image_in_message_content,
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.llms.openai.common_utils import drop_params_from_unprocessable_entity_error
from litellm.llms.openai.openai import OpenAIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, ProviderField
from litellm.utils import _add_path_to_api_base, supports_tool_choice


class AzureAIStudioConfig(OpenAIConfig):
    def get_supported_openai_params(self, model: str) -> List:
        model_supports_tool_choice = True  # azure ai supports this by default
        if not supports_tool_choice(model=f"azure_ai/{model}"):
            model_supports_tool_choice = False
        supported_params = super().get_supported_openai_params(model)
        if not model_supports_tool_choice:
            filtered_supported_params = []
            for param in supported_params:
                if param != "tool_choice":
                    filtered_supported_params.append(param)
            return filtered_supported_params
        return supported_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if api_base and self._should_use_api_key_header(api_base):
            headers["api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"

        return headers

    def _should_use_api_key_header(self, api_base: str) -> bool:
        """
        Returns True if the request should use `api-key` header for authentication.
        """
        parsed_url = urlparse(api_base)
        host = parsed_url.hostname
        if host and (
            host.endswith(".services.ai.azure.com")
            or host.endswith(".openai.azure.com")
        ):
            return True
        return False

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
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
        if api_base is None:
            raise ValueError(
                f"api_base is required for Azure AI Studio. Please set the api_base parameter. Passed `api_base={api_base}`"
            )
        original_url = httpx.URL(api_base)

        # Extract api_version or use default
        api_version = cast(Optional[str], litellm_params.get("api_version"))

        # Create a new dictionary with existing params
        query_params = dict(original_url.params)

        # Add api_version if needed
        if "api-version" not in query_params and api_version:
            query_params["api-version"] = api_version

        # Add the path to the base URL
        if "services.ai.azure.com" in api_base:
            new_url = _add_path_to_api_base(
                api_base=api_base, ending_path="/models/chat/completions"
            )
        else:
            new_url = _add_path_to_api_base(
                api_base=api_base, ending_path="/chat/completions"
            )

        # Use the new query_params dictionary
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

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        extra_body = optional_params.pop("extra_body", {})
        if extra_body and isinstance(extra_body, dict):
            optional_params.update(extra_body)
        optional_params.pop("max_retries", None)
        return super().transform_request(
            model, messages, optional_params, litellm_params, headers
        )

    def transform_response(
        self,
        model: str,
        raw_response: Response,
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
        model_response.model = f"azure_ai/{model}"
        return super().transform_response(
            model=model,
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

    def should_retry_llm_api_inside_llm_translation_on_http_error(
        self, e: httpx.HTTPStatusError, litellm_params: dict
    ) -> bool:
        should_drop_params = litellm_params.get("drop_params") or litellm.drop_params
        error_text = e.response.text
        if should_drop_params and "Extra inputs are not permitted" in error_text:
            return True
        elif (
            "unknown field: parameter index is not a valid field" in error_text
        ):  # remove index from tool calls
            return True
        return super().should_retry_llm_api_inside_llm_translation_on_http_error(
            e=e, litellm_params=litellm_params
        )

    @property
    def max_retry_on_unprocessable_entity_error(self) -> int:
        return 2

    def transform_request_on_unprocessable_entity_error(
        self, e: httpx.HTTPStatusError, request_data: dict
    ) -> dict:
        _messages = cast(Optional[List[AllMessageValues]], request_data.get("messages"))
        if (
            "unknown field: parameter index is not a valid field" in e.response.text
            and _messages is not None
        ):
            litellm.remove_index_from_tool_calls(
                messages=_messages,
            )
        data = drop_params_from_unprocessable_entity_error(e=e, data=request_data)
        return data
