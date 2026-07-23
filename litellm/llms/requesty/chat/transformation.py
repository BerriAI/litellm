"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as Requesty is openai-compatible.

Requesty is an OpenAI-compatible LLM gateway using the same `provider/model`
naming convention as OpenRouter, so this config mirrors the OpenRouter one.

Docs: https://docs.requesty.ai
"""

from typing import List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openrouter.chat.transformation import OpenrouterConfig
from ..common_utils import RequestyException


class RequestyConfig(OpenrouterConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "requesty"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("REQUESTY_API_BASE") or "https://router.requesty.ai/v1"
        dynamic_api_key = api_key or get_secret_str("REQUESTY_API_KEY")
        return api_base, dynamic_api_key

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        if non_default_params.get("reasoning_effort") == "max":
            non_default_params = {**non_default_params, "reasoning_effort": "xhigh"}

        return super(OpenrouterConfig, self).map_openai_params(non_default_params, optional_params, model, drop_params)

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        if self._supports_cache_control_in_content(model):
            messages = self._move_cache_control_to_content(messages)

        extra_body = optional_params.pop("extra_body", {})
        response = super(OpenrouterConfig, self).transform_request(
            model, messages, optional_params, litellm_params, headers
        )
        # `extra_body` is client-controlled. Do not let it overwrite the canonical
        # request fields that have already been resolved and authorized (e.g. `model`,
        # `messages`), otherwise a caller could route to an unauthorized model after
        # model-authorization and request-inspection checks have run.
        protected_fields = {"model", "messages"}
        response.update({key: value for key, value in extra_body.items() if key not in protected_fields})
        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return RequestyException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
