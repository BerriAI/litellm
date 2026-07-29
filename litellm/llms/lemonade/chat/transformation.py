"""
Translate from OpenAI's `/v1/chat/completions` to Lemonade's `/v1/chat/completions`
"""

from typing import Any, List, Optional, Tuple, Union
from urllib.parse import quote

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
)
from litellm.types.utils import ModelResponse

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class LemonadeChatConfig(OpenAILikeChatConfig):
    _DEFAULT_API_KEY = "lemonade"

    repeat_penalty: Optional[float] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None

    def __init__(
        self,
        repeat_penalty: Optional[float] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "lemonade"

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_models(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        Get available models from Lemonade API.

        This method queries the Lemonade /models endpoint to retrieve the list of available models.

        Args:
            api_key: Optional API key for authenticated Lemonade servers
            api_base: Optional API base URL (defaults to LEMONADE_API_BASE env var or http://localhost:8000)

        Returns:
            List of model names prefixed with "lemonade/"
        """
        api_base, api_key = self._get_openai_compatible_provider_info(api_base=api_base, api_key=api_key)

        if api_base is None:
            raise ValueError(
                "LEMONADE_API_BASE is not set. Please set the environment variable to query Lemonade's /models endpoint."
            )

        # Getting the list of models from lemonade
        try:
            response = litellm.module_level_client.get(
                url=f"{api_base}/models",
                headers=self._get_auth_headers(api_key),
            )
        except Exception as e:
            raise ValueError(
                f"Failed to fetch models from Lemonade. Set Lemonade API Base via `LEMONADE_API_BASE` environment variable. Error: {e}"
            )

        if response.status_code != 200:
            raise ValueError(
                f"Failed to fetch models from Lemonade. Status code: {response.status_code}, Response: {response.text}"
            )

        model_list = response.json().get("data", [])
        return ["lemonade/" + model["id"] for model in model_list]

    @staticmethod
    def _get_positive_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                return None
            if parsed > 0:
                return parsed
        return None

    @staticmethod
    def _get_provider_specific_entry(model_info: dict) -> dict:
        provider_specific_entry = model_info.get("provider_specific_entry")
        if not isinstance(provider_specific_entry, dict):
            provider_specific_entry = {}
        else:
            provider_specific_entry = provider_specific_entry.copy()

        for key in ("recipe_options", "context_window", "max_context_window"):
            if key in model_info:
                provider_specific_entry[key] = model_info[key]

        return provider_specific_entry

    def _get_context_window(self, model_info: dict) -> Optional[int]:
        provider_specific_entry = self._get_provider_specific_entry(model_info)
        recipe_options = provider_specific_entry.get("recipe_options")
        if not isinstance(recipe_options, dict):
            recipe_options = {}

        for value in (
            recipe_options.get("ctx_size"),
            model_info.get("max_input_tokens"),
            provider_specific_entry.get("context_window"),
            provider_specific_entry.get("max_context_window"),
        ):
            parsed = self._get_positive_int(value)
            if parsed is not None:
                return parsed
        return None

    def _get_default_model_info(self, model: str) -> dict:
        return {
            "key": "lemonade/" + model,
            "litellm_provider": "lemonade",
            "mode": "chat",
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "max_tokens": None,
            "max_input_tokens": None,
            "max_output_tokens": None,
        }

    def get_model_info(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Any:
        if model.startswith("lemonade/"):
            model = model.split("/", 1)[1]

        api_base, api_key = self._get_openai_compatible_provider_info(api_base=api_base, api_key=api_key)
        encoded_model = quote(model, safe="")

        try:
            response = litellm.module_level_client.get(
                url=f"{api_base}/models/{encoded_model}",
                headers=self._get_auth_headers(api_key),
            )
            response.raise_for_status()
            model_info = response.json()
        except Exception:
            verbose_logger.debug("LemonadeError: Could not get model info.")
            return self._get_default_model_info(model)

        max_input_tokens = self._get_context_window(model_info)
        max_output_tokens = self._get_positive_int(model_info.get("max_output_tokens"))
        max_tokens = self._get_positive_int(model_info.get("max_tokens"))
        provider_specific_entry = self._get_provider_specific_entry(model_info)

        model_info_response = self._get_default_model_info(model)
        model_info_response.update(
            {
                "max_tokens": max_tokens or max_output_tokens,
                "max_input_tokens": max_input_tokens,
                "max_output_tokens": max_output_tokens,
            }
        )
        if provider_specific_entry:
            model_info_response["provider_specific_entry"] = provider_specific_entry
        return model_info_response

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # lemonade is openai compatible, we just need to set this to custom_openai and have the api_base be lemonade's endpoint
        passed_api_base = api_base
        api_base = api_base or get_secret_str("LEMONADE_API_BASE") or "http://localhost:8000/api/v1"  # type: ignore
        key = self._DEFAULT_API_KEY
        if passed_api_base is None or api_key:
            key = api_key or litellm.lemonade_key or get_secret_str("LEMONADE_API_KEY") or self._DEFAULT_API_KEY
        return api_base, key

    def _get_auth_headers(self, api_key: Optional[str]) -> dict:
        if api_key is None or api_key == self._DEFAULT_API_KEY:
            return {}
        return {"Authorization": f"Bearer {api_key}"}

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
        model_response = super().transform_response(
            model=model,
            model_response=model_response,
            raw_response=raw_response,
            messages=messages,
            logging_obj=logging_obj,
            request_data=request_data,
            encoding=encoding,
            optional_params=optional_params,
            json_mode=json_mode,
            litellm_params=litellm_params,
            api_key=api_key,
        )

        # Storing lemonade in the model response for easier cost calculation later
        setattr(model_response, "model", "lemonade/" + model)

        return model_response
