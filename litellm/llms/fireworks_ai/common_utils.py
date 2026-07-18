from typing import List, Optional, Union

from httpx import Headers

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Usage

from ..base_llm.chat.transformation import BaseLLMException


class FireworksAIException(BaseLLMException):
    pass


def get_fireworks_session_id(litellm_params: dict) -> str | None:
    params = litellm_params
    for key in ("litellm_session_id", "session_id"):
        value = params.get(key)
        if value:
            return str(value)
    metadata = params.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("session_id")
        if value:
            return str(value)
    value = params.get("litellm_trace_id")
    if value:
        return str(value)
    return None


def normalize_fireworks_usage(usage: Usage) -> Usage:
    if "cache_read_input_tokens" in usage:
        return usage

    prompt_tokens_details = usage.prompt_tokens_details
    cached_tokens = getattr(prompt_tokens_details, "cached_tokens", 0) or 0
    if not isinstance(cached_tokens, int) or cached_tokens <= 0:
        return usage

    usage.cache_read_input_tokens = cached_tokens
    usage._cache_read_input_tokens = cached_tokens
    return usage


class FireworksAIMixin:
    """
    Common Base Config functions across Fireworks AI Endpoints
    """

    def get_error_class(self, error_message: str, status_code: int, headers: Union[dict, Headers]) -> BaseLLMException:
        return FireworksAIException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def _get_api_key(self, api_key: Optional[str]) -> Optional[str]:
        dynamic_api_key = api_key or (
            get_secret_str("FIREWORKS_API_KEY")
            or get_secret_str("FIREWORKS_AI_API_KEY")
            or get_secret_str("FIREWORKSAI_API_KEY")
            or get_secret_str("FIREWORKS_AI_TOKEN")
        )
        return dynamic_api_key

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
        api_key = self._get_api_key(api_key)
        if api_key is None:
            raise ValueError("FIREWORKS_API_KEY is not set")

        validated_headers = {"Authorization": "Bearer {}".format(api_key), **headers}
        if not any(key.lower() == "x-session-affinity" for key in validated_headers):
            session_id = get_fireworks_session_id(litellm_params)
            if session_id:
                validated_headers["x-session-affinity"] = session_id
        return validated_headers
