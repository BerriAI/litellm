from typing import List, Optional, Union

from httpx import Headers

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

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

        auth_headers = {"Authorization": "Bearer {}".format(api_key), **headers}
        content_type_header = (
            {} if any(key.lower() == "content-type" for key in auth_headers) else {"Content-Type": "application/json"}
        )
        return self._add_session_affinity_header({**auth_headers, **content_type_header}, litellm_params)

    def _add_session_affinity_header(self, headers: dict, litellm_params: dict) -> dict:
        if any(key.lower() == "x-session-affinity" for key in headers):
            return headers
        session_id = get_fireworks_session_id(litellm_params)
        if not session_id:
            return headers
        return {**headers, "x-session-affinity": session_id}
