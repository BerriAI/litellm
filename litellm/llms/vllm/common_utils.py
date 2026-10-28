from collections.abc import Mapping
from typing import Final

import httpx

import litellm
from litellm.constants import NON_LLM_CONNECTION_TIMEOUT
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import add_path_to_api_base


class VLLMError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        request: httpx.Request | None = None,
        response: httpx.Response | None = None,
        headers: httpx.Headers | dict | None = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
        )


def _auth_headers(api_key: str | None) -> Mapping[str, str]:
    if api_key is None:
        return {}
    return {"x-api-key": api_key, "Authorization": f"Bearer {api_key}"}


class VLLMModelInfo(BaseLLMModelInfo):
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        headers.update(_auth_headers(api_key))
        return headers

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        api_base = api_base or get_secret_str("VLLM_API_BASE")
        if api_base is None:
            raise ValueError(
                "VLLM_API_BASE is not set. Please set the environment variable, to use VLLM's pass-through - `{LITELLM_API_BASE}/vllm/{endpoint}`."
            )
        return api_base

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: mirrors override contract
        return []

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return VLLMError(status_code=status_code, message=error_message, headers=headers)


class HostedVLLMModelInfo(VLLMModelInfo):
    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        resolved_api_base: Final = api_base or get_secret_str("HOSTED_VLLM_API_BASE")
        if resolved_api_base is None:
            raise ValueError("HOSTED_VLLM_API_BASE is not set.")
        return resolved_api_base

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("HOSTED_VLLM_API_KEY")

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        url: Final = add_path_to_api_base(self.get_api_base(api_base), "/v1/models")
        response: Final = litellm.module_level_client.get(
            url=url, headers=dict(_auth_headers(self.get_api_key(api_key))), timeout=NON_LLM_CONNECTION_TIMEOUT
        )
        response.raise_for_status()
        return [model["id"] for model in response.json()["data"]]
