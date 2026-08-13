from typing import Final

import httpx

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import _add_path_to_api_base


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
        if api_key is not None:
            headers["x-api-key"] = api_key
        return headers

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        api_base = api_base or get_secret_str("VLLM_API_BASE")
        if api_base is None:
            raise ValueError(
                "VLLM_API_BASE is not set. Please set the environment variable, to use VLLM's pass-through - `{LITELLM_API_BASE}/vllm/{endpoint}`."
            )
        return api_base

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return None

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        api_base = VLLMModelInfo.get_api_base(api_base)
        api_key = VLLMModelInfo.get_api_key(api_key)
        endpoint: Final = "/v1/models"
        if api_base is None or api_key is None:
            raise ValueError(
                "VLLM_API_BASE or VLLM_API_KEY is not set. Please set the environment variable, to query VLLM's `/models` endpoint."
            )

        url: Final = _add_path_to_api_base(api_base, endpoint)
        response: Final = litellm.module_level_client.get(
            url=url,
        )

        response.raise_for_status()

        models: Final = response.json()["data"]

        return [model["id"] for model in models]

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return VLLMError(status_code=status_code, message=error_message, headers=headers)
