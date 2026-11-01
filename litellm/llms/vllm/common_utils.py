from collections.abc import Mapping
from typing import Final

import httpx

from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai_like.model_info import list_openai_compatible_model_ids
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import add_path_to_api_base


def endpoint_under_api_base(endpoint: str, api_base: str | None) -> str:
    return (
        endpoint.lstrip("/").removeprefix("v1/")
        if api_base is not None and api_base.rstrip("/").endswith("/v1")
        else endpoint
    )


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
        upstream_key: Final = self.upstream_api_key(api_key, api_base)
        return {**headers, **self.upstream_auth_headers(upstream_key)} if upstream_key else headers

    def upstream_api_key(self, api_key: str | None, api_base: str | None) -> str | None:
        return api_key

    def upstream_auth_headers(self, api_key: str) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {api_key}", "x-api-key": api_key}

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
        """
        Reads the server and key the same way `hosted_vllm` chat calls do. `get_api_key` above always answers None
        (the passthrough route sends no key), so model listing must not lean on it or every listing fails.
        """
        resolved_api_base: Final = api_base or get_secret_str("HOSTED_VLLM_API_BASE") or get_secret_str("VLLM_API_BASE")
        if resolved_api_base is None:
            raise ValueError(
                "Set api_base (or HOSTED_VLLM_API_BASE / VLLM_API_BASE) to list the models a vLLM server serves."
            )
        return list_openai_compatible_model_ids(
            models_url=add_path_to_api_base(resolved_api_base, "/v1/models"),
            api_key=api_key or get_secret_str("HOSTED_VLLM_API_KEY"),
        )

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return VLLMError(status_code=status_code, message=error_message, headers=headers)
