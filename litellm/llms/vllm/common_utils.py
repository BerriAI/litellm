from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm.constants import NON_LLM_CONNECTION_TIMEOUT, PLACEHOLDER_API_KEY
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import get_httpx_client
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


_EXTRA_HEADERS: Final = TypeAdapter(Mapping[str, str])


def _auth_headers(api_key: str | None) -> Mapping[str, str]:
    if api_key is None or api_key == PLACEHOLDER_API_KEY:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _str_param(litellm_params: Mapping[str, object], key: str) -> str | None:
    value: Final = litellm_params.get(key)
    return value if isinstance(value, str) else None


class VLLMModelInfo(BaseLLMModelInfo):
    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        upstream_key: Final = self.upstream_api_key(api_key, api_base)
        return {**headers, **self.upstream_auth_headers(upstream_key)} if upstream_key else headers

    def upstream_api_key(self, api_key: str | None, api_base: str | None) -> str | None:
        return api_key

    def upstream_auth_headers(self, api_key: str) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {api_key}", "x-api-key": api_key}

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
        return None

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
        return self._list_models(api_key=api_key, api_base=api_base, extra_headers={}, ssl_verify=None)

    def discover_models(
        self, litellm_params: Mapping[str, object] | None = None
    ) -> list[str]:  # mutable-ok: matches get_models' list[str] contract shared by every provider override
        params: Final = litellm_params or {}
        ssl_verify: Final = params.get("ssl_verify")
        return self._list_models(
            api_key=_str_param(params, "api_key"),
            api_base=_str_param(params, "api_base"),
            extra_headers=_EXTRA_HEADERS.validate_python(params.get("extra_headers") or {}),
            ssl_verify=ssl_verify if isinstance(ssl_verify, (bool, str)) else None,
        )

    def _list_models(
        self,
        *,
        api_key: str | None,
        api_base: str | None,
        extra_headers: Mapping[str, str],
        ssl_verify: bool | str | None,
    ) -> list[str]:  # mutable-ok: matches get_models' list[str] contract shared by every provider override
        url: Final = add_path_to_api_base(self.get_api_base(api_base), "/v1/models")
        response: Final = get_httpx_client(params={"ssl_verify": ssl_verify}).get(
            url=url,
            headers={**extra_headers, **_auth_headers(self.get_api_key(api_key))},
            timeout=NON_LLM_CONNECTION_TIMEOUT,
        )
        response.raise_for_status()
        return [model["id"] for model in response.json()["data"]]
