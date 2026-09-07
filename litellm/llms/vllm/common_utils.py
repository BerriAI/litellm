from typing import Annotated, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelInfoBase
from litellm.utils import _add_path_to_api_base


class _VLLMModelEntry(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    max_model_len: Annotated[int, Field(strict=True, gt=0)] | None = None


class _VLLMModelsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    data: tuple[object, ...]


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
    def __init__(self, provider: Literal["vllm", "hosted_vllm"] = "vllm") -> None:
        self._provider: Final = provider

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

    def _get_discovery_api_base(self, api_base: str | None) -> str:
        environment_variable: Final = "HOSTED_VLLM_API_BASE" if self._provider == "hosted_vllm" else "VLLM_API_BASE"
        resolved_api_base: Final = api_base or get_secret_str(environment_variable)
        if resolved_api_base is None:
            raise ValueError(f"{environment_variable} is required to query vLLM's `/models` endpoint.")
        return resolved_api_base

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def _query_models(self, api_base: str | None, api_key: str | None) -> httpx.Response:
        resolved_api_base: Final = self._get_discovery_api_base(api_base)
        environment_variable: Final = "HOSTED_VLLM_API_KEY" if self._provider == "hosted_vllm" else "VLLM_API_KEY"
        resolved_api_key: Final = (
            api_key if api_key is not None or api_base is not None else get_secret_str(environment_variable)
        )
        headers: Final = {"Authorization": f"Bearer {resolved_api_key}"} if resolved_api_key else {}
        response: Final = litellm.module_level_client.get(
            url=_add_path_to_api_base(resolved_api_base, "/v1/models"),
            headers=headers,
            follow_redirects=False,
            timeout=5.0,
        )
        response.raise_for_status()
        return response

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        response: Final = self._query_models(api_base, api_key)
        models: Final = response.json()["data"]

        return [model["id"] for model in models]

    def get_model_info(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> ModelInfoBase | None:
        response: Final = self._query_models(api_base, api_key)
        target: Final = model.removeprefix(f"{self._provider}/")
        discovered: Final = _VLLMModelsResponse.model_validate(response.json())
        for raw_entry in discovered.data:
            try:
                entry = _VLLMModelEntry.model_validate(raw_entry)
            except ValidationError:
                continue
            if entry.id == target and entry.max_model_len is not None:
                return ModelInfoBase(
                    key=model,
                    litellm_provider=self._provider,
                    mode="chat",
                    input_cost_per_token=0.0,
                    output_cost_per_token=0.0,
                    max_tokens=None,
                    max_input_tokens=entry.max_model_len,
                    max_output_tokens=None,
                )
        return None

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return VLLMError(status_code=status_code, message=error_message, headers=headers)
