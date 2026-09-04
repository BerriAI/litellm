from typing import Annotated, Final

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
    def __init__(self, provider: str = "vllm") -> None:
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

    def _get_discovery_api_key(self, api_key: str | None) -> str | None:
        environment_variable: Final = "HOSTED_VLLM_API_KEY" if self._provider == "hosted_vllm" else "VLLM_API_KEY"
        return api_key or get_secret_str(environment_variable)

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        passed_api_base: Final = api_base
        resolved_api_base: Final = self._get_discovery_api_base(api_base)
        resolved_api_key: Final = self._get_discovery_api_key(api_key) if passed_api_base is None or api_key else None
        endpoint: Final = "/v1/models"

        url: Final = _add_path_to_api_base(resolved_api_base, endpoint)
        headers: Final = (
            httpx.Headers((("Authorization", f"Bearer {resolved_api_key}"),)) if resolved_api_key else httpx.Headers()
        )
        response: Final = litellm.module_level_client.get(
            url=url,
            headers=headers,
        )

        response.raise_for_status()

        models: Final = response.json()["data"]

        return [model["id"] for model in models]

    @staticmethod
    def _strip_provider_prefix(model: str) -> str:
        for prefix in ("hosted_vllm/", "vllm/"):
            if model.startswith(prefix):
                return model[len(prefix) :]
        return model

    def _model_info_from_entry(
        self,
        raw_entry: object,
        *,
        target: str,
        model: str,
    ) -> ModelInfoBase | None:
        try:
            entry: Final = _VLLMModelEntry.model_validate(raw_entry)
        except ValidationError:
            return None
        if entry.id != target or entry.max_model_len is None:
            return None

        model_info: Final[ModelInfoBase] = {
            "key": model,
            "litellm_provider": self._provider,
            "mode": "chat",
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "max_tokens": None,
            "max_input_tokens": entry.max_model_len,
            "max_output_tokens": None,
        }
        return model_info

    def get_model_info(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> ModelInfoBase | None:
        passed_api_base: Final = api_base
        resolved_api_base: Final = self._get_discovery_api_base(api_base)
        resolved_api_key: Final = self._get_discovery_api_key(api_key) if passed_api_base is None or api_key else None

        headers: Final = (
            httpx.Headers((("Authorization", f"Bearer {resolved_api_key}"),)) if resolved_api_key else httpx.Headers()
        )
        response: Final = litellm.module_level_client.get(
            url=_add_path_to_api_base(resolved_api_base, "/v1/models"),
            headers=headers,
        )
        response.raise_for_status()

        target: Final = self._strip_provider_prefix(model)
        discovered: Final = _VLLMModelsResponse.model_validate(response.json())
        return next(
            (
                model_info
                for raw_entry in discovered.data
                if (
                    model_info := self._model_info_from_entry(
                        raw_entry,
                        target=target,
                        model=model,
                    )
                )
                is not None
            ),
            None,
        )

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return VLLMError(status_code=status_code, message=error_message, headers=headers)
