"""
Support for OpenAI's `/v1/chat/completions` endpoint on deAPI.

deAPI serves open-source chat models through an OpenAI-compatible API, so requests go through the shared HTTP
handler untouched. Every billed answer reports the charge as `usage.cost`, which this config forwards to
LiteLLM's cost tracking, and streams always ask for the final usage chunk that carries it.

Docs: https://docs.deapi.ai
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm.litellm_core_utils.core_helpers import set_response_cost_in_hidden_params
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ..common_utils import (
    DeapiException,
    authorized_headers,
    endpoint_url,
    reported_cost,
    resolve_api_base,
    resolve_api_key,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding

_OPTIONAL_MAPPING: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)


class _DeapiModel(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str


class _DeapiModelCatalog(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_DeapiModel, ...]


def _stream_options_with_usage(request: Mapping[str, object]) -> Mapping[str, object]:
    current: Final = _OPTIONAL_MAPPING.validate_python(request.get("stream_options")) or MappingProxyType({})
    return MappingProxyType({**current, "include_usage": True})


class DeapiChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: inherited contract
        inherited: Final[list[str]] = super().get_supported_openai_params(model)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # mutable-ok: untyped base
        return [*inherited, *(("reasoning_effort",) if "reasoning_effort" not in inherited else ())]

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return resolve_api_key(api_key)

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return resolve_api_base(api_base)

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        headers: dict[str, object],  # mutable-ok: inherited contract
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        request: Final[dict[str, object]] = (  # pyright: ignore[reportUnknownVariableType]  # mutable-ok: untyped base
            super().transform_request(  # pyright: ignore[reportUnknownMemberType]  # untyped base signature
                model, messages, optional_params, litellm_params, headers
            )
        )
        if not request.get("stream"):
            return request
        return {**request, "stream_options": dict(_stream_options_with_usage(request))}

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, object]:
        if not (extra_body.get("stream") or request.get("stream")):
            return extra_body
        source: Final = extra_body if "stream_options" in extra_body else request
        return {**extra_body, "stream_options": dict(_stream_options_with_usage(source))}

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict[str, object],  # mutable-ok: inherited contract
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        encoding: "Encoding | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        response: Final = super().transform_response(  # pyright: ignore[reportUnknownMemberType]  # untyped base
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        set_response_cost_in_hidden_params(response, reported_cost(raw_response))
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return DeapiException(message=error_message, status_code=status_code, headers=headers)

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: inherited contract
        client: Final = litellm.module_level_client
        response: Final = client.get(  # pyright: ignore[reportUnknownMemberType]  # untyped params dict
            url=endpoint_url(api_base, "models"), headers=authorized_headers({}, api_key, "models")
        )
        if not response.is_success:
            raise DeapiException(status_code=response.status_code, message=response.text, headers=response.headers)
        catalog: Final = _DeapiModelCatalog.model_validate(response.json())
        return [f"deapi/{model.id}" for model in catalog.data]
