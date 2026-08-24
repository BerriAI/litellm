"""
Translate from OpenAI's `/v1/chat/completions` to the Opper AI gateway's `/v3/compat/chat/completions`
"""

from collections.abc import AsyncIterator, Iterator
from typing import Final

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import OPPER_API_BASE
from litellm.llms.base_llm.chat.transformation import BaseLLMException, LiteLLMLoggingObj
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, ModelResponseStream, StreamingChoices

from ...openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
    OpenAIGPTConfig,
)
from ..common_utils import OpperException


class OpperConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "opper"

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: mirrors the OpenAIGPTConfig signature
        supported_params: Final = super().get_supported_openai_params(model=model)
        if "reasoning_effort" not in supported_params:
            supported_params.append("reasoning_effort")
        return supported_params

    def _should_preserve_cache_control_for_endpoint(
        self,
        custom_llm_provider: str | None,
        api_base: str | None,
    ) -> bool:
        return True

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: mirrors the OpenAIGPTConfig signature
        optional_params: dict,  # mutable-ok: mirrors the OpenAIGPTConfig signature
        litellm_params: dict,  # mutable-ok: mirrors the OpenAIGPTConfig signature
        headers: dict,  # mutable-ok: mirrors the OpenAIGPTConfig signature
    ) -> dict:  # mutable-ok: mirrors the OpenAIGPTConfig signature
        request: Final = super().transform_request(model, messages, optional_params, litellm_params, headers)
        if request.get("stream"):
            stream_options: Final = dict(request.get("stream_options") or {})  # mutable-ok: JSON request body
            stream_options.setdefault("include_usage", True)
            request["stream_options"] = stream_options
        return request

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,  # mutable-ok: mirrors the OpenAIGPTConfig signature
        messages: list[AllMessageValues],  # mutable-ok: mirrors the OpenAIGPTConfig signature
        optional_params: dict,  # mutable-ok: mirrors the OpenAIGPTConfig signature
        litellm_params: dict,  # mutable-ok: mirrors the OpenAIGPTConfig signature
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        transformed: Final = super().transform_response(
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

        try:
            response_json: Final = raw_response.json()
            if response_json.get("usage"):
                response_cost: Final = response_json["usage"].get("cost")
                if response_cost is not None:
                    headers: Final = transformed._hidden_params.setdefault(
                        "additional_headers",
                        {},  # mutable-ok: _hidden_params carries plain dicts
                    )
                    headers["llm_provider-x-litellm-response-cost"] = float(response_cost)
        except (ValueError, TypeError, KeyError, AttributeError):
            verbose_logger.debug("Opper: could not extract usage.cost from response")

        return transformed

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: mirrors the BaseConfig signature
    ) -> BaseLLMException:
        return OpperException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("OPPER_API_KEY")

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("OPPER_API_BASE") or OPPER_API_BASE

    def get_models(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> list[str]:  # mutable-ok: mirrors the BaseLLMModelInfo signature
        resolved_api_base: Final = self.get_api_base(api_base) or OPPER_API_BASE
        resolved_api_key: Final = self.get_api_key(api_key)
        if resolved_api_key is None:
            raise ValueError("Opper API key is not set. Pass api_key or set OPPER_API_KEY")

        response: Final = litellm.module_level_client.get(
            url=f"{resolved_api_base.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {resolved_api_key}"},  # mutable-ok: httpx request headers
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Failed to fetch models from Opper. Status code: {response.status_code}, Response: {response.text}"
            ) from e

        return [f"opper/{model['id']}" for model in response.json()["data"]]  # mutable-ok: base returns list[str]

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> "OpperChatCompletionStreamingHandler":
        return OpperChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class OpperChatCompletionStreamingHandler(OpenAIChatCompletionStreamingHandler):
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:  # mutable-ok: mirrors the base chunk_parser signature
        parsed: Final = super().chunk_parser(chunk)
        if getattr(parsed, "usage", None) is not None and not parsed.choices:
            parsed.choices = [StreamingChoices()]  # mutable-ok: ModelResponseStream.choices is a list
        return parsed
