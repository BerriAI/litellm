"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as Requesty is openai-compatible.

Requesty is an OpenAI-compatible LLM gateway using the same `provider/model`
naming convention as OpenRouter, so this config mirrors the OpenRouter one.

Docs: https://docs.requesty.ai
"""

from collections.abc import AsyncIterator, Iterator
from typing import Final

import httpx

import litellm
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, ModelResponseStream, StreamingChoices

from ...openrouter.chat.transformation import OpenrouterConfig
from ..common_utils import RequestyException

REQUESTY_REASONING_PARAMS: Final = ("reasoning_effort", "thinking")


class RequestyConfig(OpenrouterConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "requesty"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = api_base or get_secret_str("REQUESTY_API_BASE") or "https://router.requesty.ai/v1"
        dynamic_api_key: Final = api_key or get_secret_str("REQUESTY_API_KEY")
        return resolved_api_base, dynamic_api_key

    def _supports_reasoning(self, model: str) -> bool:
        try:
            return litellm.supports_reasoning(
                model=model, custom_llm_provider="requesty"
            ) or litellm.supports_reasoning(model=model)
        except Exception:
            return False

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base signature
        supported_params: Final = super(OpenrouterConfig, self).get_supported_openai_params(model=model)
        if not self._supports_reasoning(model):
            return supported_params
        return list(dict.fromkeys((*supported_params, *REQUESTY_REASONING_PARAMS)))  # mutable-ok: base returns a list

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: base signature
        optional_params: dict,  # mutable-ok: base signature
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: base signature
        mapped_params: Final = (
            {**non_default_params, "reasoning_effort": "xhigh"}  # mutable-ok: JSON params dict
            if non_default_params.get("reasoning_effort") == "max"
            else non_default_params
        )
        return super(OpenrouterConfig, self).map_openai_params(mapped_params, optional_params, model, drop_params)

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: base signature
        optional_params: dict,  # mutable-ok: base signature
        litellm_params: dict,  # mutable-ok: base signature
        headers: dict,  # mutable-ok: base signature
    ) -> dict:  # mutable-ok: base signature
        transformed_messages: Final = (
            self._move_cache_control_to_content(messages)
            if self._supports_cache_control_in_content(model)
            else messages
        )

        extra_body: Final = optional_params.pop("extra_body", {})  # mutable-ok: caller JSON dict
        response: Final = super(OpenrouterConfig, self).transform_request(
            model, transformed_messages, optional_params, litellm_params, headers
        )
        # `extra_body` is client-controlled. Do not let it overwrite the canonical
        # request fields that have already been resolved and authorized (e.g. `model`,
        # `messages`), otherwise a caller could route to an unauthorized model after
        # model-authorization and request-inspection checks have run.
        protected_fields: Final = frozenset({"model", "messages"})
        return {  # mutable-ok: JSON body
            **response,
            **{key: value for key, value in extra_body.items() if key not in protected_fields},  # mutable-ok: JSON body
        }

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: base signature
    ) -> BaseLLMException:
        return RequestyException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> BaseModelResponseIterator:
        return RequestyChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


def _delta_with_reasoning(delta: dict) -> dict:  # mutable-ok: wire JSON delta
    return {**delta, "reasoning_content": delta.get("reasoning")}  # mutable-ok: wire JSON delta


class RequestyChatCompletionStreamingHandler(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:  # mutable-ok: base signature
        try:
            if "error" in chunk:
                error_chunk: Final = chunk["error"]
                error_metadata: Final = error_chunk.get("metadata", {})  # mutable-ok: JSON metadata
                raise RequestyException(
                    message="Message: {}, Metadata: {}".format(error_chunk["message"], error_metadata),
                    status_code=error_chunk["code"],
                    headers=error_metadata.get("headers", {}),  # mutable-ok: JSON headers
                )

            choices: Final = [  # mutable-ok: choices list
                StreamingChoices(**{**choice, "delta": _delta_with_reasoning(choice["delta"])})  # mutable-ok: JSON
                for choice in chunk["choices"]
            ]
            return ModelResponseStream(
                id=chunk["id"],
                object="chat.completion.chunk",
                created=chunk["created"],
                usage=chunk.get("usage"),
                model=chunk["model"],
                choices=choices,
            )
        except KeyError as e:
            raise RequestyException(
                message=f"KeyError: {e}, Got unexpected response from Requesty: {chunk}",
                status_code=400,
                headers={"Content-Type": "application/json"},  # mutable-ok: JSON headers
            )
