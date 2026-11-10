"""
Handler for transforming /chat/completions api requests to litellm.responses requests
"""

import json
from collections.abc import Coroutine, Mapping
from typing import TYPE_CHECKING, Final

from typing_extensions import TypedDict

from litellm._internal_context import internal_sub_call

if TYPE_CHECKING:
    import httpx

    from litellm import LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
    from litellm.types.utils import CacheControlsHint


def _header_text(value: object) -> object:
    return json.dumps(value) if isinstance(value, bool | int | float) else value


def _headers_as_text(extra_headers: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if extra_headers is None:
        return None
    return {name: _header_text(value) for name, value in extra_headers.items()}


class SpeechToCompletionBridgeHandlerInputKwargs(TypedDict):
    model: str
    input: str
    voice: str | dict | None
    optional_params: dict[str, object]
    litellm_params: dict
    logging_obj: "LiteLLMLoggingObj"
    headers: dict
    custom_llm_provider: str


class SpeechToCompletionBridgeHandler:
    def __init__(self):
        from .transformation import SpeechToCompletionBridgeTransformationHandler

        super().__init__()
        self.transformation_handler = SpeechToCompletionBridgeTransformationHandler()

    def validate_input_kwargs(self, kwargs: dict) -> SpeechToCompletionBridgeHandlerInputKwargs:
        from litellm import LiteLLMLoggingObj

        model: Final = kwargs.get("model")
        if model is None or not isinstance(model, str):
            raise ValueError("model is required")

        custom_llm_provider: Final = kwargs.get("custom_llm_provider")
        if custom_llm_provider is None or not isinstance(custom_llm_provider, str):
            raise ValueError("custom_llm_provider is required")

        input: Final = kwargs.get("input")
        if input is None or not isinstance(input, str):
            raise ValueError("input is required")

        optional_params: Final = kwargs.get("optional_params")
        if optional_params is None or not isinstance(optional_params, dict):
            raise ValueError("optional_params is required")

        litellm_params: Final = kwargs.get("litellm_params")
        if litellm_params is None or not isinstance(litellm_params, dict):
            raise ValueError("litellm_params is required")

        headers = kwargs.get("headers")
        if headers is None or not isinstance(headers, dict):
            raise ValueError("headers is required")

        headers = kwargs.get("headers")
        if headers is None or not isinstance(headers, dict):
            raise ValueError("headers is required")

        logging_obj: Final = kwargs.get("logging_obj")
        if logging_obj is None or not isinstance(logging_obj, LiteLLMLoggingObj):
            raise ValueError("logging_obj is required")

        return SpeechToCompletionBridgeHandlerInputKwargs(
            model=model,
            input=input,
            voice=kwargs.get("voice"),
            optional_params=optional_params,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            headers=headers,
        )

    def speech(
        self,
        model: str,
        input: str,
        voice: str | dict | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str,
        api_base: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout: "float | httpx.Timeout | None",
        cache_controls: "CacheControlsHint",
        aspeech: bool,
    ) -> "HttpxBinaryResponseContent | Coroutine[object, object, HttpxBinaryResponseContent]":
        received_args: Final = locals()
        from litellm import acompletion, completion

        validated_kwargs: Final = self.validate_input_kwargs(received_args)
        model = validated_kwargs["model"]
        input = validated_kwargs["input"]
        optional_params = validated_kwargs["optional_params"]
        litellm_params = validated_kwargs["litellm_params"]
        headers = validated_kwargs["headers"]
        logging_obj = validated_kwargs["logging_obj"]
        custom_llm_provider = validated_kwargs["custom_llm_provider"]
        voice = validated_kwargs["voice"]

        request_data: Final = self.transformation_handler.transform_request(
            model=model,
            input=input,
            optional_params=optional_params,
            litellm_params={
                **litellm_params,
                "api_base": api_base,
                "extra_headers": _headers_as_text(extra_headers),
                "timeout": timeout,
                **cache_controls,
            },
            headers=headers,
            litellm_logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            voice=voice,
        )

        requested_response_format: Final = validated_kwargs["optional_params"].get("response_format")
        response_format: Final = requested_response_format if isinstance(requested_response_format, str) else None
        if aspeech:
            return self._aspeech(acompletion(**request_data), model, response_format, logging_obj)

        with internal_sub_call(logging_obj.litellm_call_id):
            result: Final = completion(
                **request_data,
            )
        return self._to_speech_response(result, model, response_format)

    async def _aspeech(
        self,
        pending_completion: Coroutine[object, object, object],
        model: str,
        response_format: str | None,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        with logging_obj.preserve_entrypoint_acompletion(), internal_sub_call(logging_obj.litellm_call_id):
            result: Final = await pending_completion
        return self._to_speech_response(result, model, response_format)

    def _to_speech_response(
        self, result: object, model: str, response_format: str | None
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.utils import ModelResponse

        if not isinstance(result, ModelResponse):
            raise Exception(f"Unmapped response type. Got type: {type(result)}")
        return self.transformation_handler.transform_response(
            model_response=result,
            model=model,
            response_format=response_format,
        )


speech_to_completion_bridge_handler = SpeechToCompletionBridgeHandler()
