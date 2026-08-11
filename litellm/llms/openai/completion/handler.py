import json
from collections.abc import Callable
from typing import Final

from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base import BaseLLM
from litellm.types.llms.openai import AllMessageValues, OpenAITextCompletionUserMessage
from litellm.types.utils import LlmProviders, ModelResponse, TextCompletionResponse
from litellm.utils import ProviderConfigManager

from ..common_utils import BaseOpenAILLM, OpenAIError
from .transformation import OpenAITextCompletionConfig


class OpenAITextCompletion(BaseLLM):
    openai_text_completion_global_config = OpenAITextCompletionConfig()

    def __init__(self) -> None:
        super().__init__()

    def validate_environment(self, api_key):
        headers: Final = {
            "content-type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def completion(
        self,
        model_response: ModelResponse,
        api_key: str,
        model: str,
        messages: list[AllMessageValues] | list[OpenAITextCompletionUserMessage],
        timeout: float,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        print_verbose: Callable | None = None,
        api_base: str | None = None,
        acompletion: bool = False,
        litellm_params=None,
        logger_fn=None,
        client=None,
        organization: str | None = None,
        headers: dict | None = None,
    ):
        try:
            if headers:
                optional_params = {**optional_params, "extra_headers": headers}
            if headers is None:
                headers = self.validate_environment(api_key=api_key)
            if model is None or messages is None:
                raise OpenAIError(status_code=422, message="Missing model or messages")

            # don't send max retries to the api, if set

            provider_config: Final = ProviderConfigManager.get_provider_text_completion_config(
                model=model,
                provider=LlmProviders(custom_llm_provider),
            )

            data: Final = provider_config.transform_text_completion_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                headers=headers,
            )
            max_retries: Final = data.pop("max_retries", 2)
            ## LOGGING
            logging_obj.pre_call(
                input=messages,
                api_key=api_key,
                additional_args={
                    "headers": headers,
                    "api_base": api_base,
                    "complete_input_dict": data,
                },
            )
            if acompletion is True:
                if optional_params.get("stream", False):
                    return self.async_streaming(
                        logging_obj=logging_obj,
                        api_base=api_base,
                        api_key=api_key,
                        data=data,
                        headers=headers,
                        model_response=model_response,
                        model=model,
                        timeout=timeout,
                        max_retries=max_retries,
                        client=client,
                        organization=organization,
                    )
                else:
                    return self.acompletion(
                        api_base=api_base,
                        data=data,
                        headers=headers,
                        model_response=model_response,
                        api_key=api_key,
                        logging_obj=logging_obj,
                        model=model,
                        timeout=timeout,
                        max_retries=max_retries,
                        organization=organization,
                        client=client,
                    )
            elif optional_params.get("stream", False):
                return self.streaming(
                    logging_obj=logging_obj,
                    api_base=api_base,
                    api_key=api_key,
                    data=data,
                    headers=headers,
                    model_response=model_response,
                    model=model,
                    timeout=timeout,
                    max_retries=max_retries,
                    client=client,
                    organization=organization,
                )
            else:
                if client is None:
                    openai_client = OpenAI(
                        api_key=api_key,
                        base_url=api_base,
                        http_client=litellm.client_session,
                        timeout=timeout,
                        max_retries=max_retries,
                        organization=organization,
                    )
                else:
                    openai_client = client

                raw_response: Final = openai_client.completions.with_raw_response.create(**data)
                response: Final = raw_response.parse()
                response_json: Final = response.model_dump()

                ## LOGGING
                logging_obj.post_call(
                    api_key=api_key,
                    original_response=response_json,
                    additional_args={
                        "headers": headers,
                        "api_base": api_base,
                    },
                )

                ## RESPONSE OBJECT
                return TextCompletionResponse(**response_json)
        except Exception as e:
            status_code: Final = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text: Final = getattr(e, "text", str(e))
            error_response: Final = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(status_code=status_code, message=error_text, headers=error_headers)

    async def acompletion(
        self,
        logging_obj,
        api_base: str,
        data: dict,
        headers: dict,
        model_response: ModelResponse,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
        organization: str | None = None,
        client=None,
    ):
        try:
            if client is None:
                openai_aclient = AsyncOpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=BaseOpenAILLM._get_async_http_client(),
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                )
            else:
                openai_aclient = client

            raw_response: Final = await openai_aclient.completions.with_raw_response.create(**data)
            response: Final = raw_response.parse()
            response_json: Final = response.model_dump()

            ## LOGGING
            logging_obj.post_call(
                api_key=api_key,
                original_response=response,
                additional_args={
                    "headers": headers,
                    "api_base": api_base,
                },
            )
            ## RESPONSE OBJECT
            response_obj: Final = TextCompletionResponse(**response_json)
            response_obj._hidden_params.original_response = json.dumps(response_json)
            return response_obj
        except Exception as e:
            status_code: Final = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text: Final = getattr(e, "text", str(e))
            error_response: Final = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(status_code=status_code, message=error_text, headers=error_headers)

    def streaming(
        self,
        logging_obj: LiteLLMLoggingObj,
        api_key: str,
        data: dict,
        headers: dict,
        model_response: ModelResponse,
        model: str,
        timeout: float,
        api_base: str | None = None,
        max_retries=None,
        client=None,
        organization=None,
    ):
        if client is None:
            openai_client = OpenAI(
                api_key=api_key,
                base_url=api_base,
                http_client=litellm.client_session,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
            )
        else:
            openai_client = client

        try:
            raw_response: Final = openai_client.completions.with_raw_response.create(**data)
            response: Final = raw_response.parse()
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(status_code=status_code, message=error_text, headers=error_headers)
        streamwrapper: Final = CustomStreamWrapper(
            completion_stream=response,
            model=model,
            custom_llm_provider="text-completion-openai",
            logging_obj=logging_obj,
            stream_options=data.get("stream_options", None),
        )

        try:
            for chunk in streamwrapper:
                yield chunk
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(status_code=status_code, message=error_text, headers=error_headers)

    async def async_streaming(
        self,
        logging_obj: LiteLLMLoggingObj,
        api_key: str,
        data: dict,
        headers: dict,
        model_response: ModelResponse,
        model: str,
        timeout: float,
        max_retries: int,
        api_base: str | None = None,
        client=None,
        organization=None,
    ):
        if client is None:
            openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                http_client=litellm.aclient_session,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
            )
        else:
            openai_client = client

        raw_response: Final = await openai_client.completions.with_raw_response.create(**data)
        response: Final = raw_response.parse()
        streamwrapper: Final = CustomStreamWrapper(
            completion_stream=response,
            model=model,
            custom_llm_provider="text-completion-openai",
            logging_obj=logging_obj,
            stream_options=data.get("stream_options", None),
        )

        try:
            async for transformed_chunk in streamwrapper:
                yield transformed_chunk
        except Exception as e:
            status_code: Final = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text: Final = getattr(e, "text", str(e))
            error_response: Final = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(status_code=status_code, message=error_text, headers=error_headers)
