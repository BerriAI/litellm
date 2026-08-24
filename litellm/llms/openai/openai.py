import time
import types
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, Optional, cast
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    from aiohttp import ClientSession

import openai
from openai import AsyncOpenAI, OpenAI
from openai.types.beta.assistant_deleted import AssistantDeleted
from openai.types.file_deleted import FileDeleted
from pydantic import BaseModel
from typing_extensions import overload

import litellm
from litellm import LlmProviders
from litellm._logging import verbose_logger
from litellm.constants import DEFAULT_MAX_RETRIES
from litellm.files.types import FileContentStreamingResult
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.logging_utils import speech_request_body, track_llm_api_timing
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.bedrock.chat.invoke_handler import MockResponseIterator
from litellm.types.utils import (
    EmbeddingResponse,
    ImageResponse,
    LiteLLMBatch,
    ModelResponse,
    ModelResponseStream,
)
from litellm.utils import (
    CustomStreamWrapper,
    ProviderConfigManager,
    convert_to_model_response_object,
)

from ...types.llms.openai import *
from ..base import BaseLLM
from .chat.gpt_5_transformation import OpenAIGPT5Config
from .chat.o_series_transformation import OpenAIOSeriesConfig
from .common_utils import (
    BaseOpenAILLM,
    OpenAIError,
    build_output_token_limit_response,
    drop_params_from_unprocessable_entity_error,
    is_output_token_limit_error,
)

openaiOSeriesConfig: Final = OpenAIOSeriesConfig()
openAIGPT5Config: Final = OpenAIGPT5Config()


class MistralEmbeddingConfig:
    """
    Reference: https://docs.mistral.ai/api/#operation/createEmbedding
    """

    def __init__(
        self,
    ) -> None:
        locals_: Final[Mapping[str, object]] = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        config_attrs: Final[Mapping[str, object]] = cls.__dict__
        return {
            k: v
            for k, v in config_attrs.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self):
        return [
            "encoding_format",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "encoding_format":
                optional_params["encoding_format"] = value
        return optional_params


class OpenAIConfig(BaseConfig):
    """
    Reference: https://platform.openai.com/docs/api-reference/chat/create

    The class `OpenAIConfig` provides configuration for the OpenAI's Chat API interface. Below are the parameters:

    - `frequency_penalty` (number or null): Defaults to 0. Allows a value between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, thereby minimizing repetition.

    - `function_call` (string or object): This optional parameter controls how the model calls functions.

    - `functions` (array): An optional parameter. It is a list of functions for which the model may generate JSON inputs.

    - `logit_bias` (map): This optional parameter modifies the likelihood of specified tokens appearing in the completion.

    - `max_tokens` (integer or null): This optional parameter helps to set the maximum number of tokens to generate in the chat completion. OpenAI has now deprecated in favor of max_completion_tokens, and is not compatible with o1 series models.

    - `max_completion_tokens` (integer or null): An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.

    - `n` (integer or null): This optional parameter helps to set how many chat completion choices to generate for each input message.

    - `presence_penalty` (number or null): Defaults to 0. It penalizes new tokens based on if they appear in the text so far, hence increasing the model's likelihood to talk about new topics.

    - `stop` (string / array / null): Specifies up to 4 sequences where the API will stop generating further tokens.

    - `temperature` (number or null): Defines the sampling temperature to use, varying between 0 and 2.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling.
    """

    frequency_penalty: int | None = None
    function_call: str | dict | None = None
    functions: list | None = None
    logit_bias: dict | None = None
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: int | None = None
    stop: str | list | None = None
    temperature: int | None = None
    top_p: int | None = None
    response_format: dict | None = None

    def __init__(
        self,
        frequency_penalty: int | None = None,
        function_call: str | dict | None = None,
        functions: list | None = None,
        logit_bias: dict | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: int | None = None,
        stop: str | list | None = None,
        temperature: int | None = None,
        top_p: int | None = None,
        response_format: dict | None = None,
    ) -> None:
        locals_: Final[Mapping[str, object]] = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        """
        This function returns the list
        of supported openai parameters for a given OpenAI Model

        - If O1 model, returns O1 supported params
        - If gpt-audio model, returns gpt-audio supported params
        - Else, returns gpt supported params

        Args:
            model (str): OpenAI model

        Returns:
            list: List of supported openai parameters
        """
        if openaiOSeriesConfig.is_model_o_series_model(model=model):
            return openaiOSeriesConfig.get_supported_openai_params(model=model)
        elif openAIGPT5Config.is_model_gpt_5_model(model=model):
            return openAIGPT5Config.get_supported_openai_params(model=model)
        elif litellm.openAIGPTAudioConfig.is_model_gpt_audio_model(model=model):
            return litellm.openAIGPTAudioConfig.get_supported_openai_params(model=model)
        else:
            return litellm.openAIGPTConfig.get_supported_openai_params(model=model)

    def _map_openai_params(self, non_default_params: dict, optional_params: dict, model: str) -> dict:
        supported_openai_params: Final = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def _transform_messages(self, messages: list[AllMessageValues], model: str) -> list[AllMessageValues]:
        return messages

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """ """
        if openaiOSeriesConfig.is_model_o_series_model(model=model):
            return openaiOSeriesConfig.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=drop_params,
            )
        elif openAIGPT5Config.is_model_gpt_5_model(model=model):
            return openAIGPT5Config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=drop_params,
            )
        elif litellm.openAIGPTAudioConfig.is_model_gpt_audio_model(model=model):
            return litellm.openAIGPTAudioConfig.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=drop_params,
            )

        return litellm.openAIGPTConfig.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        messages = self._transform_messages(messages=messages, model=model)
        return {"model": model, "messages": messages, **optional_params}

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        logging_obj.post_call(original_response=raw_response.text)
        logging_obj.model_call_details["response_headers"] = raw_response.headers
        final_response_obj: Final = cast(
            ModelResponse,
            convert_to_model_response_object(
                response_object=raw_response.json(),
                model_response_object=model_response,
                hidden_params={"headers": raw_response.headers},
                _response_headers=dict(raw_response.headers),
            ),
        )

        return final_response_obj

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
        return {
            "Authorization": f"Bearer {api_key}",
            **headers,
        }

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> "OpenAIChatCompletionResponseIterator":
        return OpenAIChatCompletionResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class OpenAIChatCompletionResponseIterator(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        {'choices': [{'delta': {'content': '', 'role': 'assistant'}, 'finish_reason': None, 'index': 0, 'logprobs': None}], 'created': 1735763082, 'id': 'a83a2b0fbfaf4aab9c2c93cb8ba346d7', 'model': 'mistral-large', 'object': 'chat.completion.chunk'}
        """
        try:
            return ModelResponseStream(**chunk)
        except Exception as e:
            raise e


class OpenAIChatCompletion(BaseLLM, BaseOpenAILLM):
    def __init__(self) -> None:
        super().__init__()

    def _set_dynamic_params_on_client(
        self,
        client: OpenAI | AsyncOpenAI,
        organization: str | None = None,
        max_retries: int | None = None,
    ):
        if organization is not None:
            client.organization = organization
        if max_retries is not None:
            client.max_retries = max_retries

    def _get_openai_client(
        self,
        is_async: bool,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        timeout: float | httpx.Timeout = httpx.Timeout(None),
        max_retries: int | None = DEFAULT_MAX_RETRIES,
        organization: str | None = None,
        client: OpenAI | AsyncOpenAI | None = None,
        shared_session: Optional["ClientSession"] = None,
    ) -> OpenAI | AsyncOpenAI | None:
        client_initialization_params: Final[dict] = locals()
        if client is None:
            if not isinstance(max_retries, int):
                raise OpenAIError(
                    status_code=422,
                    message=f"max retries must be an int. Passed in value: {max_retries}",
                )
            cached_client: Final = self.get_cached_openai_client(
                client_initialization_params=client_initialization_params,
                client_type="openai",
            )

            if cached_client:
                if isinstance(cached_client, OpenAI) or isinstance(cached_client, AsyncOpenAI):
                    return cached_client
            http_client: Final[httpx.Client | httpx.AsyncClient | None] = (
                OpenAIChatCompletion._get_async_http_client(shared_session=shared_session)
                if is_async
                else OpenAIChatCompletion._get_sync_http_client()
            )
            if is_async:
                _new_client: OpenAI | AsyncOpenAI = AsyncOpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=http_client,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                )
            else:
                _new_client = OpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=http_client,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                )

            ## SAVE CACHE KEY
            self.set_cached_openai_client(
                openai_client=_new_client,
                client_initialization_params=client_initialization_params,
                client_type="openai",
                litellm_owned_client=self.owns_wrapped_http_client(http_client),
            )
            return _new_client

        else:
            self._set_dynamic_params_on_client(
                client=client,
                organization=organization,
                max_retries=max_retries,
            )
            return client

    @track_llm_api_timing()
    async def make_openai_chat_completion_request(
        self,
        openai_aclient: AsyncOpenAI,
        data: dict,
        timeout: float | httpx.Timeout,
        logging_obj: LiteLLMLoggingObj,
    ) -> tuple[dict, BaseModel]:
        """
        Helper to:
        - call chat.completions.create.with_raw_response when litellm.return_response_headers is True
        - call chat.completions.create by default
        """
        start_time: Final = time.time()
        try:
            raw_response = await openai_aclient.chat.completions.with_raw_response.create(**data, timeout=timeout)
            end_time = time.time()

            if hasattr(raw_response, "headers"):
                headers = dict(raw_response.headers)
            else:
                headers = {}
            response: Final = raw_response.parse()
            if not data.get("stream") and not hasattr(response, "model_dump"):
                raise OpenAIError(
                    status_code=500,
                    message=f"Empty or invalid response from LLM endpoint. Received: {response!r}. Check the reverse proxy or model server configuration.",
                )
            return headers, response
        except openai.APITimeoutError as e:
            end_time = time.time()
            time_delta: Final = round(end_time - start_time, 2)
            e.message += f" - timeout value={timeout}, time taken={time_delta} seconds"
            raise e
        except openai.BadRequestError as e:
            if not is_output_token_limit_error(e):
                raise
            return build_output_token_limit_response(e=e, data=data, is_async=True)
        except Exception as e:
            raise e

    @track_llm_api_timing()
    def make_sync_openai_chat_completion_request(
        self,
        openai_client: OpenAI,
        data: dict,
        timeout: float | httpx.Timeout,
        logging_obj: LiteLLMLoggingObj,
    ) -> tuple[dict, BaseModel]:
        """
        Helper to:
        - call chat.completions.create.with_raw_response when litellm.return_response_headers is True
        - call chat.completions.create by default
        """
        raw_response = None
        try:
            raw_response = openai_client.chat.completions.with_raw_response.create(**data, timeout=timeout)

            if hasattr(raw_response, "headers"):
                headers = dict(raw_response.headers)
            else:
                headers = {}
            response: Final = raw_response.parse()
            if not data.get("stream") and not hasattr(response, "model_dump"):
                raise OpenAIError(
                    status_code=500,
                    message=f"Empty or invalid response from LLM endpoint. Received: {response!r}. Check the reverse proxy or model server configuration.",
                )
            return headers, response
        except OpenAIError:
            raise
        except openai.BadRequestError as e:
            if not is_output_token_limit_error(e):
                raise
            return build_output_token_limit_response(e=e, data=data, is_async=False)
        except Exception as e:
            if raw_response is not None:
                raise Exception(
                    f"error - {e}, Received response - {raw_response}, Type of response - {type(raw_response)}"
                )
            else:
                raise e

    async def _call_agentic_completion_hooks_openai(
        self,
        response: object,
        model: str,
        messages: list[dict],
        optional_params: dict,
        logging_obj: LiteLLMLoggingObj,
        stream: bool,
        litellm_params: dict,
    ) -> object | None:
        """
        Call agentic completion hooks for all custom loggers (OpenAI Chat Completions API).

        1. Call async_should_run_chat_completion_agentic_loop to check if agentic loop is needed
        2. If yes, call async_run_chat_completion_agentic_loop to execute the loop

        Returns the response from agentic loop, or None if no hook runs.
        """
        from litellm._logging import verbose_logger
        from litellm.integrations.custom_logger import CustomLogger

        callbacks: Final = litellm.callbacks + (logging_obj.dynamic_success_callbacks or [])
        # Avoid logging full callback objects to prevent leaking sensitive data
        verbose_logger.debug("LiteLLM.AgenticHooks: callbacks_count=%s", len(callbacks))
        tools: Final = optional_params.get("tools", [])
        # Avoid logging full tools payloads; they may contain sensitive parameters
        verbose_logger.debug(
            "LiteLLM.AgenticHooks: tools_count=%s",
            len(tools) if isinstance(tools, list) else 1 if tools else 0,
        )
        # Get custom_llm_provider from litellm_params
        custom_llm_provider: Final = litellm_params.get("custom_llm_provider", "openai")

        for callback in callbacks:
            try:
                if isinstance(callback, CustomLogger):
                    # Check if the callback has the chat completion agentic loop methods
                    if not hasattr(callback, "async_should_run_chat_completion_agentic_loop"):
                        continue

                    # First: Check if agentic loop should run (using chat completion method)
                    (
                        should_run,
                        tool_calls,
                    ) = await callback.async_should_run_chat_completion_agentic_loop(
                        response=response,
                        model=model,
                        messages=messages,
                        tools=tools,
                        stream=stream,
                        custom_llm_provider=custom_llm_provider,
                        kwargs=litellm_params,
                    )

                    if should_run:
                        # Second: Execute agentic loop
                        kwargs_with_provider = litellm_params.copy() if litellm_params else {}
                        kwargs_with_provider["custom_llm_provider"] = custom_llm_provider

                        # For OpenAI Chat Completions, use the chat completion agentic loop method
                        agentic_response: object = await callback.async_run_chat_completion_agentic_loop(
                            tools=tool_calls,
                            model=model,
                            messages=messages,
                            response=response,
                            optional_params=optional_params,
                            logging_obj=logging_obj,
                            stream=stream,
                            kwargs=kwargs_with_provider,
                        )
                        # First hook that runs agentic loop wins
                        return agentic_response

            except Exception as e:
                verbose_logger.exception(
                    "LiteLLM.AgenticHookError: Exception in agentic completion hooks for OpenAI: %s", e
                )

        return None

    def mock_streaming(
        self,
        response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        model: str,
        stream_options: dict | None = None,
    ) -> CustomStreamWrapper:
        completion_stream: Final = MockResponseIterator(model_response=response)
        streaming_response: Final = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="openai",
            logging_obj=logging_obj,
            stream_options=stream_options,
        )

        return streaming_response

    def completion(
        self,
        model_response: ModelResponse,
        timeout: float | httpx.Timeout,
        optional_params: dict,
        litellm_params: dict,
        logging_obj: LiteLLMLoggingObj,
        model: str | None = None,
        messages: list | None = None,
        print_verbose: Callable | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        dynamic_params: bool | None = None,
        azure_ad_token: str | None = None,
        acompletion: bool = False,
        logger_fn=None,
        headers: dict | None = None,
        custom_prompt_dict: dict = {},
        client=None,
        organization: str | None = None,
        custom_llm_provider: str | None = None,
        drop_params: bool | None = None,
        shared_session: Optional["ClientSession"] = None,
    ):
        super().completion(shared_session=shared_session)
        try:
            fake_stream: bool = False
            inference_params = optional_params.copy()
            stream_options: Final[dict | None] = inference_params.pop("stream_options", None)
            stream: Final[bool | None] = inference_params.pop("stream", False)
            provider_config: BaseConfig | None = None

            if custom_llm_provider is not None and model is not None:
                try:
                    provider_config = ProviderConfigManager.get_provider_chat_config(
                        model=model, provider=LlmProviders(custom_llm_provider)
                    )
                except ValueError:
                    # JSON-configured providers may not be in LlmProviders enum
                    provider_config = None

            if provider_config is None:
                provider_config = OpenAIConfig()

            if provider_config:
                fake_stream = provider_config.should_fake_stream(
                    model=model, custom_llm_provider=custom_llm_provider, stream=stream
                )

            if headers:
                inference_params["extra_headers"] = headers
            if model is None or messages is None:
                raise OpenAIError(status_code=422, message="Missing model or messages")

            if not isinstance(timeout, float) and not isinstance(timeout, httpx.Timeout):
                raise OpenAIError(
                    status_code=422,
                    message="Timeout needs to be a float or httpx.Timeout",
                )

            if custom_llm_provider is not None and custom_llm_provider != "openai":
                model_response.model = f"{custom_llm_provider}/{model}"

            for _ in range(2):  # if call fails due to alternating messages, retry with reformatted message
                try:
                    max_retries = inference_params.pop("max_retries", 2)
                    if acompletion is True:
                        if stream is True and fake_stream is False:
                            return self.async_streaming(
                                logging_obj=logging_obj,
                                headers=headers,
                                messages=messages,
                                optional_params=inference_params,
                                litellm_params=litellm_params,
                                provider_config=provider_config,
                                model=model,
                                api_base=api_base,
                                api_key=api_key,
                                api_version=api_version,
                                timeout=timeout,
                                client=client,
                                max_retries=max_retries,
                                organization=organization,
                                drop_params=drop_params,
                                stream_options=stream_options,
                                shared_session=shared_session,
                            )
                        else:
                            return self.acompletion(
                                messages=messages,
                                optional_params=inference_params,
                                litellm_params=litellm_params,
                                provider_config=provider_config,
                                headers=headers,
                                model=model,
                                logging_obj=logging_obj,
                                model_response=model_response,
                                api_base=api_base,
                                api_key=api_key,
                                api_version=api_version,
                                timeout=timeout,
                                client=client,
                                max_retries=max_retries,
                                organization=organization,
                                drop_params=drop_params,
                                fake_stream=fake_stream,
                                shared_session=shared_session,
                            )

                    data = provider_config.transform_request(
                        model=model,
                        messages=messages,
                        optional_params=inference_params,
                        litellm_params=litellm_params,
                        headers=headers or {},
                    )
                    if stream is True and fake_stream is False:
                        return self.streaming(
                            logging_obj=logging_obj,
                            headers=headers,
                            data=data,
                            model=model,
                            api_base=api_base,
                            api_key=api_key,
                            api_version=api_version,
                            timeout=timeout,
                            client=client,
                            max_retries=max_retries,
                            organization=organization,
                            stream_options=stream_options,
                        )
                    else:
                        if not isinstance(max_retries, int):
                            raise OpenAIError(status_code=422, message="max retries must be an int")
                        openai_client: OpenAI = self._get_openai_client(
                            is_async=False,
                            api_key=api_key,
                            api_base=api_base,
                            api_version=api_version,
                            timeout=timeout,
                            max_retries=max_retries,
                            organization=organization,
                            client=client,
                        )

                        ## LOGGING
                        logging_obj.pre_call(
                            input=messages,
                            api_key=openai_client.api_key,
                            additional_args={
                                "headers": headers,
                                "api_base": openai_client._base_url._uri_reference,
                                "acompletion": acompletion,
                                "complete_input_dict": data,
                            },
                        )

                        (
                            headers,
                            response,
                        ) = self.make_sync_openai_chat_completion_request(
                            openai_client=openai_client,
                            data=data,
                            timeout=timeout,
                            logging_obj=logging_obj,
                        )

                        logging_obj.model_call_details["response_headers"] = headers
                        stringified_response = provider_config.transform_parsed_response_dict(response.model_dump())
                        logging_obj.post_call(
                            input=messages,
                            api_key=api_key,
                            original_response=stringified_response,
                            additional_args={"complete_input_dict": data},
                        )

                        final_response_obj = convert_to_model_response_object(
                            response_object=stringified_response,
                            model_response_object=model_response,
                            _response_headers=headers,
                        )
                        if fake_stream is True:
                            return self.mock_streaming(
                                response=cast(ModelResponse, final_response_obj),
                                logging_obj=logging_obj,
                                model=model,
                                stream_options=stream_options,
                            )

                        return final_response_obj
                except openai.UnprocessableEntityError as e:
                    ## check if body contains unprocessable params - related issue https://github.com/BerriAI/litellm/issues/4800
                    if litellm.drop_params is True or drop_params is True:
                        inference_params = drop_params_from_unprocessable_entity_error(e, inference_params)
                    else:
                        raise e
                    # e.message
                except Exception as e:
                    if print_verbose is not None:
                        print_verbose(f"openai.py: Received openai error - {e}")
                    if (
                        "Conversation roles must alternate user/assistant" in str(e)
                        or "user and assistant roles should be alternating" in str(e)
                    ) and messages is not None:
                        if print_verbose is not None:
                            print_verbose("openai.py: REFORMATS THE MESSAGE!")
                        # reformat messages to ensure user/assistant are alternating, if there's either 2 consecutive 'user' messages or 2 consecutive 'assistant' message, add a blank 'user' or 'assistant' message to ensure compatibility
                        new_messages = []
                        for i in range(len(messages) - 1):
                            new_messages.append(messages[i])
                            if messages[i]["role"] == messages[i + 1]["role"]:
                                if messages[i]["role"] == "user":
                                    new_messages.append({"role": "assistant", "content": ""})
                                else:
                                    new_messages.append({"role": "user", "content": ""})
                        new_messages.append(messages[-1])
                        messages = new_messages
                    elif ("Last message must have role `user`" in str(e)) and messages is not None:
                        new_messages = messages
                        new_messages.append({"role": "user", "content": ""})
                        messages = new_messages
                    elif "unknown field: parameter index is not a valid field" in str(e):
                        litellm.remove_index_from_tool_calls(messages=messages)
                    else:
                        raise e
        except OpenAIError as e:
            raise e
        except Exception as e:
            status_code: Final = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text: Final = getattr(e, "text", str(e))
            error_response: Final = getattr(e, "response", None)
            error_body: Final = getattr(e, "body", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(
                status_code=status_code,
                message=error_text,
                headers=error_headers,
                body=error_body,
            )

    async def acompletion(
        self,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        provider_config: BaseConfig,
        model: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        organization: str | None = None,
        client=None,
        max_retries=None,
        headers=None,
        drop_params: bool | None = None,
        stream_options: dict | None = None,
        fake_stream: bool = False,
        shared_session: Optional["ClientSession"] = None,
    ):
        response = None
        data = await provider_config.async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers or {},
        )
        for _ in range(2):  # if call fails due to alternating messages, retry with reformatted message
            try:
                openai_aclient: AsyncOpenAI = self._get_openai_client(
                    is_async=True,
                    api_key=api_key,
                    api_base=api_base,
                    api_version=api_version,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                    client=client,
                    shared_session=shared_session,
                )

                ## LOGGING
                logging_obj.pre_call(
                    input=data["messages"],
                    api_key=openai_aclient.api_key,
                    additional_args={
                        "headers": {"Authorization": f"Bearer {openai_aclient.api_key}"},
                        "api_base": openai_aclient._base_url._uri_reference,
                        "acompletion": True,
                        "complete_input_dict": data,
                    },
                )

                headers, response = await self.make_openai_chat_completion_request(
                    openai_aclient=openai_aclient,
                    data=data,
                    timeout=timeout,
                    logging_obj=logging_obj,
                )
                stringified_response = provider_config.transform_parsed_response_dict(response.model_dump())
                logging_obj.post_call(
                    input=data["messages"],
                    api_key=api_key,
                    original_response=stringified_response,
                    additional_args={"complete_input_dict": data},
                )
                logging_obj.model_call_details["response_headers"] = headers
                final_response_obj = convert_to_model_response_object(
                    response_object=stringified_response,
                    model_response_object=model_response,
                    hidden_params={"headers": headers},
                    _response_headers=headers,
                )

                # Call agentic completion hooks (e.g., for websearch_interception)
                agentic_response = await self._call_agentic_completion_hooks_openai(
                    response=final_response_obj,
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                    logging_obj=logging_obj,
                    stream=False,
                    litellm_params=litellm_params,
                )

                if agentic_response is not None:
                    final_response_obj = agentic_response

                if fake_stream is True:
                    return self.mock_streaming(
                        response=cast(ModelResponse, final_response_obj),
                        logging_obj=logging_obj,
                        model=model,
                        stream_options=stream_options,
                    )

                return final_response_obj
            except openai.UnprocessableEntityError as e:
                ## check if body contains unprocessable params - related issue https://github.com/BerriAI/litellm/issues/4800
                if litellm.drop_params is True or drop_params is True:
                    data = drop_params_from_unprocessable_entity_error(e, data)
                else:
                    raise e
                # e.message
            except Exception as e:
                exception_response = getattr(e, "response", None)
                status_code = getattr(e, "status_code", 500)
                exception_body = getattr(e, "body", None)
                error_headers = getattr(e, "headers", None)
                if error_headers is None and exception_response:
                    error_headers = getattr(exception_response, "headers", None)
                message = getattr(e, "message", str(e))

                raise OpenAIError(
                    status_code=status_code,
                    message=message,
                    headers=error_headers,
                    body=exception_body,
                )

    def streaming(
        self,
        logging_obj,
        timeout: float | httpx.Timeout,
        data: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        organization: str | None = None,
        client=None,
        max_retries=None,
        headers=None,
        stream_options: dict | None = None,
    ):
        data["stream"] = True
        data.update(self.get_stream_options(stream_options=stream_options, api_base=api_base))

        openai_client: Final[OpenAI] = self._get_openai_client(
            is_async=False,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )
        ## LOGGING
        logging_obj.pre_call(
            input=data["messages"],
            api_key=api_key,
            additional_args={
                "headers": {"Authorization": f"Bearer {openai_client.api_key}"},
                "api_base": openai_client._base_url._uri_reference,
                "acompletion": False,
                "complete_input_dict": data,
            },
        )
        headers, response = self.make_sync_openai_chat_completion_request(
            openai_client=openai_client,
            data=data,
            timeout=timeout,
            logging_obj=logging_obj,
        )

        logging_obj.model_call_details["response_headers"] = headers
        streamwrapper: Final = CustomStreamWrapper(
            completion_stream=response,
            model=model,
            custom_llm_provider="openai",
            logging_obj=logging_obj,
            stream_options=data.get("stream_options", None),
            _response_headers=headers,
        )
        return streamwrapper

    async def async_streaming(
        self,
        timeout: float | httpx.Timeout,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        provider_config: BaseConfig,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        organization: str | None = None,
        client=None,
        max_retries=None,
        headers=None,
        drop_params: bool | None = None,
        stream_options: dict | None = None,
        shared_session: Optional["ClientSession"] = None,
    ):
        response = None
        data = provider_config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers or {},
        )
        data["stream"] = True
        data.update(self.get_stream_options(stream_options=stream_options, api_base=api_base))
        for _ in range(2):
            try:
                openai_aclient: AsyncOpenAI = self._get_openai_client(
                    is_async=True,
                    api_key=api_key,
                    api_base=api_base,
                    api_version=api_version,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                    client=client,
                    shared_session=shared_session,
                )
                ## LOGGING
                logging_obj.pre_call(
                    input=data["messages"],
                    api_key=api_key,
                    additional_args={
                        "headers": headers,
                        "api_base": api_base,
                        "acompletion": True,
                        "complete_input_dict": data,
                    },
                )

                headers, response = await self.make_openai_chat_completion_request(
                    openai_aclient=openai_aclient,
                    data=data,
                    timeout=timeout,
                    logging_obj=logging_obj,
                )
                logging_obj.model_call_details["response_headers"] = headers
                streamwrapper = CustomStreamWrapper(
                    completion_stream=response,
                    model=model,
                    custom_llm_provider="openai",
                    logging_obj=logging_obj,
                    stream_options=data.get("stream_options", None),
                    _response_headers=headers,
                )
                return streamwrapper
            except openai.UnprocessableEntityError as e:
                ## check if body contains unprocessable params - related issue https://github.com/BerriAI/litellm/issues/4800
                if litellm.drop_params is True or drop_params is True:
                    data = drop_params_from_unprocessable_entity_error(e, data)
                else:
                    raise e
            except (
                Exception
            ) as e:  # need to exception handle here. async exceptions don't get caught in sync functions.
                if isinstance(e, OpenAIError):
                    raise e

                error_headers = getattr(e, "headers", None)
                status_code = getattr(e, "status_code", 500)
                error_response = getattr(e, "response", None)
                exception_body = getattr(e, "body", None)
                if error_headers is None and error_response:
                    error_headers = getattr(error_response, "headers", None)
                if response is not None and hasattr(response, "text"):
                    raise OpenAIError(
                        status_code=status_code,
                        message=f"{e}\n\nOriginal Response: {response.text}",
                        headers=error_headers,
                        body=exception_body,
                    )
                else:
                    if type(e).__name__ == "ReadTimeout":
                        raise OpenAIError(
                            status_code=408,
                            message=f"{type(e).__name__}",
                            headers=error_headers,
                            body=exception_body,
                        )
                    elif hasattr(e, "status_code"):
                        raise OpenAIError(
                            status_code=getattr(e, "status_code", 500),
                            message=str(e),
                            headers=error_headers,
                            body=exception_body,
                        )
                    else:
                        raise OpenAIError(
                            status_code=500,
                            message=f"{e}",
                            headers=error_headers,
                            body=exception_body,
                        )

    def get_stream_options(self, stream_options: dict | None, api_base: str | None) -> dict:
        """
        Pass `stream_options` to the data dict for OpenAI requests
        """
        if stream_options is not None:
            return {"stream_options": stream_options}
        else:
            # by default litellm will include usage for openai endpoints
            if api_base is None or urlparse(api_base).hostname == "api.openai.com":
                return {"stream_options": {"include_usage": True}}
        return {}

    # Embedding
    @track_llm_api_timing()
    async def make_openai_embedding_request(
        self,
        openai_aclient: AsyncOpenAI,
        data: dict,
        timeout: float | httpx.Timeout,
        logging_obj: LiteLLMLoggingObj,
    ):
        """
        Helper to:
        - call embeddings.create.with_raw_response when litellm.return_response_headers is True
        - call embeddings.create by default
        """
        try:
            raw_response = await openai_aclient.embeddings.with_raw_response.create(**data, timeout=timeout)
            headers: Final = dict(raw_response.headers)
            response: Final = raw_response.parse()
            return headers, response
        except Exception as e:
            raise e

    @track_llm_api_timing()
    def make_sync_openai_embedding_request(
        self,
        openai_client: OpenAI,
        data: dict,
        timeout: float | httpx.Timeout,
        logging_obj: LiteLLMLoggingObj,
    ):
        """
        Helper to:
        - call embeddings.create.with_raw_response when litellm.return_response_headers is True
        - call embeddings.create by default
        """
        try:
            raw_response = openai_client.embeddings.with_raw_response.create(**data, timeout=timeout)

            headers: Final = dict(raw_response.headers)
            response: Final = raw_response.parse()
            return headers, response
        except Exception as e:
            raise e

    async def aembedding(
        self,
        input: list,
        data: dict,
        model_response: EmbeddingResponse,
        timeout: float,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        api_base: str | None = None,
        client: AsyncOpenAI | None = None,
        max_retries=None,
        shared_session: Optional["ClientSession"] = None,
    ):
        try:
            openai_aclient: Final[AsyncOpenAI] = self._get_openai_client(
                is_async=True,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                shared_session=shared_session,
            )
            headers, response = await self.make_openai_embedding_request(
                openai_aclient=openai_aclient,
                data=data,
                timeout=timeout,
                logging_obj=logging_obj,
            )
            logging_obj.model_call_details["response_headers"] = headers
            stringified_response: Final = response.model_dump()
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=stringified_response,
            )
            returned_response: Final[EmbeddingResponse] = convert_to_model_response_object(
                response_object=stringified_response,
                model_response_object=model_response,
                response_type="embedding",
                _response_headers=headers,
            )
            return returned_response
        except OpenAIError as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            raise e
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            status_code: Final = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text: Final = getattr(e, "text", str(e))
            error_response: Final = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(status_code=status_code, message=error_text, headers=error_headers)

    def embedding(
        self,
        model: str,
        input: list,
        timeout: float,
        logging_obj,
        model_response: EmbeddingResponse,
        optional_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
        client=None,
        aembedding=None,
        max_retries: int | None = None,
        shared_session: Optional["ClientSession"] = None,
    ) -> EmbeddingResponse:
        super().embedding()
        try:
            data: Final = {"model": model, "input": input, **optional_params}
            max_retries = max_retries or litellm.DEFAULT_MAX_RETRIES
            if not isinstance(max_retries, int):
                raise OpenAIError(status_code=422, message="max retries must be an int")
            ## LOGGING
            logging_obj.pre_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data, "api_base": api_base},
            )

            if aembedding is True:
                return self.aembedding(
                    data=data,
                    input=input,
                    logging_obj=logging_obj,
                    model_response=model_response,
                    api_base=api_base,
                    api_key=api_key,
                    timeout=timeout,
                    client=client,
                    max_retries=max_retries,
                    shared_session=shared_session,
                )

            openai_client: Final[OpenAI] = self._get_openai_client(
                is_async=False,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
            )

            ## embedding CALL
            headers: dict | None = None
            headers, sync_embedding_response = self.make_sync_openai_embedding_request(
                openai_client=openai_client,
                data=data,
                timeout=timeout,
                logging_obj=logging_obj,
            )

            ## LOGGING
            logging_obj.model_call_details["response_headers"] = headers
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=sync_embedding_response,
            )
            response: Final[EmbeddingResponse] = convert_to_model_response_object(
                response_object=sync_embedding_response.model_dump(),
                model_response_object=model_response,
                _response_headers=headers,
                response_type="embedding",
            )
            return response
        except OpenAIError as e:
            raise e
        except Exception as e:
            status_code: Final = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text: Final = getattr(e, "text", str(e))
            error_response: Final = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(status_code=status_code, message=error_text, headers=error_headers)

    async def aimage_generation(
        self,
        prompt: str,
        data: dict,
        model_response: ModelResponse,
        timeout: float,
        logging_obj: Any,
        api_key: str | None = None,
        api_base: str | None = None,
        client=None,
        max_retries=None,
        organization: str | None = None,
        headers: dict | None = None,
    ):
        response = None
        try:
            openai_aclient: Final = self._get_openai_client(
                is_async=True,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )

            logging_obj.pre_call(
                input=prompt,
                api_key=openai_aclient.api_key,
                additional_args={  # mutable-ok: loggers isinstance-check this payload as a dict
                    "headers": {"Authorization": f"Bearer {openai_aclient.api_key}"},  # mutable-ok: logged header map
                    "api_base": str(openai_aclient.base_url),
                    "acompletion": True,
                    "complete_input_dict": data,
                },
            )

            request_data: Final = (  # mutable-ok: the OpenAI SDK takes the request body as a dict
                {**data, "extra_headers": headers} if headers else data
            )
            response = await openai_aclient.images.generate(**request_data, timeout=timeout)
            stringified_response: Final = response.model_dump()
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=stringified_response,
            )
            return convert_to_model_response_object(
                response_object=stringified_response,
                model_response_object=model_response,
                response_type="image_generation",
            )
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                original_response=str(e),
            )
            raise e

    def image_generation(
        self,
        model: str | None,
        prompt: str,
        timeout: float,
        optional_params: dict,
        logging_obj: Any,
        api_key: str | None = None,
        api_base: str | None = None,
        model_response: ImageResponse | None = None,
        client=None,
        aimg_generation=None,
        organization: str | None = None,
        headers: dict | None = None,
    ) -> ImageResponse:
        data = {}
        try:
            data = {"model": model, "prompt": prompt, **optional_params}
            max_retries: Final = data.pop("max_retries", 2)
            if not isinstance(max_retries, int):
                raise OpenAIError(status_code=422, message="max retries must be an int")

            if aimg_generation is True:
                return self.aimage_generation(
                    data=data,
                    prompt=prompt,
                    logging_obj=logging_obj,
                    model_response=model_response,
                    api_base=api_base,
                    api_key=api_key,
                    timeout=timeout,
                    client=client,
                    max_retries=max_retries,
                    organization=organization,
                    headers=headers,
                )

            openai_client: Final[OpenAI] = self._get_openai_client(
                is_async=False,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )

            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=openai_client.api_key,
                additional_args={
                    "headers": {"Authorization": f"Bearer {openai_client.api_key}"},
                    "api_base": openai_client._base_url._uri_reference,
                    "acompletion": True,
                    "complete_input_dict": data,
                },
            )

            ## COMPLETION CALL
            request_data: Final = (  # mutable-ok: the OpenAI SDK takes the request body as a dict
                {**data, "extra_headers": headers} if headers else data
            )
            _response: Final = openai_client.images.generate(**request_data, timeout=timeout)

            response: Final = _response.model_dump()
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=response,
            )
            return convert_to_model_response_object(
                response_object=response,
                model_response_object=model_response,
                response_type="image_generation",
            )
        except OpenAIError as e:
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            raise e
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=prompt,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=str(e),
            )
            if hasattr(e, "status_code"):
                raise OpenAIError(status_code=getattr(e, "status_code", 500), message=str(e))
            else:
                raise OpenAIError(status_code=500, message=str(e))

    def audio_speech(
        self,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        api_key: str | None,
        api_base: str | None,
        organization: str | None,
        project: str | None,
        max_retries: int,
        timeout: float | httpx.Timeout,
        logging_obj: LiteLLMLoggingObj,
        aspeech: bool | None = None,
        client=None,
        shared_session: Optional["ClientSession"] = None,
    ) -> HttpxBinaryResponseContent:
        if aspeech is not None and aspeech is True:
            return self.async_audio_speech(
                model=model,
                input=input,
                voice=voice,
                optional_params=optional_params,
                api_key=api_key,
                api_base=api_base,
                organization=organization,
                project=project,
                max_retries=max_retries,
                timeout=timeout,
                logging_obj=logging_obj,
                client=client,
                shared_session=shared_session,
            )

        openai_client: Final = self._get_openai_client(
            is_async=False,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
            shared_session=shared_session,
        )

        sync_client: Final = cast(OpenAI, openai_client)
        logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={  # mutable-ok: loggers isinstance-check this payload as a dict
                "complete_input_dict": speech_request_body(model, voice, optional_params),
                "api_base": str(sync_client.base_url),
            },
        )

        response: Final = sync_client.audio.speech.create(
            model=model,
            voice=voice,
            input=input,
            **optional_params,
        )
        return HttpxBinaryResponseContent(response=response.response)

    async def async_audio_speech(
        self,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        api_key: str | None,
        api_base: str | None,
        organization: str | None,
        project: str | None,
        max_retries: int,
        timeout: float | httpx.Timeout,
        logging_obj: LiteLLMLoggingObj,
        client=None,
        shared_session: Optional["ClientSession"] = None,
    ) -> HttpxBinaryResponseContent:
        openai_client: Final = cast(
            AsyncOpenAI,
            self._get_openai_client(
                is_async=True,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                shared_session=shared_session,
            ),
        )

        logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={  # mutable-ok: loggers isinstance-check this payload as a dict
                "complete_input_dict": speech_request_body(model, voice, optional_params),
                "api_base": str(openai_client.base_url),
            },
        )

        response: Final = await openai_client.audio.speech.create(
            model=model,
            voice=voice,
            input=input,
            **optional_params,
        )

        return HttpxBinaryResponseContent(response=response.response)


class OpenAIFilesAPI(BaseLLM):
    """
    OpenAI methods to support for batches
    - create_file()
    - retrieve_file()
    - list_files()
    - delete_file()
    - file_content()
    - update_file()
    """

    def __init__(self) -> None:
        super().__init__()

    def get_openai_client(
        self,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | None = None,
        _is_async: bool = False,
    ) -> OpenAI | AsyncOpenAI | None:
        received_args: Final[Mapping[str, object]] = locals()
        openai_client: OpenAI | AsyncOpenAI | None = None
        if client is None:
            data: Final = {}
            for k, v in received_args.items():
                if k == "self" or k == "client" or k == "_is_async":
                    pass
                elif k == "api_base" and v is not None:
                    data["base_url"] = v
                elif v is not None:
                    data[k] = v
            if _is_async is True:
                openai_client = AsyncOpenAI(**data)
            else:
                openai_client = OpenAI(**data)
        else:
            openai_client = client

        return openai_client

    async def acreate_file(
        self,
        create_file_data: CreateFileRequest,
        openai_client: AsyncOpenAI,
    ) -> OpenAIFileObject:
        response: Final = await openai_client.files.create(**create_file_data)
        return OpenAIFileObject.model_validate(response.model_dump())

    def create_file(
        self,
        _is_async: bool,
        create_file_data: CreateFileRequest,
        api_base: str,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | None = None,
    ) -> OpenAIFileObject | Coroutine[None, None, OpenAIFileObject]:
        openai_client: Final[OpenAI | AsyncOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.acreate_file(create_file_data=create_file_data, openai_client=openai_client)
        response: Final = cast(OpenAI, openai_client).files.create(**create_file_data)
        return OpenAIFileObject.model_validate(response.model_dump())

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        openai_client: AsyncOpenAI,
    ) -> HttpxBinaryResponseContent:
        response: Final = await openai_client.files.content(**file_content_request)
        return HttpxBinaryResponseContent(response=response.response)

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: str,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | None = None,
    ) -> HttpxBinaryResponseContent | Coroutine[None, None, HttpxBinaryResponseContent]:
        openai_client: Final[OpenAI | AsyncOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.afile_content(
                file_content_request=file_content_request,
                openai_client=openai_client,
            )
        response: Final = cast(OpenAI, openai_client).files.content(**file_content_request)

        return HttpxBinaryResponseContent(response=response.response)

    async def afile_content_streaming(
        self,
        file_content_request: FileContentRequest,
        openai_client: AsyncOpenAI,
        chunk_size: int = 1024 * 1024,
    ) -> FileContentStreamingResult:
        response_cm: Final = openai_client.files.with_streaming_response.content(**file_content_request)
        response: Final = await response_cm.__aenter__()
        headers: Final = dict(response.headers)

        async def _stream() -> AsyncIterator[bytes]:
            exc: BaseException | None = None
            try:
                async for chunk in response.iter_bytes(chunk_size=chunk_size):
                    yield chunk
            except BaseException as e:
                exc = e
                raise
            finally:
                if exc is None:
                    await response_cm.__aexit__(None, None, None)
                else:
                    await response_cm.__aexit__(type(exc), exc, exc.__traceback__)

        return FileContentStreamingResult(stream_iterator=_stream(), headers=headers)

    def file_content_streaming(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: str,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        chunk_size: int = 1024 * 1024,
        client: OpenAI | AsyncOpenAI | None = None,
    ) -> FileContentStreamingResult:
        openai_client: Final[OpenAI | AsyncOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.afile_content_streaming(
                file_content_request=file_content_request,
                openai_client=openai_client,
                chunk_size=chunk_size,
            )

        response_cm: Final = cast(OpenAI, openai_client).files.with_streaming_response.content(**file_content_request)
        response: Final = response_cm.__enter__()
        headers: Final = dict(response.headers)

        def _stream() -> Iterator[bytes]:
            exc: BaseException | None = None
            try:
                yield from response.iter_bytes(chunk_size=chunk_size)
            except BaseException as e:
                exc = e
                raise
            finally:
                if exc is None:
                    response_cm.__exit__(None, None, None)
                else:
                    response_cm.__exit__(type(exc), exc, exc.__traceback__)

        return FileContentStreamingResult(stream_iterator=_stream(), headers=headers)

    async def aretrieve_file(
        self,
        file_id: str,
        openai_client: AsyncOpenAI,
    ) -> FileObject:
        response: Final = await openai_client.files.retrieve(file_id=file_id)
        return response

    def retrieve_file(
        self,
        _is_async: bool,
        file_id: str,
        api_base: str,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | None = None,
    ):
        openai_client: Final[OpenAI | AsyncOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.aretrieve_file(
                file_id=file_id,
                openai_client=openai_client,
            )
        response: Final = openai_client.files.retrieve(file_id=file_id)

        return response

    async def adelete_file(
        self,
        file_id: str,
        openai_client: AsyncOpenAI,
    ) -> FileDeleted:
        response: Final = await openai_client.files.delete(file_id=file_id)
        return response

    def delete_file(
        self,
        _is_async: bool,
        file_id: str,
        api_base: str,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | None = None,
    ):
        openai_client: Final[OpenAI | AsyncOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.adelete_file(
                file_id=file_id,
                openai_client=openai_client,
            )
        response: Final = openai_client.files.delete(file_id=file_id)

        return response

    async def alist_files(
        self,
        openai_client: AsyncOpenAI,
        purpose: str | None = None,
    ):
        if isinstance(purpose, str):
            response = await openai_client.files.list(purpose=purpose)
        else:
            response = await openai_client.files.list()
        return response

    def list_files(
        self,
        _is_async: bool,
        api_base: str,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        purpose: str | None = None,
        client: OpenAI | AsyncOpenAI | None = None,
    ):
        openai_client: Final[OpenAI | AsyncOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.alist_files(
                purpose=purpose,
                openai_client=openai_client,
            )

        if isinstance(purpose, str):
            response = openai_client.files.list(purpose=purpose)
        else:
            response = openai_client.files.list()

        return response


class OpenAIBatchesAPI(BaseLLM):
    """
    OpenAI methods to support for batches
    - create_batch()
    - retrieve_batch()
    - cancel_batch()
    - list_batch()
    """

    def __init__(self) -> None:
        super().__init__()

    def get_openai_client(
        self,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | None = None,
        _is_async: bool = False,
    ) -> OpenAI | AsyncOpenAI | None:
        received_args: Final[Mapping[str, object]] = locals()
        openai_client: OpenAI | AsyncOpenAI | None = None
        if client is None:
            data: Final = {}
            for k, v in received_args.items():
                if k == "self" or k == "client" or k == "_is_async":
                    pass
                elif k == "api_base" and v is not None:
                    data["base_url"] = v
                elif v is not None:
                    data[k] = v
            if _is_async is True:
                openai_client = AsyncOpenAI(**data)
            else:
                openai_client = OpenAI(**data)
        else:
            openai_client = client

        return openai_client

    async def acreate_batch(
        self,
        create_batch_data: CreateBatchRequest,
        openai_client: AsyncOpenAI,
    ) -> LiteLLMBatch:
        response: Final = await openai_client.batches.create(**create_batch_data)
        return LiteLLMBatch.model_validate(response.model_dump())

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | None = None,
    ) -> LiteLLMBatch | Coroutine[None, None, LiteLLMBatch]:
        openai_client: Final[OpenAI | AsyncOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.acreate_batch(create_batch_data=create_batch_data, openai_client=openai_client)
        response: Final = cast(OpenAI, openai_client).batches.create(**create_batch_data)

        return LiteLLMBatch.model_validate(response.model_dump())

    async def aretrieve_batch(
        self,
        retrieve_batch_data: RetrieveBatchRequest,
        openai_client: AsyncOpenAI,
    ) -> LiteLLMBatch:
        verbose_logger.debug("retrieving batch, args= %s", retrieve_batch_data)
        response: Final = await openai_client.batches.retrieve(**retrieve_batch_data)
        return LiteLLMBatch.model_validate(response.model_dump())

    def retrieve_batch(
        self,
        _is_async: bool,
        retrieve_batch_data: RetrieveBatchRequest,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | None = None,
    ):
        openai_client: Final[OpenAI | AsyncOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.aretrieve_batch(retrieve_batch_data=retrieve_batch_data, openai_client=openai_client)
        response: Final = cast(OpenAI, openai_client).batches.retrieve(**retrieve_batch_data)
        return LiteLLMBatch.model_validate(response.model_dump())

    async def acancel_batch(
        self,
        cancel_batch_data: CancelBatchRequest,
        openai_client: AsyncOpenAI,
    ) -> LiteLLMBatch:
        verbose_logger.debug("async cancelling batch, args= %s", cancel_batch_data)
        response: Final = await openai_client.batches.cancel(**cancel_batch_data)
        return LiteLLMBatch.model_validate(response.model_dump())

    def cancel_batch(
        self,
        _is_async: bool,
        cancel_batch_data: CancelBatchRequest,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | None = None,
    ):
        openai_client: Final[OpenAI | AsyncOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.acancel_batch(cancel_batch_data=cancel_batch_data, openai_client=openai_client)

        # At this point, openai_client is guaranteed to be a sync OpenAI client
        if not isinstance(openai_client, OpenAI):
            raise ValueError("OpenAI client is not an instance of OpenAI. Make sure you passed a sync OpenAI client.")
        response: Final = openai_client.batches.cancel(**cancel_batch_data)
        return LiteLLMBatch.model_validate(response.model_dump())

    async def alist_batches(
        self,
        openai_client: AsyncOpenAI,
        after: str | None = None,
        limit: int | None = None,
    ):
        verbose_logger.debug("listing batches, after= %s, limit= %s", after, limit)
        response: Final = await openai_client.batches.list(after=after, limit=limit)
        return response

    def list_batches(
        self,
        _is_async: bool,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        after: str | None = None,
        limit: int | None = None,
        client: OpenAI | None = None,
    ):
        openai_client: Final[OpenAI | AsyncOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.alist_batches(openai_client=openai_client, after=after, limit=limit)
        response: Final = openai_client.batches.list(after=after, limit=limit)
        return response


class OpenAIAssistantsAPI(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def get_openai_client(
        self,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | None = None,
    ) -> OpenAI:
        received_args: Final[Mapping[str, object]] = locals()
        if client is None:
            data: Final = {}
            for k, v in received_args.items():
                if k == "self" or k == "client":
                    pass
                elif k == "api_base" and v is not None:
                    data["base_url"] = v
                elif v is not None:
                    data[k] = v
            openai_client = OpenAI(**data)
        else:
            openai_client = client

        return openai_client

    def async_get_openai_client(
        self,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None = None,
    ) -> AsyncOpenAI:
        received_args: Final[Mapping[str, object]] = locals()
        if client is None:
            data: Final = {}
            for k, v in received_args.items():
                if k == "self" or k == "client":
                    pass
                elif k == "api_base" and v is not None:
                    data["base_url"] = v
                elif v is not None:
                    data[k] = v
            openai_client = AsyncOpenAI(**data)
        else:
            openai_client = client

        return openai_client

    ### ASSISTANTS ###

    async def async_get_assistants(
        self,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None,
        order: str | None = "desc",
        limit: int | None = 20,
        before: str | None = None,
        after: str | None = None,
    ) -> AsyncCursorPage[Assistant]:
        openai_client: Final = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )
        request_params: Final = {
            "order": order,
            "limit": limit,
        }
        if before:
            request_params["before"] = before
        if after:
            request_params["after"] = after

        response: Final = await openai_client.beta.assistants.list(**request_params)

        return response

    # fmt: off

    @overload
    def get_assistants(
        self, 
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None,
        aget_assistants: Literal[True], 
    ) -> Coroutine[None, None, AsyncCursorPage[Assistant]]:
        ...

    @overload
    def get_assistants(
        self, 
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | None,
        aget_assistants: Literal[False] | None, 
    ) -> SyncCursorPage[Assistant]: 
        ...

    # fmt: on

    def get_assistants(
        self,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client=None,
        aget_assistants=None,
        order: str | None = "desc",
        limit: int | None = 20,
        before: str | None = None,
        after: str | None = None,
    ):
        if aget_assistants is not None and aget_assistants is True:
            return self.async_get_assistants(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )
        openai_client: Final = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        request_params: Final = {
            "order": order,
            "limit": limit,
        }

        if before:
            request_params["before"] = before
        if after:
            request_params["after"] = after

        response: Final = openai_client.beta.assistants.list(**request_params)

        return response

    # Create Assistant
    async def async_create_assistants(
        self,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None,
        create_assistant_data: dict,
    ) -> Assistant:
        openai_client: Final = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response: Final = await openai_client.beta.assistants.create(**create_assistant_data)

        return response

    def create_assistants(
        self,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        create_assistant_data: dict,
        client=None,
        async_create_assistants=None,
    ):
        if async_create_assistants is not None and async_create_assistants is True:
            return self.async_create_assistants(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
                create_assistant_data=create_assistant_data,
            )
        openai_client: Final = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response: Final = openai_client.beta.assistants.create(**create_assistant_data)
        return response

    # Delete Assistant
    async def async_delete_assistant(
        self,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None,
        assistant_id: str,
    ) -> AssistantDeleted:
        openai_client: Final = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response: Final = await openai_client.beta.assistants.delete(assistant_id=assistant_id)

        return response

    def delete_assistant(
        self,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        assistant_id: str,
        client=None,
        async_delete_assistants=None,
    ):
        if async_delete_assistants is not None and async_delete_assistants is True:
            return self.async_delete_assistant(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
                assistant_id=assistant_id,
            )
        openai_client: Final = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response: Final = openai_client.beta.assistants.delete(assistant_id=assistant_id)
        return response

    ### MESSAGES ###

    async def a_add_message(
        self,
        thread_id: str,
        message_data: dict,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None = None,
    ) -> OpenAIMessage:
        openai_client: Final = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        thread_message: Final[OpenAIMessage] = await openai_client.beta.threads.messages.create(
            thread_id,
            **message_data,
        )

        response_obj: OpenAIMessage | None = None
        if getattr(thread_message, "status", None) is None:
            thread_message.status = "completed"
            response_obj = OpenAIMessage.model_validate(thread_message.dict())
        else:
            response_obj = OpenAIMessage.model_validate(thread_message.dict())
        return response_obj

    # fmt: off

    @overload
    def add_message(
        self, 
        thread_id: str,
        message_data: dict,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None,
        a_add_message: Literal[True], 
    ) -> Coroutine[None, None, OpenAIMessage]:
        ...

    @overload
    def add_message(
        self, 
        thread_id: str,
        message_data: dict,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | None,
        a_add_message: Literal[False] | None, 
    ) -> OpenAIMessage: 
        ...

    # fmt: on

    def add_message(
        self,
        thread_id: str,
        message_data: dict,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client=None,
        a_add_message: bool | None = None,
    ):
        if a_add_message is not None and a_add_message is True:
            return self.a_add_message(
                thread_id=thread_id,
                message_data=message_data,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )
        openai_client: Final = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        thread_message: Final[OpenAIMessage] = openai_client.beta.threads.messages.create(
            thread_id,
            **message_data,
        )

        response_obj: OpenAIMessage | None = None
        if getattr(thread_message, "status", None) is None:
            thread_message.status = "completed"
            response_obj = OpenAIMessage.model_validate(thread_message.dict())
        else:
            response_obj = OpenAIMessage.model_validate(thread_message.dict())
        return response_obj

    async def async_get_messages(
        self,
        thread_id: str,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None = None,
    ) -> AsyncCursorPage[OpenAIMessage]:
        openai_client: Final = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response: Final = await openai_client.beta.threads.messages.list(thread_id=thread_id)

        return response

    # fmt: off

    @overload
    def get_messages(
        self, 
        thread_id: str,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None,
        aget_messages: Literal[True], 
    ) -> Coroutine[None, None, AsyncCursorPage[OpenAIMessage]]:
        ...

    @overload
    def get_messages(
        self, 
        thread_id: str,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | None,
        aget_messages: Literal[False] | None, 
    ) -> SyncCursorPage[OpenAIMessage]: 
        ...

    # fmt: on

    def get_messages(
        self,
        thread_id: str,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client=None,
        aget_messages=None,
    ):
        if aget_messages is not None and aget_messages is True:
            return self.async_get_messages(
                thread_id=thread_id,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )
        openai_client: Final = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response: Final = openai_client.beta.threads.messages.list(thread_id=thread_id)

        return response

    ### THREADS ###

    async def async_create_thread(
        self,
        metadata: dict | None,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None,
        messages: Iterable[OpenAICreateThreadParamsMessage] | None,
    ) -> Thread:
        openai_client: Final = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        data: Final = {}
        if messages is not None:
            data["messages"] = messages
        if metadata is not None:
            data["metadata"] = metadata

        message_thread: Final = await openai_client.beta.threads.create(**data)

        return Thread(**message_thread.dict())

    # fmt: off

    @overload
    def create_thread(
        self, 
        metadata: dict | None,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        messages: Iterable[OpenAICreateThreadParamsMessage] | None,
        client: AsyncOpenAI | None,
        acreate_thread: Literal[True], 
    ) -> Coroutine[None, None, Thread]:
        ...

    @overload
    def create_thread(
        self, 
        metadata: dict | None,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        messages: Iterable[OpenAICreateThreadParamsMessage] | None,
        client: OpenAI | None,
        acreate_thread: Literal[False] | None, 
    ) -> Thread: 
        ...

    # fmt: on

    def create_thread(
        self,
        metadata: dict | None,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        messages: Iterable[OpenAICreateThreadParamsMessage] | None,
        client=None,
        acreate_thread=None,
    ):
        """
        Here's an example:
        ```
        from litellm.llms.openai.openai import OpenAIAssistantsAPI, MessageData

        # create thread
        message: MessageData = {"role": "user", "content": "Hey, how's it going?"}
        openai_api.create_thread(messages=[message])
        ```
        """
        if acreate_thread is not None and acreate_thread is True:
            return self.async_create_thread(
                metadata=metadata,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
                messages=messages,
            )
        openai_client: Final = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        data: Final = {}
        if messages is not None:
            data["messages"] = messages
        if metadata is not None:
            data["metadata"] = metadata

        message_thread: Final = openai_client.beta.threads.create(**data)

        return Thread(**message_thread.dict())

    async def async_get_thread(
        self,
        thread_id: str,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None,
    ) -> Thread:
        openai_client: Final = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response: Final = await openai_client.beta.threads.retrieve(thread_id=thread_id)

        return Thread(**response.dict())

    # fmt: off

    @overload
    def get_thread(
        self, 
        thread_id: str,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None,
        aget_thread: Literal[True], 
    ) -> Coroutine[None, None, Thread]:
        ...

    @overload
    def get_thread(
        self, 
        thread_id: str,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | None,
        aget_thread: Literal[False] | None, 
    ) -> Thread: 
        ...

    # fmt: on

    def get_thread(
        self,
        thread_id: str,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client=None,
        aget_thread=None,
    ):
        if aget_thread is not None and aget_thread is True:
            return self.async_get_thread(
                thread_id=thread_id,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )
        openai_client: Final = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response: Final = openai_client.beta.threads.retrieve(thread_id=thread_id)

        return Thread(**response.dict())

    def delete_thread(self):
        pass

    ### RUNS ###

    async def arun_thread(
        self,
        thread_id: str,
        assistant_id: str,
        additional_instructions: str | None,
        instructions: str | None,
        metadata: dict[str, str] | None,
        model: str | None,
        stream: bool | None,
        tools: Iterable[AssistantToolParam] | None,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: AsyncOpenAI | None,
    ) -> Run:
        openai_client: Final = self.async_get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        response: Final = await openai_client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions,
            instructions=instructions,
            metadata=metadata,
            model=model,
            tools=tools,
        )

        return response

    def async_run_thread_stream(
        self,
        client: AsyncOpenAI,
        thread_id: str,
        assistant_id: str,
        additional_instructions: str | None,
        instructions: str | None,
        metadata: dict | None,
        model: str | None,
        tools: Iterable[AssistantToolParam] | None,
        event_handler: AssistantEventHandler | None,
    ) -> AsyncAssistantStreamManager[AsyncAssistantEventHandler]:
        data: Final[dict[str, Any]] = {
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "additional_instructions": additional_instructions,
            "instructions": instructions,
            "metadata": metadata,
            "model": model,
            "tools": tools,
        }
        if event_handler is not None:
            data["event_handler"] = event_handler
        return client.beta.threads.runs.stream(**data)

    def run_thread_stream(
        self,
        client: OpenAI,
        thread_id: str,
        assistant_id: str,
        additional_instructions: str | None,
        instructions: str | None,
        metadata: dict[str, str] | None,
        model: str | None,
        tools: Iterable[AssistantToolParam] | None,
        event_handler: AssistantEventHandler | None,
    ) -> AssistantStreamManager[AssistantEventHandler]:
        runs_stream: Final = client.beta.threads.runs.stream
        if event_handler is not None:
            return runs_stream(
                thread_id=thread_id,
                assistant_id=assistant_id,
                additional_instructions=additional_instructions,
                instructions=instructions,
                metadata=metadata,
                model=model,
                tools=tools,
                event_handler=event_handler,
            )
        return runs_stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions,
            instructions=instructions,
            metadata=metadata,
            model=model,
            tools=tools,
        )

    # fmt: off

    @overload
    def run_thread(
        self, 
        thread_id: str,
        assistant_id: str,
        additional_instructions: str | None,
        instructions: str | None,
        metadata: dict | None,
        model: str | None,
        stream: bool | None,
        tools: Iterable[AssistantToolParam] | None,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client,
        arun_thread: Literal[True], 
        event_handler: AssistantEventHandler | None,
    ) -> Coroutine[None, None, Run]:
        ...

    @overload
    def run_thread(
        self, 
        thread_id: str,
        assistant_id: str,
        additional_instructions: str | None,
        instructions: str | None,
        metadata: dict | None,
        model: str | None,
        stream: bool | None,
        tools: Iterable[AssistantToolParam] | None,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client,
        arun_thread: Literal[False] | None, 
        event_handler: AssistantEventHandler | None,
    ) -> Run: 
        ...

    # fmt: on

    def run_thread(
        self,
        thread_id: str,
        assistant_id: str,
        additional_instructions: str | None,
        instructions: str | None,
        metadata: dict[str, str] | None,
        model: str | None,
        stream: bool | None,
        tools: Iterable[AssistantToolParam] | None,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client=None,
        arun_thread=None,
        event_handler: AssistantEventHandler | None = None,
    ):
        if arun_thread is not None and arun_thread is True:
            if stream is not None and stream is True:
                _client: Final = self.async_get_openai_client(
                    api_key=api_key,
                    api_base=api_base,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                    client=client,
                )
                return self.async_run_thread_stream(
                    client=_client,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    additional_instructions=additional_instructions,
                    instructions=instructions,
                    metadata=metadata,
                    model=model,
                    tools=tools,
                    event_handler=event_handler,
                )
            return self.arun_thread(
                thread_id=thread_id,
                assistant_id=assistant_id,
                additional_instructions=additional_instructions,
                instructions=instructions,
                metadata=metadata,
                model=model,
                stream=stream,
                tools=tools,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
            )
        openai_client: Final = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
        )

        if stream is not None and stream is True:
            return self.run_thread_stream(
                client=openai_client,
                thread_id=thread_id,
                assistant_id=assistant_id,
                additional_instructions=additional_instructions,
                instructions=instructions,
                metadata=metadata,
                model=model,
                tools=tools,
                event_handler=event_handler,
            )

        response: Final = openai_client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
            additional_instructions=additional_instructions,
            instructions=instructions,
            metadata=metadata,
            model=model,
            tools=tools,
        )

        return response
