import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import httpx  # type: ignore

import litellm
import litellm.litellm_core_utils
import litellm.types
import litellm.types.utils
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.base_llm.files.transformation import BaseFilesConfig
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    MockResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import (
    CreateFileRequest,
    OpenAIFileObject,
    ResponseInputParam,
    ResponsesAPIResponse,
)
from litellm.types.rerank import OptionalRerankParams, RerankResponse
from litellm.types.responses.main import DeleteResponseResult
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import EmbeddingResponse, FileTypes, TranscriptionResponse
from litellm.utils import (
    CustomStreamWrapper,
    ImageResponse,
    ModelResponse,
    ProviderConfigManager,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseLLMHTTPHandler:
    async def _make_common_async_call(
        self,
        async_httpx_client: AsyncHTTPHandler,
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        logging_obj: LiteLLMLoggingObj,
        stream: bool = False,
        signed_json_body: Optional[bytes] = None,
    ) -> httpx.Response:
        """Common implementation across stream + non-stream calls. Meant to ensure consistent error-handling."""
        max_retry_on_unprocessable_entity_error = (
            provider_config.max_retry_on_unprocessable_entity_error
        )

        response: Optional[httpx.Response] = None
        for i in range(max(max_retry_on_unprocessable_entity_error, 1)):
            try:
                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    data=(
                        signed_json_body
                        if signed_json_body is not None
                        else json.dumps(data)
                    ),
                    timeout=timeout,
                    stream=stream,
                    logging_obj=logging_obj,
                )
            except httpx.HTTPStatusError as e:
                hit_max_retry = i + 1 == max_retry_on_unprocessable_entity_error
                should_retry = provider_config.should_retry_llm_api_inside_llm_translation_on_http_error(
                    e=e, litellm_params=litellm_params
                )
                if should_retry and not hit_max_retry:
                    data = (
                        provider_config.transform_request_on_unprocessable_entity_error(
                            e=e, request_data=data
                        )
                    )
                    continue
                else:
                    raise self._handle_error(e=e, provider_config=provider_config)
            except Exception as e:
                raise self._handle_error(e=e, provider_config=provider_config)
            break

        if response is None:
            raise provider_config.get_error_class(
                error_message="No response from the API",
                status_code=422,  # don't retry on this error
                headers={},
            )

        return response

    def _make_common_sync_call(
        self,
        sync_httpx_client: HTTPHandler,
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        logging_obj: LiteLLMLoggingObj,
        stream: bool = False,
        signed_json_body: Optional[bytes] = None,
    ) -> httpx.Response:
        max_retry_on_unprocessable_entity_error = (
            provider_config.max_retry_on_unprocessable_entity_error
        )

        response: Optional[httpx.Response] = None

        for i in range(max(max_retry_on_unprocessable_entity_error, 1)):
            try:
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    data=(
                        signed_json_body
                        if signed_json_body is not None
                        else json.dumps(data)
                    ),
                    timeout=timeout,
                    stream=stream,
                    logging_obj=logging_obj,
                )
            except httpx.HTTPStatusError as e:
                hit_max_retry = i + 1 == max_retry_on_unprocessable_entity_error
                should_retry = provider_config.should_retry_llm_api_inside_llm_translation_on_http_error(
                    e=e, litellm_params=litellm_params
                )
                if should_retry and not hit_max_retry:
                    data = (
                        provider_config.transform_request_on_unprocessable_entity_error(
                            e=e, request_data=data
                        )
                    )
                    continue
                else:
                    raise self._handle_error(e=e, provider_config=provider_config)
            except Exception as e:
                raise self._handle_error(e=e, provider_config=provider_config)
            break

        if response is None:
            raise provider_config.get_error_class(
                error_message="No response from the API",
                status_code=422,  # don't retry on this error
                headers={},
            )

        return response

    async def async_completion(
        self,
        custom_llm_provider: str,
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: dict,
        timeout: Union[float, httpx.Timeout],
        model: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        client: Optional[AsyncHTTPHandler] = None,
        json_mode: bool = False,
        signed_json_body: Optional[bytes] = None,
    ):
        if client is None:
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        response = await self._make_common_async_call(
            async_httpx_client=async_httpx_client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            data=data,
            timeout=timeout,
            litellm_params=litellm_params,
            stream=False,
            logging_obj=logging_obj,
            signed_json_body=signed_json_body,
        )
        return provider_config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            json_mode=json_mode,
        )

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_llm_provider: str,
        model_response: ModelResponse,
        encoding,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        acompletion: bool,
        stream: Optional[bool] = False,
        fake_stream: bool = False,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        provider_config: Optional[BaseConfig] = None,
    ):
        json_mode: bool = optional_params.pop("json_mode", False)
        extra_body: Optional[dict] = optional_params.pop("extra_body", None)

        provider_config = (
            provider_config
            or ProviderConfigManager.get_provider_chat_config(
                model=model, provider=litellm.LlmProviders(custom_llm_provider)
            )
        )
        if provider_config is None:
            raise ValueError(
                f"Provider config not found for model: {model} and provider: {custom_llm_provider}"
            )

        fake_stream = (
            fake_stream
            or optional_params.pop("fake_stream", False)
            or provider_config.should_fake_stream(
                model=model, custom_llm_provider=custom_llm_provider, stream=stream
            )
        )

        # get config from model, custom llm provider
        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers or {},
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_base=api_base,
            litellm_params=litellm_params,
        )

        api_base = provider_config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            stream=stream,
            litellm_params=litellm_params,
        )

        data = provider_config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        if extra_body is not None:
            data = {**data, **extra_body}

        headers, signed_json_body = provider_config.sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data=data,
            api_base=api_base,
            stream=stream,
            fake_stream=fake_stream,
            model=model,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        if acompletion is True:
            if stream is True:
                data = self._add_stream_param_to_request_body(
                    data=data,
                    provider_config=provider_config,
                    fake_stream=fake_stream,
                )
                return self.acompletion_stream_function(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    headers=headers,
                    custom_llm_provider=custom_llm_provider,
                    provider_config=provider_config,
                    timeout=timeout,
                    logging_obj=logging_obj,
                    data=data,
                    fake_stream=fake_stream,
                    client=(
                        client
                        if client is not None and isinstance(client, AsyncHTTPHandler)
                        else None
                    ),
                    litellm_params=litellm_params,
                    json_mode=json_mode,
                    optional_params=optional_params,
                    signed_json_body=signed_json_body,
                )

            else:
                return self.async_completion(
                    custom_llm_provider=custom_llm_provider,
                    provider_config=provider_config,
                    api_base=api_base,
                    headers=headers,
                    data=data,
                    timeout=timeout,
                    model=model,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    api_key=api_key,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    encoding=encoding,
                    client=(
                        client
                        if client is not None and isinstance(client, AsyncHTTPHandler)
                        else None
                    ),
                    json_mode=json_mode,
                    signed_json_body=signed_json_body,
                )

        if stream is True:
            data = self._add_stream_param_to_request_body(
                data=data,
                provider_config=provider_config,
                fake_stream=fake_stream,
            )
            if provider_config.has_custom_stream_wrapper is True:
                return provider_config.get_sync_custom_stream_wrapper(
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                    logging_obj=logging_obj,
                    api_base=api_base,
                    headers=headers,
                    data=data,
                    signed_json_body=signed_json_body,
                    messages=messages,
                    client=client,
                    json_mode=json_mode,
                )
            completion_stream, headers = self.make_sync_call(
                provider_config=provider_config,
                api_base=api_base,
                headers=headers,  # type: ignore
                data=data,
                signed_json_body=signed_json_body,
                original_data=data,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                timeout=timeout,
                fake_stream=fake_stream,
                client=(
                    client
                    if client is not None and isinstance(client, HTTPHandler)
                    else None
                ),
                litellm_params=litellm_params,
                json_mode=json_mode,
                optional_params=optional_params,
            )
            return CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        response = self._make_common_sync_call(
            sync_httpx_client=sync_httpx_client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            data=data,
            signed_json_body=signed_json_body,
            timeout=timeout,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
        )
        return provider_config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            json_mode=json_mode,
        )

    def make_sync_call(
        self,
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: dict,
        signed_json_body: Optional[bytes],
        original_data: dict,
        model: str,
        messages: list,
        logging_obj,
        optional_params: dict,
        litellm_params: dict,
        timeout: Union[float, httpx.Timeout],
        fake_stream: bool = False,
        client: Optional[HTTPHandler] = None,
        json_mode: bool = False,
    ) -> Tuple[Any, dict]:
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                {
                    "ssl_verify": litellm_params.get("ssl_verify", None),
                }
            )
        else:
            sync_httpx_client = client
        stream = True
        if fake_stream is True:
            stream = False

        response = self._make_common_sync_call(
            sync_httpx_client=sync_httpx_client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            data=data,
            signed_json_body=signed_json_body,
            timeout=timeout,
            litellm_params=litellm_params,
            stream=stream,
            logging_obj=logging_obj,
        )

        if fake_stream is True:
            model_response: ModelResponse = provider_config.transform_response(
                model=model,
                raw_response=response,
                model_response=litellm.ModelResponse(),
                logging_obj=logging_obj,
                request_data=original_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=None,
                json_mode=json_mode,
            )

            completion_stream: Any = MockResponseIterator(
                model_response=model_response, json_mode=json_mode
            )
        else:
            completion_stream = provider_config.get_model_response_iterator(
                streaming_response=response.iter_lines(),
                sync_stream=True,
                json_mode=json_mode,
            )

        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return completion_stream, dict(response.headers)

    async def acompletion_stream_function(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_llm_provider: str,
        headers: dict,
        provider_config: BaseConfig,
        timeout: Union[float, httpx.Timeout],
        logging_obj: LiteLLMLoggingObj,
        data: dict,
        litellm_params: dict,
        optional_params: dict,
        fake_stream: bool = False,
        client: Optional[AsyncHTTPHandler] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ):
        if provider_config.has_custom_stream_wrapper is True:
            return await provider_config.get_async_custom_stream_wrapper(
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
                api_base=api_base,
                headers=headers,
                data=data,
                messages=messages,
                client=client,
                json_mode=json_mode,
                signed_json_body=signed_json_body,
            )

        completion_stream, _response_headers = await self.make_async_call_stream_helper(
            model=model,
            custom_llm_provider=custom_llm_provider,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            data=data,
            messages=messages,
            logging_obj=logging_obj,
            timeout=timeout,
            fake_stream=fake_stream,
            client=client,
            litellm_params=litellm_params,
            optional_params=optional_params,
            json_mode=json_mode,
            signed_json_body=signed_json_body,
        )
        streamwrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )
        return streamwrapper

    async def make_async_call_stream_helper(
        self,
        model: str,
        custom_llm_provider: str,
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        optional_params: dict,
        fake_stream: bool = False,
        client: Optional[AsyncHTTPHandler] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> Tuple[Any, httpx.Headers]:
        """
        Helper function for making an async call with stream.

        Handles fake stream as well.
        """
        if client is None:
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client
        stream = True
        if fake_stream is True:
            stream = False

        response = await self._make_common_async_call(
            async_httpx_client=async_httpx_client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            data=data,
            signed_json_body=signed_json_body,
            timeout=timeout,
            litellm_params=litellm_params,
            stream=stream,
            logging_obj=logging_obj,
        )

        if fake_stream is True:
            model_response: ModelResponse = provider_config.transform_response(
                model=model,
                raw_response=response,
                model_response=litellm.ModelResponse(),
                logging_obj=logging_obj,
                request_data=data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=None,
                json_mode=json_mode,
            )

            completion_stream: Any = MockResponseIterator(
                model_response=model_response, json_mode=json_mode
            )
        else:
            completion_stream = provider_config.get_model_response_iterator(
                streaming_response=response.aiter_lines(), sync_stream=False
            )
        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return completion_stream, response.headers

    def _add_stream_param_to_request_body(
        self,
        data: dict,
        provider_config: BaseConfig,
        fake_stream: bool,
    ) -> dict:
        """
        Some providers like Bedrock invoke do not support the stream parameter in the request body, we only pass `stream` in the request body the provider supports it.
        """

        if fake_stream is True:
            # remove 'stream' from data
            new_data = data.copy()
            new_data.pop("stream", None)
            return new_data
        if provider_config.supports_stream_param_in_request_body is True:
            data["stream"] = True
        return data

    def embedding(
        self,
        model: str,
        input: list,
        timeout: float,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: Optional[str],
        optional_params: dict,
        litellm_params: dict,
        model_response: EmbeddingResponse,
        api_key: Optional[str] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        aembedding: bool = False,
        headers: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResponse:
        provider_config = ProviderConfigManager.get_provider_embedding_config(
            model=model, provider=litellm.LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(
                f"Provider {custom_llm_provider} does not support embedding"
            )
        # get config from model, custom llm provider
        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers or {},
            model=model,
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        api_base = provider_config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        data = provider_config.transform_embedding_request(
            model=model,
            input=input,
            optional_params=optional_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        if aembedding is True:
            return self.aembedding(  # type: ignore
                request_data=data,
                api_base=api_base,
                headers=headers,
                model=model,
                custom_llm_provider=custom_llm_provider,
                provider_config=provider_config,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key=api_key,
                timeout=timeout,
                client=client,
                optional_params=optional_params,
                litellm_params=litellm_params,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client()
        else:
            sync_httpx_client = client

        try:
            response = sync_httpx_client.post(
                url=api_base,
                headers=headers,
                data=json.dumps(data),
                timeout=timeout,
            )
        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=provider_config,
            )

        return provider_config.transform_embedding_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

    async def aembedding(
        self,
        request_data: dict,
        api_base: str,
        headers: dict,
        model: str,
        custom_llm_provider: str,
        provider_config: BaseEmbeddingConfig,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> EmbeddingResponse:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider)
            )
        else:
            async_httpx_client = client

        try:
            response = await async_httpx_client.post(
                url=api_base,
                headers=headers,
                json=request_data,
                timeout=timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=provider_config)

        return provider_config.transform_embedding_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=request_data,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

    def rerank(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        provider_config: BaseRerankConfig,
        optional_rerank_params: OptionalRerankParams,
        timeout: Optional[Union[float, httpx.Timeout]],
        model_response: RerankResponse,
        _is_async: bool = False,
        headers: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> RerankResponse:
        # get config from model, custom llm provider
        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers or {},
            model=model,
        )

        api_base = provider_config.get_complete_url(
            api_base=api_base,
            model=model,
        )

        data = provider_config.transform_rerank_request(
            model=model,
            optional_rerank_params=optional_rerank_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=optional_rerank_params.get("query", ""),
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        if _is_async is True:
            return self.arerank(  # type: ignore
                model=model,
                request_data=data,
                custom_llm_provider=custom_llm_provider,
                provider_config=provider_config,
                logging_obj=logging_obj,
                model_response=model_response,
                api_base=api_base,
                headers=headers,
                api_key=api_key,
                timeout=timeout,
                client=client,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client()
        else:
            sync_httpx_client = client

        try:
            response = sync_httpx_client.post(
                url=api_base,
                headers=headers,
                data=json.dumps(data),
                timeout=timeout,
            )
        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=provider_config,
            )

        return provider_config.transform_rerank_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
        )

    async def arerank(
        self,
        model: str,
        request_data: dict,
        custom_llm_provider: str,
        provider_config: BaseRerankConfig,
        logging_obj: LiteLLMLoggingObj,
        model_response: RerankResponse,
        api_base: str,
        headers: dict,
        api_key: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> RerankResponse:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider)
            )
        else:
            async_httpx_client = client
        try:
            response = await async_httpx_client.post(
                url=api_base,
                headers=headers,
                data=json.dumps(request_data),
                timeout=timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=provider_config)

        return provider_config.transform_rerank_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=request_data,
        )

    def audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        max_retries: int,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        custom_llm_provider: str,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        atranscription: bool = False,
        headers: Optional[Dict[str, Any]] = None,
        provider_config: Optional[BaseAudioTranscriptionConfig] = None,
    ) -> TranscriptionResponse:
        if provider_config is None:
            raise ValueError(
                f"No provider config found for model: {model} and provider: {custom_llm_provider}"
            )
        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers or {},
            model=model,
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        if client is None or not isinstance(client, HTTPHandler):
            client = _get_httpx_client()

        complete_url = provider_config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Handle the audio file based on type
        data = provider_config.transform_audio_transcription_request(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        binary_data: Optional[bytes] = None
        json_data: Optional[dict] = None
        if isinstance(data, bytes):
            binary_data = data
        else:
            json_data = data

        try:
            # Make the POST request
            response = client.post(
                url=complete_url,
                headers=headers,
                content=binary_data,
                json=json_data,
                timeout=timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=provider_config)

        if isinstance(provider_config, litellm.DeepgramAudioTranscriptionConfig):
            returned_response = provider_config.transform_audio_transcription_response(
                model=model,
                raw_response=response,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data={},
                optional_params=optional_params,
                litellm_params={},
                api_key=api_key,
            )
            return returned_response
        return model_response

    async def async_anthropic_messages_handler(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_provider_config: BaseAnthropicMessagesConfig,
        anthropic_messages_optional_request_params: Dict,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        client: Optional[AsyncHTTPHandler] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        stream: Optional[bool] = False,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[AnthropicMessagesResponse, AsyncIterator]:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.ANTHROPIC
            )
        else:
            async_httpx_client = client

        # Prepare headers
        kwargs = kwargs or {}
        provider_specific_header = cast(
            Optional[litellm.types.utils.ProviderSpecificHeader],
            kwargs.get("provider_specific_header", None),
        )
        extra_headers = (
            provider_specific_header.get("extra_headers", {})
            if provider_specific_header
            else {}
        )
        (
            headers,
            api_base,
        ) = anthropic_messages_provider_config.validate_anthropic_messages_environment(
            headers=extra_headers or {},
            model=model,
            messages=messages,
            optional_params=anthropic_messages_optional_request_params,
            litellm_params=dict(litellm_params),
            api_key=api_key,
            api_base=api_base,
        )

        logging_obj.update_environment_variables(
            model=model,
            optional_params=dict(anthropic_messages_optional_request_params),
            litellm_params={
                "metadata": kwargs.get("metadata", {}),
                "preset_cache_key": None,
                "stream_response": {},
                **anthropic_messages_optional_request_params,
            },
            custom_llm_provider=custom_llm_provider,
        )
        # Prepare request body
        request_body = anthropic_messages_provider_config.transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        logging_obj.stream = stream
        logging_obj.model_call_details.update(request_body)

        # Make the request
        request_url = anthropic_messages_provider_config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=dict(
                litellm_params
            ),  # this uses the invoke config, which expects aws_* params in optional_params
            litellm_params=dict(litellm_params),
            stream=stream,
        )

        headers, signed_json_body = anthropic_messages_provider_config.sign_request(
            headers=headers,
            optional_params=dict(
                litellm_params
            ),  # dynamic aws_* params are passed under litellm_params
            request_data=request_body,
            api_base=request_url,
            stream=stream,
            fake_stream=False,
            model=model,
        )

        logging_obj.pre_call(
            input=[{"role": "user", "content": json.dumps(request_body)}],
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": str(request_url),
                "headers": headers,
            },
        )

        response = await async_httpx_client.post(
            url=request_url,
            headers=headers,
            data=signed_json_body or json.dumps(request_body),
            stream=stream or False,
            logging_obj=logging_obj,
        )
        response.raise_for_status()

        # used for logging + cost tracking
        logging_obj.model_call_details["httpx_response"] = response

        if stream:
            completion_stream = anthropic_messages_provider_config.get_async_streaming_response_iterator(
                model=model,
                httpx_response=response,
                request_body=request_body,
                litellm_logging_obj=logging_obj,
            )
            return completion_stream
        else:
            return anthropic_messages_provider_config.transform_anthropic_messages_response(
                model=model,
                raw_response=response,
                logging_obj=logging_obj,
            )

    def anthropic_messages_handler(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_provider_config: BaseAnthropicMessagesConfig,
        anthropic_messages_optional_request_params: Dict,
        custom_llm_provider: str,
        _is_async: bool,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        stream: Optional[bool] = False,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[
        AnthropicMessagesResponse,
        Coroutine[Any, Any, Union[AnthropicMessagesResponse, AsyncIterator]],
    ]:
        """
        LLM HTTP Handler for Anthropic Messages
        """
        if _is_async:
            # Return the async coroutine if called with _is_async=True
            return self.async_anthropic_messages_handler(
                model=model,
                messages=messages,
                anthropic_messages_provider_config=anthropic_messages_provider_config,
                anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                api_key=api_key,
                api_base=api_base,
                stream=stream,
                kwargs=kwargs,
            )
        raise ValueError("anthropic_messages_handler is not implemented for sync calls")

    def response_api_handler(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_provider_config: BaseResponsesAPIConfig,
        response_api_optional_request_params: Dict,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
        fake_stream: bool = False,
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[
        ResponsesAPIResponse,
        BaseResponsesAPIStreamingIterator,
        Coroutine[
            Any, Any, Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]
        ],
    ]:
        """
        Handles responses API requests.
        When _is_async=True, returns a coroutine instead of making the call directly.
        """
        if _is_async:
            # Return the async coroutine if called with _is_async=True
            return self.async_response_api_handler(
                model=model,
                input=input,
                responses_api_provider_config=responses_api_provider_config,
                response_api_optional_request_params=response_api_optional_request_params,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
                fake_stream=fake_stream,
                litellm_metadata=litellm_metadata,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            api_key=litellm_params.api_key,
            headers=response_api_optional_request_params.get("extra_headers", {}) or {},
            model=model,
        )

        if extra_headers:
            headers.update(extra_headers)

        # Check if streaming is requested
        stream = response_api_optional_request_params.get("stream", False)

        api_base = responses_api_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        data = responses_api_provider_config.transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            if stream:
                # For streaming, use stream=True in the request
                if fake_stream is True:
                    stream, data = self._prepare_fake_stream_request(
                        stream=stream,
                        data=data,
                        fake_stream=fake_stream,
                    )
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout
                    or response_api_optional_request_params.get("timeout"),
                    stream=stream,
                )
                if fake_stream is True:
                    return MockResponsesAPIStreamingIterator(
                        response=response,
                        model=model,
                        logging_obj=logging_obj,
                        responses_api_provider_config=responses_api_provider_config,
                        litellm_metadata=litellm_metadata,
                        custom_llm_provider=custom_llm_provider,
                    )

                return SyncResponsesAPIStreamingIterator(
                    response=response,
                    model=model,
                    logging_obj=logging_obj,
                    responses_api_provider_config=responses_api_provider_config,
                    litellm_metadata=litellm_metadata,
                    custom_llm_provider=custom_llm_provider,
                )
            else:
                # For non-streaming requests
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout
                    or response_api_optional_request_params.get("timeout"),
                )
        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=responses_api_provider_config,
            )

        return responses_api_provider_config.transform_response_api_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_response_api_handler(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_provider_config: BaseResponsesAPIConfig,
        response_api_optional_request_params: Dict,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        fake_stream: bool = False,
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:
        """
        Async version of the responses API handler.
        Uses async HTTP client to make requests.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            api_key=litellm_params.api_key,
            headers=response_api_optional_request_params.get("extra_headers", {}) or {},
            model=model,
        )

        if extra_headers:
            headers.update(extra_headers)

        # Check if streaming is requested
        stream = response_api_optional_request_params.get("stream", False)

        api_base = responses_api_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        data = responses_api_provider_config.transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            if stream:
                # For streaming, we need to use stream=True in the request
                if fake_stream is True:
                    stream, data = self._prepare_fake_stream_request(
                        stream=stream,
                        data=data,
                        fake_stream=fake_stream,
                    )

                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout
                    or response_api_optional_request_params.get("timeout"),
                    stream=stream,
                )

                if fake_stream is True:
                    return MockResponsesAPIStreamingIterator(
                        response=response,
                        model=model,
                        logging_obj=logging_obj,
                        responses_api_provider_config=responses_api_provider_config,
                        litellm_metadata=litellm_metadata,
                        custom_llm_provider=custom_llm_provider,
                    )

                # Return the streaming iterator
                return ResponsesAPIStreamingIterator(
                    response=response,
                    model=model,
                    logging_obj=logging_obj,
                    responses_api_provider_config=responses_api_provider_config,
                    litellm_metadata=litellm_metadata,
                    custom_llm_provider=custom_llm_provider,
                )
            else:
                # For non-streaming, proceed as before
                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout
                    or response_api_optional_request_params.get("timeout"),
                )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=responses_api_provider_config,
            )

        return responses_api_provider_config.transform_response_api_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_delete_response_api_handler(
        self,
        response_id: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str],
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> DeleteResponseResult:
        """
        Async version of the delete response API handler.
        Uses async HTTP client to make requests.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            api_key=litellm_params.api_key,
            headers=extra_headers or {},
            model="None",
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = responses_api_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        url, data = responses_api_provider_config.transform_delete_response_api_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.delete(
                url=url, headers=headers, json=data, timeout=timeout
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=responses_api_provider_config,
            )

        return responses_api_provider_config.transform_delete_response_api_response(
            raw_response=response,
            logging_obj=logging_obj,
        )

    def delete_response_api_handler(
        self,
        response_id: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str],
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[DeleteResponseResult, Coroutine[Any, Any, DeleteResponseResult]]:
        """
        Async version of the responses API handler.
        Uses async HTTP client to make requests.
        """
        if _is_async:
            return self.async_delete_response_api_handler(
                response_id=response_id,
                responses_api_provider_config=responses_api_provider_config,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client,
            )
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            api_key=litellm_params.api_key,
            headers=extra_headers or {},
            model="None",
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = responses_api_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        url, data = responses_api_provider_config.transform_delete_response_api_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.delete(
                url=url, headers=headers, json=data, timeout=timeout
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=responses_api_provider_config,
            )

        return responses_api_provider_config.transform_delete_response_api_response(
            raw_response=response,
            logging_obj=logging_obj,
        )

    def get_responses(
        self,
        response_id: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[ResponsesAPIResponse, Coroutine[Any, Any, ResponsesAPIResponse]]:
        """
        Get a response by ID
        Uses GET /v1/responses/{response_id} endpoint in the responses API
        """
        if _is_async:
            return self.async_get_responses(
                response_id=response_id,
                responses_api_provider_config=responses_api_provider_config,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            api_key=litellm_params.api_key,
            headers=extra_headers or {},
            model="None",
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = responses_api_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        url, data = responses_api_provider_config.transform_get_response_api_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.get(url=url, headers=headers, params=data)
        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=responses_api_provider_config,
            )

        return responses_api_provider_config.transform_get_response_api_response(
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_get_responses(
        self,
        response_id: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> ResponsesAPIResponse:
        """
        Async version of get_responses
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            api_key=litellm_params.api_key,
            headers=extra_headers or {},
            model="None",
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = responses_api_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        url, data = responses_api_provider_config.transform_get_response_api_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.get(
                url=url, headers=headers, params=data
            )

        except Exception as e:
            verbose_logger.exception(f"Error retrieving response: {e}")
            raise self._handle_error(
                e=e,
                provider_config=responses_api_provider_config,
            )

        return responses_api_provider_config.transform_get_response_api_response(
            raw_response=response,
            logging_obj=logging_obj,
        )

    def create_file(
        self,
        create_file_data: CreateFileRequest,
        litellm_params: dict,
        provider_config: BaseFilesConfig,
        headers: dict,
        api_base: Optional[str],
        api_key: Optional[str],
        logging_obj: LiteLLMLoggingObj,
        _is_async: bool = False,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Union[OpenAIFileObject, Coroutine[Any, Any, OpenAIFileObject]]:
        """
        Creates a file using Gemini's two-step upload process
        """
        # get config from model, custom llm provider
        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers,
            model="",
            messages=[],
            optional_params={},
            litellm_params=litellm_params,
        )

        api_base = provider_config.get_complete_file_url(
            api_base=api_base,
            api_key=api_key,
            model="",
            optional_params={},
            litellm_params=litellm_params,
            data=create_file_data,
        )
        if api_base is None:
            raise ValueError("api_base is required for create_file")

        # Get the transformed request data for both steps
        transformed_request = provider_config.transform_create_file_request(
            model="",
            create_file_data=create_file_data,
            litellm_params=litellm_params,
            optional_params={},
        )

        if _is_async:
            return self.async_create_file(
                transformed_request=transformed_request,
                litellm_params=litellm_params,
                provider_config=provider_config,
                headers=headers,
                api_base=api_base,
                logging_obj=logging_obj,
                client=client,
                timeout=timeout,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client()
        else:
            sync_httpx_client = client

        if isinstance(transformed_request, str) or isinstance(
            transformed_request, bytes
        ):
            upload_response = sync_httpx_client.post(
                url=api_base,
                headers=headers,
                data=transformed_request,
                timeout=timeout,
            )
        else:
            try:
                # Step 1: Initial request to get upload URL
                initial_response = sync_httpx_client.post(
                    url=api_base,
                    headers={
                        **headers,
                        **transformed_request["initial_request"]["headers"],
                    },
                    data=json.dumps(transformed_request["initial_request"]["data"]),
                    timeout=timeout,
                )

                # Extract upload URL from response headers
                upload_url = initial_response.headers.get("X-Goog-Upload-URL")

                if not upload_url:
                    raise ValueError("Failed to get upload URL from initial request")

                # Step 2: Upload the actual file
                upload_response = sync_httpx_client.post(
                    url=upload_url,
                    headers=transformed_request["upload_request"]["headers"],
                    data=transformed_request["upload_request"]["data"],
                    timeout=timeout,
                )
            except Exception as e:
                raise self._handle_error(
                    e=e,
                    provider_config=provider_config,
                )

        return provider_config.transform_create_file_response(
            model=None,
            raw_response=upload_response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
        )

    async def async_create_file(
        self,
        transformed_request: Union[bytes, str, dict],
        litellm_params: dict,
        provider_config: BaseFilesConfig,
        headers: dict,
        api_base: str,
        logging_obj: LiteLLMLoggingObj,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ):
        """
        Creates a file using Gemini's two-step upload process
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=provider_config.custom_llm_provider
            )
        else:
            async_httpx_client = client

        if isinstance(transformed_request, str) or isinstance(
            transformed_request, bytes
        ):
            upload_response = await async_httpx_client.post(
                url=api_base,
                headers=headers,
                data=transformed_request,
                timeout=timeout,
            )
        else:
            try:
                # Step 1: Initial request to get upload URL
                initial_response = await async_httpx_client.post(
                    url=api_base,
                    headers={
                        **headers,
                        **transformed_request["initial_request"]["headers"],
                    },
                    data=json.dumps(transformed_request["initial_request"]["data"]),
                    timeout=timeout,
                )

                # Extract upload URL from response headers
                upload_url = initial_response.headers.get("X-Goog-Upload-URL")

                if not upload_url:
                    raise ValueError("Failed to get upload URL from initial request")

                # Step 2: Upload the actual file
                upload_response = await async_httpx_client.post(
                    url=upload_url,
                    headers=transformed_request["upload_request"]["headers"],
                    data=transformed_request["upload_request"]["data"],
                    timeout=timeout,
                )
            except Exception as e:
                verbose_logger.exception(f"Error creating file: {e}")
                raise self._handle_error(
                    e=e,
                    provider_config=provider_config,
                )

        return provider_config.transform_create_file_response(
            model=None,
            raw_response=upload_response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
        )

    def list_files(self):
        """
        Lists all files
        """
        pass

    def delete_file(self):
        """
        Deletes a file
        """
        pass

    def retrieve_file(self):
        """
        Returns the metadata of the file
        """
        pass

    def retrieve_file_content(self):
        """
        Returns the content of the file
        """
        pass

    def _prepare_fake_stream_request(
        self,
        stream: bool,
        data: dict,
        fake_stream: bool,
    ) -> Tuple[bool, dict]:
        """
        Handles preparing a request when `fake_stream` is True.
        """
        if fake_stream is True:
            stream = False
            data.pop("stream", None)
            return stream, data
        return stream, data

    def _handle_error(
        self,
        e: Exception,
        provider_config: Union[
            BaseConfig, BaseRerankConfig, BaseResponsesAPIConfig, BaseImageEditConfig
        ],
    ):
        status_code = getattr(e, "status_code", 500)
        error_headers = getattr(e, "headers", None)
        if isinstance(e, httpx.HTTPStatusError):
            error_text = e.response.text
            status_code = e.response.status_code
        else:
            error_text = getattr(e, "text", str(e))
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        if error_response and hasattr(error_response, "text"):
            error_text = getattr(error_response, "text", error_text)
        if error_headers:
            error_headers = dict(error_headers)
        else:
            error_headers = {}

        raise provider_config.get_error_class(
            error_message=error_text,
            status_code=status_code,
            headers=error_headers,
        )

    async def async_realtime(
        self,
        model: str,
        websocket: Any,
        logging_obj: LiteLLMLoggingObj,
        provider_config: BaseRealtimeConfig,
        headers: dict,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        timeout: Optional[float] = None,
    ):
        import websockets
        from websockets.asyncio.client import ClientConnection

        url = provider_config.get_complete_url(api_base, model, api_key)
        headers = provider_config.validate_environment(
            headers=headers,
            model=model,
            api_key=api_key,
        )

        try:
            async with websockets.connect(  # type: ignore
                url, extra_headers=headers
            ) as backend_ws:
                realtime_streaming = RealTimeStreaming(
                    websocket,
                    cast(ClientConnection, backend_ws),
                    logging_obj,
                    provider_config,
                    model,
                )
                await realtime_streaming.bidirectional_forward()

        except websockets.exceptions.InvalidStatusCode as e:  # type: ignore
            verbose_logger.exception(f"Error connecting to backend: {e}")
            await websocket.close(code=e.status_code, reason=str(e))
        except Exception as e:
            verbose_logger.exception(f"Error connecting to backend: {e}")
            try:
                await websocket.close(
                    code=1011, reason=f"Internal server error: {str(e)}"
                )
            except RuntimeError as close_error:
                if "already completed" in str(close_error) or "websocket.close" in str(
                    close_error
                ):
                    # The WebSocket is already closed or the response is completed, so we can ignore this error
                    pass
                else:
                    # If it's a different RuntimeError, we might want to log it or handle it differently
                    raise Exception(
                        f"Unexpected error while closing WebSocket: {close_error}"
                    )

    def image_edit_handler(
        self,
        model: str,
        image: Any,
        prompt: str,
        image_edit_provider_config: BaseImageEditConfig,
        image_edit_optional_request_params: Dict,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
        fake_stream: bool = False,
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[ImageResponse, Coroutine[Any, Any, ImageResponse],]:
        """

        Handles image edit requests.
        When _is_async=True, returns a coroutine instead of making the call directly.
        """
        if _is_async:
            # Return the async coroutine if called with _is_async=True
            return self.async_image_edit_handler(
                model=model,
                image=image,
                prompt=prompt,
                image_edit_provider_config=image_edit_provider_config,
                image_edit_optional_request_params=image_edit_optional_request_params,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
                fake_stream=fake_stream,
                litellm_metadata=litellm_metadata,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = image_edit_provider_config.validate_environment(
            api_key=litellm_params.api_key,
            headers=image_edit_optional_request_params.get("extra_headers", {}) or {},
            model=model,
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = image_edit_provider_config.get_complete_url(
            model=model,
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        data, files = image_edit_provider_config.transform_image_edit_request(
            model=model,
            image=image,
            prompt=prompt,
            image_edit_optional_request_params=image_edit_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.post(
                url=api_base,
                headers=headers,
                data=data,
                files=files,
                timeout=timeout,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=image_edit_provider_config,
            )

        return image_edit_provider_config.transform_image_edit_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_image_edit_handler(
        self,
        model: str,
        image: FileTypes,
        prompt: str,
        image_edit_provider_config: BaseImageEditConfig,
        image_edit_optional_request_params: Dict,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        fake_stream: bool = False,
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ) -> ImageResponse:
        """
        Async version of the image edit handler.
        Uses async HTTP client to make requests.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = image_edit_provider_config.validate_environment(
            api_key=litellm_params.api_key,
            headers=image_edit_optional_request_params.get("extra_headers", {}) or {},
            model=model,
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = image_edit_provider_config.get_complete_url(
            model=model,
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        data, files = image_edit_provider_config.transform_image_edit_request(
            model=model,
            image=image,
            prompt=prompt,
            image_edit_optional_request_params=image_edit_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.post(
                url=api_base,
                headers=headers,
                data=data,
                files=files,
                timeout=timeout,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=image_edit_provider_config,
            )

        return image_edit_provider_config.transform_image_edit_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )
