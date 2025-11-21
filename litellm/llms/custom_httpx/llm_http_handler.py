import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    List,
    Literal,
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
from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
from litellm.llms.base_llm.batches.transformation import BaseBatchesConfig
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.base_llm.containers.transformation import BaseContainerConfig
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.base_llm.files.transformation import BaseFilesConfig
from litellm.llms.base_llm.google_genai.transformation import (
    BaseGoogleGenAIGenerateContentConfig,
)
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig, OCRResponse
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.base_llm.search.transformation import BaseSearchConfig, SearchResponse
from litellm.llms.base_llm.text_to_speech.transformation import BaseTextToSpeechConfig
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.llms.base_llm.vector_store_files.transformation import (
    BaseVectorStoreFilesConfig,
)
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
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
from litellm.types.containers.main import (
    ContainerListResponse,
    ContainerObject,
    DeleteContainerResult,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import (
    CreateBatchRequest,
    CreateFileRequest,
    HttpxBinaryResponseContent,
    OpenAIFileObject,
    ResponseInputParam,
    ResponsesAPIResponse,
)
from litellm.types.rerank import RerankResponse
from litellm.types.responses.main import DeleteResponseResult
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import (
    EmbeddingResponse,
    FileTypes,
    LiteLLMBatch,
    TranscriptionResponse,
)
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
)
from litellm.types.vector_store_files import (
    VectorStoreFileContentResponse,
    VectorStoreFileCreateRequest,
    VectorStoreFileDeleteResponse,
    VectorStoreFileListQueryParams,
    VectorStoreFileListResponse,
    VectorStoreFileObject,
    VectorStoreFileUpdateRequest,
)
from litellm.types.videos.main import VideoObject
from litellm.utils import (
    CustomStreamWrapper,
    ImageResponse,
    ModelResponse,
    ProviderConfigManager,
)

if TYPE_CHECKING:
    from aiohttp import ClientSession

    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig

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
        shared_session: Optional["ClientSession"] = None,
    ):
        if client is None:
            verbose_logger.debug(
                f"Creating HTTP client with shared_session: {id(shared_session) if shared_session else None}"
            )
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
                shared_session=shared_session,
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
        api_base: Optional[str],
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
        shared_session: Optional["ClientSession"] = None,
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
            api_key=api_key,
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
                    shared_session=shared_session,
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
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
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
        aembedding: Optional[bool] = False,
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
        optional_rerank_params: Dict,
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
            optional_params=optional_rerank_params,
        )

        api_base = provider_config.get_complete_url(
            api_base=api_base,
            model=model,
            optional_params=optional_rerank_params,
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

    def _prepare_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        headers: Optional[Dict[str, Any]],
        provider_config: BaseAudioTranscriptionConfig,
    ) -> Tuple[dict, str, Union[dict, bytes, None], Optional[dict]]:
        """
        Shared logic for preparing audio transcription requests.
        Returns: (headers, complete_url, data, files)
        """
        # Handle the response based on type
        from litellm.llms.base_llm.audio_transcription.transformation import (
            AudioTranscriptionRequestData,
        )

        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers or {},
            model=model,
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        complete_url = provider_config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Transform the request to get data
        transformed_result = provider_config.transform_audio_transcription_request(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # All providers now return AudioTranscriptionRequestData
        if not isinstance(transformed_result, AudioTranscriptionRequestData):
            raise ValueError(
                f"Provider {provider_config.__class__.__name__} must return AudioTranscriptionRequestData"
            )

        data = transformed_result.data
        files = transformed_result.files

        ## LOGGING
        logging_obj.pre_call(
            input=optional_params.get("query", ""),
            api_key=api_key,
            additional_args={
                "complete_input_dict": data or {},
                "api_base": complete_url,
                "headers": headers,
            },
        )

        return headers, complete_url, data, files

    def _transform_audio_transcription_response(
        self,
        provider_config: BaseAudioTranscriptionConfig,
        model: str,
        response: httpx.Response,
        model_response: TranscriptionResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        api_key: Optional[str],
    ) -> TranscriptionResponse:
        """Shared logic for transforming audio transcription responses."""
        return provider_config.transform_audio_transcription_response(
            raw_response=response,
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
        shared_session: Optional["ClientSession"] = None,
    ) -> Union[TranscriptionResponse, Coroutine[Any, Any, TranscriptionResponse]]:
        if provider_config is None:
            raise ValueError(
                f"No provider config found for model: {model} and provider: {custom_llm_provider}"
            )

        if atranscription is True:
            return self.async_audio_transcriptions(  # type: ignore
                model=model,
                audio_file=audio_file,
                optional_params=optional_params,
                litellm_params=litellm_params,
                model_response=model_response,
                timeout=timeout,
                max_retries=max_retries,
                logging_obj=logging_obj,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                client=client,
                headers=headers,
                provider_config=provider_config,
                shared_session=shared_session,
            )

        # Prepare the request
        (
            headers,
            complete_url,
            data,
            files,
        ) = self._prepare_audio_transcription_request(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            headers=headers,
            provider_config=provider_config,
        )

        if client is None or not isinstance(client, HTTPHandler):
            client = _get_httpx_client()

        try:
            # Make the POST request - clean and simple, always use data and files
            response = client.post(
                url=complete_url,
                headers=headers,
                data=data,
                files=files,
                json=(
                    data if files is None and isinstance(data, dict) else None
                ),  # Use json param only when no files and data is dict
                timeout=timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=provider_config)

        return self._transform_audio_transcription_response(
            provider_config=provider_config,
            model=model,
            response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_key=api_key,
        )

    async def async_audio_transcriptions(
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
        headers: Optional[Dict[str, Any]] = None,
        provider_config: Optional[BaseAudioTranscriptionConfig] = None,
        shared_session: Optional["ClientSession"] = None,
    ) -> TranscriptionResponse:
        if provider_config is None:
            raise ValueError(
                f"No provider config found for model: {model} and provider: {custom_llm_provider}"
            )

        # Prepare the request
        (
            headers,
            complete_url,
            data,
            files,
        ) = self._prepare_audio_transcription_request(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            headers=headers,
            provider_config=provider_config,
        )

        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
                shared_session=shared_session,
            )
        else:
            async_httpx_client = client

        try:
            # Make the async POST request - clean and simple, always use data and files
            response = await async_httpx_client.post(
                url=complete_url,
                headers=headers,
                data=data,
                files=files,
                json=(
                    data if files is None and isinstance(data, dict) else None
                ),  # Use json param only when no files and data is dict
                timeout=timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=provider_config)

        return self._transform_audio_transcription_response(
            provider_config=provider_config,
            model=model,
            response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_key=api_key,
        )

    def _prepare_ocr_request(
        self,
        model: str,
        document: Dict[str, str],
        optional_params: dict,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        headers: Optional[Dict[str, Any]],
        provider_config: BaseOCRConfig,
        litellm_params: dict,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], None]:
        """
        Shared logic for preparing OCR requests.
        Returns: (headers, complete_url, data, files)
        """
        from litellm.llms.base_llm.ocr.transformation import OCRRequestData
        headers = provider_config.validate_environment(
            api_key=api_key,
            api_base=api_base,
            headers=headers or {},
            model=model,
            litellm_params=litellm_params,
        )

        complete_url = provider_config.get_complete_url(
            api_base=api_base,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Transform the request to get data and files
        transformed_result = provider_config.transform_ocr_request(
            model=model,
            document=document,
            optional_params=optional_params,
            headers=headers,
        )

        # All providers return OCRRequestData
        if not isinstance(transformed_result, OCRRequestData):
            raise ValueError(
                f"Provider {provider_config.__class__.__name__} must return OCRRequestData"
            )

        # Data is always a dict for Mistral OCR format
        if not isinstance(transformed_result.data, dict):
            raise ValueError(
                f"Expected dict data for OCR request, got {type(transformed_result.data)}"
            )

        data = transformed_result.data

        ## LOGGING
        logging_obj.pre_call(
            input="OCR document processing",
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": complete_url,
                "headers": headers,
            },
        )

        return headers, complete_url, data, None

    async def _async_prepare_ocr_request(
        self,
        model: str,
        document: Dict[str, str],
        optional_params: dict,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        headers: Optional[Dict[str, Any]],
        provider_config: BaseOCRConfig,
        litellm_params: dict,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], None]:
        """
        Async version of _prepare_ocr_request for providers that need async transforms.
        Returns: (headers, complete_url, data, files)
        """
        from litellm.llms.base_llm.ocr.transformation import OCRRequestData

        headers = provider_config.validate_environment(
            api_key=api_key,
            api_base=api_base,
            headers=headers or {},
            model=model,
            litellm_params=litellm_params,
        )

        complete_url = provider_config.get_complete_url(
            api_base=api_base,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Use async transform (providers can override this method if they need async operations)
        transformed_result = await provider_config.async_transform_ocr_request(
            model=model,
            document=document,
            optional_params=optional_params,
            headers=headers,
        )

        # All providers return OCRRequestData
        if not isinstance(transformed_result, OCRRequestData):
            raise ValueError(
                f"Provider {provider_config.__class__.__name__} must return OCRRequestData"
            )

        # Data is always a dict for Mistral OCR format
        if not isinstance(transformed_result.data, dict):
            raise ValueError(
                f"Expected dict data for OCR request, got {type(transformed_result.data)}"
            )

        data = transformed_result.data

        ## LOGGING
        logging_obj.pre_call(
            input="OCR document processing",
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": complete_url,
                "headers": headers,
            },
        )

        return headers, complete_url, data, None

    def _transform_ocr_response(
        self,
        provider_config: BaseOCRConfig,
        model: str,
        response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> OCRResponse:
        """Shared logic for transforming OCR responses."""
        return provider_config.transform_ocr_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

    def ocr(
        self,
        model: str,
        document: Dict[str, str],
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        custom_llm_provider: str,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        aocr: bool = False,
        headers: Optional[Dict[str, Any]] = None,
        provider_config: Optional[BaseOCRConfig] = None,
        litellm_params: Optional[dict] = None,
    ) -> Union[OCRResponse, Coroutine[Any, Any, OCRResponse]]:
        """
        Sync OCR handler.
        """
        if provider_config is None:
            raise ValueError(
                f"No provider config found for model: {model} and provider: {custom_llm_provider}"
            )

        if litellm_params is None:
            litellm_params = {}

        if aocr is True:
            return self.async_ocr(
                model=model,
                document=document,
                optional_params=optional_params,
                timeout=timeout,
                logging_obj=logging_obj,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                client=client,
                headers=headers,
                provider_config=provider_config,
                litellm_params=litellm_params,
            )

        # Prepare the request
        headers, complete_url, data, files = self._prepare_ocr_request(
            model=model,
            document=document,
            optional_params=optional_params,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            headers=headers,
            provider_config=provider_config,
            litellm_params=litellm_params,
        )

        if client is None or not isinstance(client, HTTPHandler):
            client = _get_httpx_client()

        try:
            # Make the POST request with JSON data (Mistral format)
            response = client.post(
                url=complete_url,
                headers=headers,
                json=data,
                timeout=timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=provider_config)

        return self._transform_ocr_response(
            provider_config=provider_config,
            model=model,
            response=response,
            logging_obj=logging_obj,
        )

    async def async_ocr(
        self,
        model: str,
        document: Dict[str, str],
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        custom_llm_provider: str,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        headers: Optional[Dict[str, Any]] = None,
        provider_config: Optional[BaseOCRConfig] = None,
        litellm_params: Optional[dict] = None,
    ) -> OCRResponse:
        """
        Async OCR handler.
        """
        if provider_config is None:
            raise ValueError(
                f"No provider config found for model: {model} and provider: {custom_llm_provider}"
            )

        if litellm_params is None:
            litellm_params = {}

        # Prepare the request using async prepare method
        headers, complete_url, data, files = await self._async_prepare_ocr_request(
            model=model,
            document=document,
            optional_params=optional_params,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            headers=headers,
            provider_config=provider_config,
            litellm_params=litellm_params,
        )

        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
            )
        else:
            async_httpx_client = client

        try:
            # Make the async POST request with JSON data (Mistral format)
            response = await async_httpx_client.post(
                url=complete_url,
                headers=headers,
                json=data,
                timeout=timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=provider_config)

        # Use async response transform for async operations
        return await provider_config.async_transform_ocr_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

    def search(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        custom_llm_provider: str,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        asearch: bool = False,
        headers: Optional[Dict[str, Any]] = None,
        provider_config: Optional[BaseSearchConfig] = None,
    ) -> Union[SearchResponse, Coroutine[Any, Any, SearchResponse]]:
        """
        Sync Search handler.
        """
        if provider_config is None:
            raise ValueError(
                f"No provider config found for provider: {custom_llm_provider}"
            )

        if asearch is True:
            return self.async_search(
                query=query,
                optional_params=optional_params,
                timeout=timeout,
                logging_obj=logging_obj,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                client=client,
                headers=headers,
                provider_config=provider_config,
            )

        # Validate environment and get headers
        headers = provider_config.validate_environment(
            api_key=api_key,
            api_base=api_base,
            headers=headers or {},
        )

        # Transform the request
        data = provider_config.transform_search_request(
            query=query,
            optional_params=optional_params,
        )

        # Get complete URL (pass data for providers that need request body for URL construction)
        complete_url = provider_config.get_complete_url(
            api_base=api_base,
            optional_params=optional_params,
            data=data,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=query if isinstance(query, str) else str(query),
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": complete_url,
                "headers": headers,
            },
        )

        if client is None or not isinstance(client, HTTPHandler):
            client = _get_httpx_client()

        # Check HTTP method from provider config
        http_method = provider_config.get_http_method()

        try:
            if http_method == "GET":
                # Make GET request (URL already contains query params from get_complete_url)
                # Note: timeout is set on the client itself, not per-request for GET
                response = client.get(
                    url=complete_url,
                    headers=headers,
                )
            else:
                # Make POST request with JSON data
                response = client.post(
                    url=complete_url,
                    headers=headers,
                    json=data,
                    timeout=timeout,
                )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=provider_config)

        return provider_config.transform_search_response(
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_search(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        api_base: Optional[str],
        custom_llm_provider: str,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        headers: Optional[Dict[str, Any]] = None,
        provider_config: Optional[BaseSearchConfig] = None,
    ) -> SearchResponse:
        """
        Async Search handler.
        """
        if provider_config is None:
            raise ValueError(
                f"No provider config found for provider: {custom_llm_provider}"
            )

        # Validate environment and get headers
        headers = provider_config.validate_environment(
            api_key=api_key,
            api_base=api_base,
            headers=headers or {},
        )

        # Transform the request first
        data = provider_config.transform_search_request(
            query=query,
            optional_params=optional_params,
        )

        # Get complete URL (pass data for providers that need request body for URL construction)
        complete_url = provider_config.get_complete_url(
            api_base=api_base,
            optional_params=optional_params,
            data=data,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=query if isinstance(query, str) else str(query),
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": complete_url,
                "headers": headers,
            },
        )

        if client is None or not isinstance(client, AsyncHTTPHandler):
            # For search providers, use special Search provider type
            from litellm.types.llms.custom_http import httpxSpecialProvider

            async_httpx_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.Search
            )
        else:
            async_httpx_client = client

        # Check HTTP method from provider config
        http_method = provider_config.get_http_method().upper()

        try:
            if http_method == "GET":
                # Make async GET request (URL already contains query params from get_complete_url)
                # Note: timeout is set on the client itself, not per-request for GET
                response = await async_httpx_client.get(
                    url=complete_url,
                    headers=headers,
                )
            else:
                # Make async POST request with JSON data
                response = await async_httpx_client.post(
                    url=complete_url,
                    headers=headers,
                    json=data,  # type: ignore
                    timeout=timeout,
                )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=provider_config)

        return provider_config.transform_search_response(
            raw_response=response,
            logging_obj=logging_obj,
        )

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
        from litellm.litellm_core_utils.get_provider_specific_headers import (
            ProviderSpecificHeaderUtils,
        )

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
        extra_headers = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            provider_specific_header=provider_specific_header,
            custom_llm_provider=custom_llm_provider,
        )
        forwarded_headers = kwargs.get("headers", None)
        if forwarded_headers and extra_headers:
            merged_headers = {**forwarded_headers, **extra_headers}
        else:
            merged_headers = forwarded_headers or extra_headers
        (
            headers,
            api_base,
        ) = anthropic_messages_provider_config.validate_anthropic_messages_environment(
            headers=merged_headers or {},
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
            api_key=api_key,
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

        try:
            response = await async_httpx_client.post(
                url=request_url,
                headers=headers,
                data=signed_json_body or json.dumps(request_body),
                stream=stream or False,
                logging_obj=logging_obj,
            )
            response.raise_for_status()
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=anthropic_messages_provider_config
            )

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
        shared_session: Optional["ClientSession"] = None,
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
                shared_session=shared_session,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            headers=response_api_optional_request_params.get("extra_headers", {}) or {},
            model=model,
            litellm_params=litellm_params,
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

        if extra_body:
            data.update(extra_body)

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
                    or float(response_api_optional_request_params.get("timeout", 0)),
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
                    or float(response_api_optional_request_params.get("timeout", 0)),
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
        shared_session: Optional["ClientSession"] = None,
    ) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:
        """
        Async version of the responses API handler.
        Uses async HTTP client to make requests.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            verbose_logger.debug(
                f"Creating HTTP client for responses API with shared_session: {id(shared_session) if shared_session else None}"
            )
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
                shared_session=shared_session,
            )
        else:
            async_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            headers=response_api_optional_request_params.get("extra_headers", {}) or {},
            model=model,
            litellm_params=litellm_params,
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

        if extra_body:
            data.update(extra_body)

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
                    or float(response_api_optional_request_params.get("timeout", 0)),
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
                    or float(response_api_optional_request_params.get("timeout", 0)),
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
        shared_session: Optional["ClientSession"] = None,
    ) -> DeleteResponseResult:
        """
        Async version of the delete response API handler.
        Uses async HTTP client to make requests.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            verbose_logger.debug(
                f"Creating HTTP client for delete_response with shared_session: {id(shared_session) if shared_session else None}"
            )
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
                shared_session=shared_session,
            )
        else:
            async_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            headers=extra_headers or {}, model="None", litellm_params=litellm_params
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
        shared_session: Optional["ClientSession"] = None,
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
                shared_session=shared_session,
            )
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            headers=extra_headers or {}, model="None", litellm_params=litellm_params
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
        shared_session: Optional["ClientSession"] = None,
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
                shared_session=shared_session,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            headers=extra_headers or {}, model="None", litellm_params=litellm_params
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
        shared_session: Optional["ClientSession"] = None,
    ) -> ResponsesAPIResponse:
        """
        Async version of get_responses
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            verbose_logger.debug(
                f"Creating HTTP client for get_responses with shared_session: {id(shared_session) if shared_session else None}"
            )
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
                shared_session=shared_session,
            )
        else:
            async_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            headers=extra_headers or {}, model="None", litellm_params=litellm_params
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

    #####################################################################
    ################ LIST RESPONSES INPUT ITEMS HANDLER ###########################
    #####################################################################
    def list_responses_input_items(
        self,
        response_id: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        include: Optional[List[str]] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
        shared_session: Optional["ClientSession"] = None,
    ) -> Union[Dict, Coroutine[Any, Any, Dict]]:
        if _is_async:
            return self.async_list_responses_input_items(
                response_id=response_id,
                responses_api_provider_config=responses_api_provider_config,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
                after=after,
                before=before,
                include=include,
                limit=limit,
                order=order,
                extra_headers=extra_headers,
                timeout=timeout,
                client=client,
                shared_session=shared_session,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            headers=extra_headers or {}, model="None", litellm_params=litellm_params
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = responses_api_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        url, params = responses_api_provider_config.transform_list_input_items_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
            after=after,
            before=before,
            include=include,
            limit=limit,
            order=order,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.get(url=url, headers=headers, params=params)
        except Exception as e:
            raise self._handle_error(e=e, provider_config=responses_api_provider_config)

        return responses_api_provider_config.transform_list_input_items_response(
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_list_responses_input_items(
        self,
        response_id: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        include: Optional[List[str]] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        shared_session: Optional["ClientSession"] = None,
    ) -> Dict:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            verbose_logger.debug(
                f"Creating HTTP client for list_input_items with shared_session: {id(shared_session) if shared_session else None}"
            )
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
                shared_session=shared_session,
            )
        else:
            async_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            headers=extra_headers or {}, model="None", litellm_params=litellm_params
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = responses_api_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        url, params = responses_api_provider_config.transform_list_input_items_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
            after=after,
            before=before,
            include=include,
            limit=limit,
            order=order,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.get(
                url=url, headers=headers, params=params
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=responses_api_provider_config)

        return responses_api_provider_config.transform_list_input_items_response(
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

        if isinstance(transformed_request, dict) and "method" in transformed_request:
            # Handle pre-signed requests (e.g., from Bedrock S3 uploads)
            upload_response = getattr(
                sync_httpx_client, transformed_request["method"].lower()
            )(
                url=transformed_request["url"],
                headers=transformed_request["headers"],
                data=transformed_request["data"],
                timeout=timeout,
            )
        elif isinstance(transformed_request, str) or isinstance(
            transformed_request, bytes
        ):
            # Handle traditional file uploads
            # Ensure transformed_request is a string for httpx compatibility
            if isinstance(transformed_request, bytes):
                transformed_request = transformed_request.decode("utf-8")

            # Use the HTTP method specified by the provider config
            http_method = provider_config.file_upload_http_method.upper()
            if http_method == "PUT":
                upload_response = sync_httpx_client.put(
                    url=api_base,
                    headers=headers,
                    data=transformed_request,
                    timeout=timeout,
                )
            else:  # Default to POST
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

        # Store the upload URL in litellm_params for the transformation method
        litellm_params_with_url = dict(litellm_params)
        litellm_params_with_url["upload_url"] = api_base

        return provider_config.transform_create_file_response(
            model=None,
            raw_response=upload_response,
            logging_obj=logging_obj,
            litellm_params=litellm_params_with_url,
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

        #########################################################
        # Debug Logging
        #########################################################
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": transformed_request,
                "api_base": api_base,
                "headers": headers,
            },
        )

        if isinstance(transformed_request, dict) and "method" in transformed_request:
            # Handle pre-signed requests (e.g., from Bedrock S3 uploads)
            upload_response = await getattr(
                async_httpx_client, transformed_request["method"].lower()
            )(
                url=transformed_request["url"],
                headers=transformed_request["headers"],
                data=transformed_request["data"],
                timeout=timeout,
            )
        elif isinstance(transformed_request, str) or isinstance(
            transformed_request, bytes
        ):
            # Handle traditional file uploads
            # Ensure transformed_request is a string for httpx compatibility
            if isinstance(transformed_request, bytes):
                transformed_request = transformed_request.decode("utf-8")

            # Use the HTTP method specified by the provider config
            http_method = provider_config.file_upload_http_method.upper()
            if http_method == "PUT":
                upload_response = await async_httpx_client.put(
                    url=api_base,
                    headers=headers,
                    data=transformed_request,
                    timeout=timeout,
                )
            else:  # Default to POST
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

    def create_batch(
        self,
        create_batch_data: "CreateBatchRequest",
        litellm_params: dict,
        provider_config: "BaseBatchesConfig",
        headers: dict,
        api_base: Optional[str],
        api_key: Optional[str],
        logging_obj: "LiteLLMLoggingObj",
        _is_async: bool = False,
        client: Optional[Union["HTTPHandler", "AsyncHTTPHandler"]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        model: Optional[str] = None,
    ) -> Union["LiteLLMBatch", Coroutine[Any, Any, "LiteLLMBatch"]]:
        """
        Creates a batch using provider-specific batch creation process
        """
        # get config from model, custom llm provider
        if model is None:
            raise ValueError("model is required for create_batch")

        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            messages=[],
            optional_params={},
            litellm_params=litellm_params,
        )

        api_base = provider_config.get_complete_batch_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params={},
            litellm_params=litellm_params,
            data=create_batch_data,
        )
        if api_base is None:
            raise ValueError("api_base is required for create_batch")

        # Get the transformed request data
        transformed_request = provider_config.transform_create_batch_request(
            model=model,
            create_batch_data=create_batch_data,
            litellm_params=litellm_params,
            optional_params={},
        )

        if _is_async:
            return self.async_create_batch(
                transformed_request=transformed_request,
                litellm_params=litellm_params,
                provider_config=provider_config,
                headers=headers,
                api_base=api_base,
                logging_obj=logging_obj,
                client=client,
                timeout=timeout,
                create_batch_data=create_batch_data,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client()
        else:
            sync_httpx_client = client

        try:
            if (
                isinstance(transformed_request, dict)
                and "method" in transformed_request
            ):
                # Handle pre-signed requests (e.g., from Bedrock with AWS auth)
                batch_response = getattr(
                    sync_httpx_client, transformed_request["method"].lower()
                )(
                    url=transformed_request["url"],
                    headers=transformed_request["headers"],
                    data=transformed_request["data"],
                    timeout=timeout,
                )
            elif isinstance(transformed_request, dict):
                # For other providers that use JSON requests
                batch_response = sync_httpx_client.post(
                    url=api_base,
                    headers={**headers, "Content-Type": "application/json"},
                    json=transformed_request,
                    timeout=timeout,
                )
            else:
                # Handle other request types if needed
                batch_response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    data=transformed_request,
                    timeout=timeout,
                )
        except Exception as e:
            verbose_logger.exception(f"Error creating batch: {e}")
            raise self._handle_error(
                e=e,
                provider_config=provider_config,
            )

        # Store original request for response transformation
        litellm_params_with_request = {
            **litellm_params,
            "original_batch_request": create_batch_data,
        }

        return provider_config.transform_create_batch_response(
            model=model,
            raw_response=batch_response,
            logging_obj=logging_obj,
            litellm_params=litellm_params_with_request,
        )

    def retrieve_batch(
        self,
        batch_id: str,
        litellm_params: dict,
        provider_config: "BaseBatchesConfig",
        headers: dict,
        api_base: Optional[str],
        api_key: Optional[str],
        logging_obj: "LiteLLMLoggingObj",
        _is_async: bool = False,
        client: Optional[Union["HTTPHandler", "AsyncHTTPHandler"]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        model: Optional[str] = None,
    ) -> Union["LiteLLMBatch", Coroutine[Any, Any, "LiteLLMBatch"]]:
        """
        Retrieve a batch using provider-specific configuration.
        """
        # Transform the request using provider config
        transformed_request = provider_config.transform_retrieve_batch_request(
            batch_id=batch_id,
            optional_params=litellm_params,
            litellm_params=litellm_params,
        )

        if _is_async:
            return self.async_retrieve_batch(
                transformed_request=transformed_request,
                litellm_params=litellm_params,
                provider_config=provider_config,
                headers=headers,
                api_base=api_base,
                logging_obj=logging_obj,
                client=client,
                timeout=timeout,
                batch_id=batch_id,
                model=model,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client()
        else:
            sync_httpx_client = client

        try:
            if (
                isinstance(transformed_request, dict)
                and "method" in transformed_request
            ):
                # Handle pre-signed requests (e.g., from Bedrock with AWS auth)
                method = transformed_request["method"].lower()
                request_kwargs = {
                    "url": transformed_request["url"],
                    "headers": transformed_request["headers"],
                }

                # Only add data for non-GET requests
                if method != "get" and transformed_request.get("data") is not None:
                    request_kwargs["data"] = transformed_request["data"]

                batch_response = getattr(sync_httpx_client, method)(**request_kwargs)
            elif isinstance(transformed_request, dict) and api_base:
                # For other providers that use JSON requests
                batch_response = sync_httpx_client.get(
                    url=api_base,
                    headers={**headers, "Content-Type": "application/json"},
                    params=transformed_request,
                )
            else:
                # Handle other request types if needed
                if not api_base:
                    raise ValueError("api_base is required for non-pre-signed requests")
                batch_response = sync_httpx_client.get(
                    url=api_base,
                    headers=headers,
                )
        except Exception as e:
            verbose_logger.exception(f"Error retrieving batch: {e}")
            raise self._handle_error(
                e=e,
                provider_config=provider_config,
            )

        return provider_config.transform_retrieve_batch_response(
            model=model,
            raw_response=batch_response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
        )

    async def async_create_batch(
        self,
        transformed_request: Union[bytes, str, dict],
        litellm_params: dict,
        provider_config: "BaseBatchesConfig",
        headers: dict,
        api_base: str,
        logging_obj: "LiteLLMLoggingObj",
        client: Optional[Union["HTTPHandler", "AsyncHTTPHandler"]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        create_batch_data: Optional["CreateBatchRequest"] = None,
        model: Optional[str] = None,
    ):
        """
        Async version of create_batch
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=provider_config.custom_llm_provider
            )
        else:
            async_httpx_client = client

        #########################################################
        # Debug Logging
        #########################################################
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": transformed_request,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            if (
                isinstance(transformed_request, dict)
                and "method" in transformed_request
            ):
                # Handle pre-signed requests (e.g., from Bedrock with AWS auth)
                batch_response = await getattr(
                    async_httpx_client, transformed_request["method"].lower()
                )(
                    url=transformed_request["url"],
                    headers=transformed_request["headers"],
                    data=transformed_request["data"],
                    timeout=timeout,
                )
            elif isinstance(transformed_request, dict):
                # For other providers that use JSON requests
                batch_response = await async_httpx_client.post(
                    url=api_base,
                    headers={**headers, "Content-Type": "application/json"},
                    json=transformed_request,
                    timeout=timeout,
                )
            else:
                # Handle other request types if needed
                batch_response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    data=transformed_request,
                    timeout=timeout,
                )
        except Exception as e:
            verbose_logger.exception(f"Error creating batch: {e}")
            raise self._handle_error(
                e=e,
                provider_config=provider_config,
            )

        # Store original request for response transformation (for async version)
        litellm_params_with_request = {
            **litellm_params,
            "original_batch_request": create_batch_data or {},
        }

        return provider_config.transform_create_batch_response(
            model=model,
            raw_response=batch_response,
            logging_obj=logging_obj,
            litellm_params=litellm_params_with_request,
        )

    async def async_retrieve_batch(
        self,
        transformed_request: Union[bytes, str, dict],
        litellm_params: dict,
        provider_config: "BaseBatchesConfig",
        headers: dict,
        api_base: Optional[str],
        logging_obj: "LiteLLMLoggingObj",
        client: Optional[Union["HTTPHandler", "AsyncHTTPHandler"]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        batch_id: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Async version of retrieve_batch
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=provider_config.custom_llm_provider
            )
        else:
            async_httpx_client = client

        #########################################################
        # Debug Logging
        #########################################################
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": transformed_request,
                "api_base": api_base,
                "headers": headers,
                "batch_id": batch_id,
            },
        )

        try:
            if (
                isinstance(transformed_request, dict)
                and "method" in transformed_request
            ):
                # Handle pre-signed requests (e.g., from Bedrock with AWS auth)
                method = transformed_request["method"].lower()
                request_kwargs = {
                    "url": transformed_request["url"],
                    "headers": transformed_request["headers"],
                }

                # Only add data for non-GET requests
                if method != "get" and transformed_request.get("data") is not None:
                    request_kwargs["data"] = transformed_request["data"]

                batch_response = await getattr(async_httpx_client, method)(
                    **request_kwargs
                )
            elif isinstance(transformed_request, dict) and api_base:
                # For other providers that use JSON requests
                batch_response = await async_httpx_client.get(
                    url=api_base,
                    headers={**headers, "Content-Type": "application/json"},
                    params=transformed_request,
                )
            else:
                # Handle other request types if needed
                if not api_base:
                    raise ValueError("api_base is required for non-pre-signed requests")
                batch_response = await async_httpx_client.get(
                    url=api_base,
                    headers=headers,
                )
        except Exception as e:
            verbose_logger.exception(f"Error retrieving batch: {e}")
            raise self._handle_error(
                e=e,
                provider_config=provider_config,
            )

        return provider_config.transform_retrieve_batch_response(
            model=model,
            raw_response=batch_response,
            logging_obj=logging_obj,
            litellm_params=litellm_params,
        )

    def cancel_response_api_handler(
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
        shared_session: Optional["ClientSession"] = None,
    ) -> Union[ResponsesAPIResponse, Coroutine[Any, Any, ResponsesAPIResponse]]:
        """
        Async version of the responses API handler.
        Uses async HTTP client to make requests.
        """
        if _is_async:
            return self.async_cancel_response_api_handler(
                response_id=response_id,
                responses_api_provider_config=responses_api_provider_config,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client,
                shared_session=shared_session,
            )
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            headers=extra_headers or {}, model="None", litellm_params=litellm_params
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = responses_api_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        url, data = responses_api_provider_config.transform_cancel_response_api_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=response_id,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.post(
                url=url, headers=headers, json=data, timeout=timeout
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=responses_api_provider_config,
            )

        return responses_api_provider_config.transform_cancel_response_api_response(
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_cancel_response_api_handler(
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
        shared_session: Optional["ClientSession"] = None,
    ) -> ResponsesAPIResponse:
        """
        Async version of the cancel response API handler.
        Uses async HTTP client to make requests.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            verbose_logger.debug(
                f"Creating HTTP client for cancel_response with shared_session: {id(shared_session) if shared_session else None}"
            )
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
                shared_session=shared_session,
            )
        else:
            async_httpx_client = client

        headers = responses_api_provider_config.validate_environment(
            headers=extra_headers or {}, model="None", litellm_params=litellm_params
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = responses_api_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        url, data = responses_api_provider_config.transform_cancel_response_api_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=response_id,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.post(
                url=url, headers=headers, json=data, timeout=timeout
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=responses_api_provider_config,
            )

        return responses_api_provider_config.transform_cancel_response_api_response(
            raw_response=response,
            logging_obj=logging_obj,
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
            BaseConfig,
            BaseRerankConfig,
            BaseResponsesAPIConfig,
            BaseImageEditConfig,
            BaseImageGenerationConfig,
            BaseVectorStoreConfig,
            BaseVectorStoreFilesConfig,
            BaseGoogleGenAIGenerateContentConfig,
            BaseAnthropicMessagesConfig,
            BaseBatchesConfig,
            BaseOCRConfig,
            BaseVideoConfig,
            BaseSearchConfig,
            BaseTextToSpeechConfig,
            "BasePassthroughConfig",
            "BaseContainerConfig",
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

        if provider_config is None:
            from litellm.llms.base_llm.chat.transformation import BaseLLMException

            raise BaseLLMException(
                status_code=status_code,
                message=error_text,
                headers=error_headers,
            )

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
                url,
                extra_headers=headers,
                max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
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
    ) -> Union[
        ImageResponse,
        Coroutine[Any, Any, ImageResponse],
    ]:
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

    def image_generation_handler(
        self,
        model: str,
        prompt: str,
        image_generation_provider_config: BaseImageGenerationConfig,
        image_generation_optional_request_params: Dict,
        custom_llm_provider: str,
        litellm_params: Dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
        fake_stream: bool = False,
        litellm_metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> Union[
        ImageResponse,
        Coroutine[Any, Any, ImageResponse],
    ]:
        """
        Handles image generation requests.
        When _is_async=True, returns a coroutine instead of making the call directly.
        """
        if _is_async:
            # Return the async coroutine if called with _is_async=True
            return self.async_image_generation_handler(
                model=model,
                prompt=prompt,
                image_generation_provider_config=image_generation_provider_config,
                image_generation_optional_request_params=image_generation_optional_request_params,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
                fake_stream=fake_stream,
                litellm_metadata=litellm_metadata,
                api_key=api_key,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = image_generation_provider_config.validate_environment(
            api_key=api_key,
            headers=image_generation_optional_request_params.get("extra_headers", {})
            or {},
            model=model,
            messages=[],
            optional_params=image_generation_optional_request_params,
            litellm_params=dict(litellm_params),
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = image_generation_provider_config.get_complete_url(
            model=model,
            api_base=litellm_params.get("api_base", None),
            api_key=litellm_params.get("api_key", None),
            optional_params=image_generation_optional_request_params,
            litellm_params=dict(litellm_params),
        )

        data = image_generation_provider_config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=image_generation_optional_request_params,
            litellm_params=dict(litellm_params),
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
                json=data,
                timeout=timeout,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=image_generation_provider_config,
            )

        model_response: ImageResponse = (
            image_generation_provider_config.transform_image_generation_response(
                model=model,
                raw_response=response,
                model_response=litellm.ImageResponse(),
                logging_obj=logging_obj,
                request_data=data,
                optional_params=image_generation_optional_request_params,
                litellm_params=dict(litellm_params),
                encoding=None,
            )
        )

        return model_response

    async def async_image_generation_handler(
        self,
        model: str,
        prompt: str,
        image_generation_provider_config: BaseImageGenerationConfig,
        image_generation_optional_request_params: Dict,
        custom_llm_provider: str,
        litellm_params: Dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        fake_stream: bool = False,
        litellm_metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> ImageResponse:
        """
        Async version of the image generation handler.
        Uses async HTTP client to make requests.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = image_generation_provider_config.validate_environment(
            api_key=api_key,
            headers=image_generation_optional_request_params.get("extra_headers", {})
            or {},
            model=model,
            messages=[],
            optional_params=image_generation_optional_request_params,
            litellm_params=dict(litellm_params),
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = image_generation_provider_config.get_complete_url(
            model=model,
            api_base=litellm_params.get("api_base", None),
            api_key=litellm_params.get("api_key", None),
            optional_params=image_generation_optional_request_params,
            litellm_params=dict(litellm_params),
        )

        data = image_generation_provider_config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=image_generation_optional_request_params,
            litellm_params=dict(litellm_params),
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
                json=data,
                timeout=timeout,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=image_generation_provider_config,
            )

        model_response: ImageResponse = (
            image_generation_provider_config.transform_image_generation_response(
                model=model,
                raw_response=response,
                model_response=litellm.ImageResponse(),
                logging_obj=logging_obj,
                request_data=data,
                optional_params=image_generation_optional_request_params,
                litellm_params=dict(litellm_params),
                encoding=None,
            )
        )

        return model_response

    ###### VIDEO GENERATION HANDLER ######
    def video_generation_handler(
        self,
        model: str,
        prompt: str,
        video_generation_provider_config: BaseVideoConfig,
        video_generation_optional_request_params: Dict,
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
        api_key: Optional[str] = None,
    ) -> Union[
        VideoObject,
        Coroutine[Any, Any, VideoObject],
    ]:
        """
        Handles video generation requests.
        When _is_async=True, returns a coroutine instead of making the call directly.
        """
        if _is_async:
            # Return the async coroutine if called with _is_async=True
            return self.async_video_generation_handler(
                model=model,
                prompt=prompt,
                video_generation_provider_config=video_generation_provider_config,
                video_generation_optional_request_params=video_generation_optional_request_params,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
                fake_stream=fake_stream,
                litellm_metadata=litellm_metadata,
                api_key=api_key,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = video_generation_provider_config.validate_environment(
            api_key=api_key or litellm_params.get("api_key", None),
            headers=video_generation_optional_request_params.get("extra_headers", {})
            or {},
            model=model,
        )
        
        if extra_headers:
            headers.update(extra_headers)

        api_base = video_generation_provider_config.get_complete_url(
            model=model,
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        data, files, api_base = video_generation_provider_config.transform_video_create_request(
            model=model,
            prompt=prompt,
            video_create_optional_request_params=video_generation_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
            api_base=api_base,
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
            # Use JSON when no files, otherwise use form data with files
            if files and len(files) > 0:
                # Use multipart/form-data when files are present
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=timeout,
                )

            else:
                # Use JSON content type for POST requests without files
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout,
                )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=video_generation_provider_config,
            )

        return video_generation_provider_config.transform_video_create_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            request_data=data,
        )

    async def async_video_generation_handler(
        self,
        model: str,
        prompt: str,
        video_generation_provider_config: "BaseVideoConfig",
        video_generation_optional_request_params: Dict,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        fake_stream: bool = False,
        litellm_metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> VideoObject:
        """
        Async version of the video generation handler.
        Uses async HTTP client to make requests.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = video_generation_provider_config.validate_environment(
            api_key=api_key or litellm_params.get("api_key", None),
            headers=video_generation_optional_request_params.get("extra_headers", {})
            or {},
            model=model,
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = video_generation_provider_config.get_complete_url(
            model=model,
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        data, files, api_base = video_generation_provider_config.transform_video_create_request(
            model=model,
            prompt=prompt,
            api_base=api_base,
            video_create_optional_request_params=video_generation_optional_request_params,
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
            #Use JSON when no files, otherwise use form data with files
            if files is None or len(files) == 0:
                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout,
                )
            else:
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
                provider_config=video_generation_provider_config,
            )

        return video_generation_provider_config.transform_video_create_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            request_data=data,
        )

    ###### VIDEO CONTENT HANDLER ######
    def video_content_handler(
        self,
        video_id: str,
        video_content_provider_config: BaseVideoConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[bytes, Coroutine[Any, Any, bytes]]:
        """
        Handle video content download requests.
        """
        if _is_async:
            return self.async_video_content_handler(
                video_id=video_id,
                video_content_provider_config=video_content_provider_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                timeout=timeout,
                extra_headers=extra_headers,
                api_key=api_key,
                client=client,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = video_content_provider_config.validate_environment(
            headers=extra_headers or {},
            model="",
            api_key=api_key,
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = video_content_provider_config.get_complete_url(
            model="",
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, data = video_content_provider_config.transform_video_content_request(
            video_id=video_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        try:
            # Use POST if params contains data (e.g., Vertex AI fetchPredictOperation)
            # Otherwise use GET (e.g., OpenAI video content download)
            if data:
                response = sync_httpx_client.post(
                    url=url,
                    headers=headers,
                    json=data,
                )
            else:
                # Otherwise it's a GET request with query params
                response = sync_httpx_client.get(
                    url=url,
                    headers=headers,
                    params=data,
                )

            # Transform the response using the provider config
            return video_content_provider_config.transform_video_content_response(
                raw_response=response,
                logging_obj=logging_obj,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=video_content_provider_config,
            )

    async def async_video_content_handler(
        self,
        video_id: str,
        video_content_provider_config: BaseVideoConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> bytes:
        """
        Async version of the video content download handler.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = video_content_provider_config.validate_environment(
            headers=extra_headers or {},
            model="",
            api_key=api_key,
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = video_content_provider_config.get_complete_url(
            model="",
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, data = video_content_provider_config.transform_video_content_request(
            video_id=video_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        try:
            # Use POST if params contains data (e.g., Vertex AI fetchPredictOperation)
            # Otherwise use GET (e.g., OpenAI video content download)
            if data:
                response = await async_httpx_client.post(
                    url=url,
                    headers=headers,
                    json=data,
                )
            else:
                # Otherwise it's a GET request with query params
                response = await async_httpx_client.get(
                    url=url,
                    headers=headers,
                    params=data,
                )

            # Transform the response using the provider config
            return await video_content_provider_config.async_transform_video_content_response(
                raw_response=response,
                logging_obj=logging_obj,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=video_content_provider_config,
            )

    def video_remix_handler(
        self,
        video_id: str,
        prompt: str,
        video_remix_provider_config: BaseVideoConfig,
        custom_llm_provider: str,
        litellm_params,
        logging_obj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        _is_async: bool = False,
        client=None,
        api_key: Optional[str] = None,
    ):
        """
        Handler for video remix requests.
        When _is_async=True, returns a coroutine instead of making the call directly.
        """
        if _is_async:
            # Return the async coroutine if called with _is_async=True
            return self.async_video_remix_handler(
                video_id=video_id,
                prompt=prompt,
                video_remix_provider_config=video_remix_provider_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client,
                api_key=api_key,
            )

        # For sync calls, use sync HTTP client directly (like video_generation does)
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = video_remix_provider_config.validate_environment(
            api_key=api_key,
            headers=extra_headers or {},
            model="",
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = video_remix_provider_config.get_complete_url(
            model="",
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, data = video_remix_provider_config.transform_video_remix_request(
            video_id=video_id,
            prompt=prompt,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
            extra_body=extra_body,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
                "video_id": video_id,
            },
        )

        try:
            response = sync_httpx_client.post(
                url=url,
                headers=headers,
                json=data,
                timeout=timeout,
            )

            return video_remix_provider_config.transform_video_remix_response(
                raw_response=response,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=video_remix_provider_config,
            )

    async def async_video_remix_handler(
        self,
        video_id: str,
        prompt: str,
        video_remix_provider_config: BaseVideoConfig,
        custom_llm_provider: str,
        litellm_params,
        logging_obj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        client=None,
        api_key: Optional[str] = None,
    ):
        """
        Async version of the video remix handler.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = video_remix_provider_config.validate_environment(
            api_key=api_key,
            headers=extra_headers or {},
            model="",
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = video_remix_provider_config.get_complete_url(
            model="",
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, data = video_remix_provider_config.transform_video_remix_request(
            video_id=video_id,
            prompt=prompt,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
            extra_body=extra_body,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
                "video_id": video_id,
            },
        )

        try:
            response = await async_httpx_client.post(
                url=url,
                headers=headers,
                json=data,
                timeout=timeout,
            )

            return video_remix_provider_config.transform_video_remix_response(
                raw_response=response,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=video_remix_provider_config,
            )

    def video_list_handler(
        self,
        after: Optional[str],
        limit: Optional[int],
        order: Optional[str],
        video_list_provider_config,
        custom_llm_provider: str,
        litellm_params,
        logging_obj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        _is_async: bool = False,
        client=None,
        api_key: Optional[str] = None,
    ):
        """
        Handler for video list requests.
        """
        if _is_async:
            return self.async_video_list_handler(
                after=after,
                limit=limit,
                order=order,
                video_list_provider_config=video_list_provider_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=timeout,
                client=client,
                api_key=api_key,
            )
        else:
            # For sync calls, we'll use the async handler in a sync context
            import asyncio

            return asyncio.run(
                self.async_video_list_handler(
                    after=after,
                    limit=limit,
                    order=order,
                    video_list_provider_config=video_list_provider_config,
                    custom_llm_provider=custom_llm_provider,
                    litellm_params=litellm_params,
                    logging_obj=logging_obj,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    timeout=timeout,
                    client=client,
                )
            )

    async def async_video_list_handler(
        self,
        after: Optional[str],
        limit: Optional[int],
        order: Optional[str],
        video_list_provider_config: BaseVideoConfig,
        custom_llm_provider: str,
        litellm_params,
        logging_obj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        client=None,
        api_key: Optional[str] = None,
    ):
        """
        Async version of the video list handler.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = video_list_provider_config.validate_environment(
            api_key=api_key,
            headers=extra_headers or {},
            model="",
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = video_list_provider_config.get_complete_url(
            model="",
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, params = video_list_provider_config.transform_video_list_request(
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
            after=after,
            limit=limit,
            order=order,
            extra_query=extra_query,
        )

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "params": params,
            },
        )

        try:
            response = await async_httpx_client.get(
                url=url,
                headers=headers,
                params=params,
            )

            return video_list_provider_config.transform_video_list_response(
                raw_response=response,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=video_list_provider_config,
            )

    async def async_video_delete_handler(
        self,
        video_id: str,
        video_delete_provider_config: BaseVideoConfig,
        custom_llm_provider: str,
        litellm_params,
        logging_obj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        client=None,
        api_key: Optional[str] = None,
    ):
        """
        Async version of the video delete handler.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = video_delete_provider_config.validate_environment(
            api_key=api_key,
            headers=extra_headers or {},
            model="",
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = video_delete_provider_config.get_complete_url(
            model="",
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, data = video_delete_provider_config.transform_video_delete_request(
            video_id=video_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "video_id": video_id,
            },
        )

        try:
            response = await async_httpx_client.delete(
                url=url,
                headers=headers,
                timeout=timeout,
            )

            return video_delete_provider_config.transform_video_delete_response(
                raw_response=response,
                logging_obj=logging_obj,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=video_delete_provider_config,
            )

    def video_status_handler(
        self,
        video_id: str,
        video_status_provider_config: BaseVideoConfig,
        custom_llm_provider: str,
        litellm_params,
        logging_obj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        _is_async: bool = False,
        client=None,
        api_key: Optional[str] = None,
    ):
        """
        Handler for video status requests.
        When _is_async=True, returns a coroutine instead of making the call directly.
        """
        if _is_async:
            # Return the async coroutine if called with _is_async=True
            return self.async_video_status_handler(
                video_id=video_id,
                video_status_provider_config=video_status_provider_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client,
                api_key=api_key,
            )

        # For sync calls, use sync HTTP client directly (like video_generation does)
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = video_status_provider_config.validate_environment(
            api_key=api_key,
            headers=extra_headers or {},
            model="",
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = video_status_provider_config.get_complete_url(
            model="",
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, data = video_status_provider_config.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "video_id": video_id,
                "data": data,
            },
        )

        try:
            # Use POST if data is provided (e.g., Vertex AI fetchPredictOperation)
            # Otherwise use GET (e.g., OpenAI video status)
            if data:
                response = sync_httpx_client.post(
                    url=url,
                    headers=headers,
                    json=data,
                )
            else:
                response = sync_httpx_client.get(
                    url=url,
                    headers=headers,
                )

            return video_status_provider_config.transform_video_status_retrieve_response(
                raw_response=response,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=video_status_provider_config,
            )

    async def async_video_status_handler(
        self,
        video_id: str,
        video_status_provider_config: BaseVideoConfig,
        custom_llm_provider: str,
        litellm_params,
        logging_obj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        client=None,
        api_key: Optional[str] = None,
    ):
        """
        Async version of the video status handler.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = video_status_provider_config.validate_environment(
            api_key=api_key,
            headers=extra_headers or {},
            model="",
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = video_status_provider_config.get_complete_url(
            model="",
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, data = video_status_provider_config.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "video_id": video_id,
                "data": data,
            },
        )

        try:
            # Use POST if data is provided (e.g., Vertex AI fetchPredictOperation)
            # Otherwise use GET (e.g., OpenAI video status)
            if data:
                response = await async_httpx_client.post(
                    url=url,
                    headers=headers,
                    json=data,
                )
            else:
                response = await async_httpx_client.get(
                    url=url,
                    headers=headers,
                )
            return video_status_provider_config.transform_video_status_retrieve_response(
                raw_response=response,
                logging_obj=logging_obj,
                custom_llm_provider=custom_llm_provider,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=video_status_provider_config,
            )
    
    ###### CONTAINER HANDLER ######
    def container_create_handler(
        self,
        name: str,
        container_create_request_params: Dict,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Union[float, httpx.Timeout] = 600,
        _is_async: bool = False,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> Union["ContainerObject", Coroutine[Any, Any, "ContainerObject"]]:
        if _is_async:
            # Return the async coroutine if called with _is_async=True
            return self.async_container_create_handler(
                name=name,
                container_create_request_params=container_create_request_params,
                container_provider_config=container_provider_config,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
                client=client,
            )

        # For sync calls, use sync HTTP client
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        # Validate environment and get headers
        headers = container_provider_config.validate_environment(
            headers=extra_headers or {},
            api_key=litellm_params.get("api_key", None),
        )
        
        # Add Content-Type header for JSON requests
        headers["Content-Type"] = "application/json"

        if extra_headers:
            headers.update(extra_headers)

        # Get the complete URL for the request
        api_base = container_provider_config.get_complete_url(
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        data = container_provider_config.transform_container_create_request(
            name=name,
            container_create_optional_request_params=container_create_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=name,
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
                json=data,
                timeout=timeout,
            )

            return container_provider_config.transform_container_create_response(
                raw_response=response,
                logging_obj=logging_obj,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=container_provider_config,
            )
    
    async def async_container_create_handler(
        self,
        name: str,
        container_create_request_params: Dict,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Union[float, httpx.Timeout] = 600,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> "ContainerObject":
        # For async calls, use async HTTP client
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.OPENAI,
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        # Validate environment and get headers
        headers = container_provider_config.validate_environment(
            headers=extra_headers or {},
            api_key=litellm_params.get("api_key", None),
        )
        
        # Add Content-Type header for JSON requests
        headers["Content-Type"] = "application/json"

        if extra_headers:
            headers.update(extra_headers)

        # Get the complete URL for the request
        api_base = container_provider_config.get_complete_url(
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        data = container_provider_config.transform_container_create_request(
            name=name,
            container_create_optional_request_params=container_create_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=name,
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
                json=data,
                timeout=timeout,
            )

            return container_provider_config.transform_container_create_response(
                raw_response=response,
                logging_obj=logging_obj,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=container_provider_config,
            )
    
    def container_list_handler(
        self,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Union[float, httpx.Timeout] = 600,
        _is_async: bool = False,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> Union["ContainerListResponse", Coroutine[Any, Any, "ContainerListResponse"]]:
        if _is_async:
            # Return the async coroutine if called with _is_async=True
            return self.async_container_list_handler(
                container_provider_config=container_provider_config,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                after=after,
                limit=limit,
                order=order,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=timeout,
                client=client,
            )

        # For sync calls, use sync HTTP client
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        # Validate environment and get headers
        headers = container_provider_config.validate_environment(
            headers=extra_headers or {},
            api_key=litellm_params.get("api_key", None),
        )

        if extra_headers:
            headers.update(extra_headers)

        # Get the complete URL for the request
        api_base = container_provider_config.get_complete_url(
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, params = container_provider_config.transform_container_list_request(
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
            after=after,
            limit=limit,
            order=order,
            extra_query=extra_query,
        )

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "params": params,
            },
        )

        try:
            response = sync_httpx_client.get(
                url=url,
                headers=headers,
                params=params,
            )

            return container_provider_config.transform_container_list_response(
                raw_response=response,
                logging_obj=logging_obj,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=container_provider_config,
            )
    
    async def async_container_list_handler(
        self,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Union[float, httpx.Timeout] = 600,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> "ContainerListResponse":
        # For async calls, use async HTTP client
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.OPENAI,
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        # Validate environment and get headers
        headers = container_provider_config.validate_environment(
            headers=extra_headers or {},
            api_key=litellm_params.get("api_key", None),
        )

        if extra_headers:
            headers.update(extra_headers)

        # Get the complete URL for the request
        api_base = container_provider_config.get_complete_url(
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, params = container_provider_config.transform_container_list_request(
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
            after=after,
            limit=limit,
            order=order,
            extra_query=extra_query,
        )

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "params": params,
            },
        )

        try:
            response = await async_httpx_client.get(
                url=url,
                headers=headers,
                params=params,
            )

            return container_provider_config.transform_container_list_response(
                raw_response=response,
                logging_obj=logging_obj,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=container_provider_config,
            )
    
    def container_retrieve_handler(
        self,
        container_id: str,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Union[float, httpx.Timeout] = 600,
        _is_async: bool = False,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> Union["ContainerObject", Coroutine[Any, Any, "ContainerObject"]]:
        if _is_async:
            # Return the async coroutine if called with _is_async=True
            return self.async_container_retrieve_handler(
                container_id=container_id,
                container_provider_config=container_provider_config,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=timeout,
                client=client,
            )

        # For sync calls, use sync HTTP client
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        # Validate environment and get headers
        headers = container_provider_config.validate_environment(
            headers=extra_headers or {},
            api_key=litellm_params.get("api_key", None),
        )

        if extra_headers:
            headers.update(extra_headers)

        # Get the complete URL for the request
        api_base = container_provider_config.get_complete_url(
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, params = container_provider_config.transform_container_retrieve_request(
            container_id=container_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        
        # Add any extra query parameters
        if extra_query:
            params.update(extra_query)

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "params": params,
                "container_id": container_id,
            },
        )

        try:
            response = sync_httpx_client.get(
                url=url,
                headers=headers,
                params=params,
            )

            return container_provider_config.transform_container_retrieve_response(
                raw_response=response,
                logging_obj=logging_obj,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=container_provider_config,
            )
    
    async def async_container_retrieve_handler(
        self,
        container_id: str,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Union[float, httpx.Timeout] = 600,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> "ContainerObject":
        # For async calls, use async HTTP client
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.OPENAI,
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        # Validate environment and get headers
        headers = container_provider_config.validate_environment(
            headers=extra_headers or {},
            api_key=litellm_params.get("api_key", None),
        )

        if extra_headers:
            headers.update(extra_headers)

        # Get the complete URL for the request
        api_base = container_provider_config.get_complete_url(
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, params = container_provider_config.transform_container_retrieve_request(
            container_id=container_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        
        # Add any extra query parameters
        if extra_query:
            params.update(extra_query)

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "params": params,
                "container_id": container_id,
            },
        )

        try:
            response = await async_httpx_client.get(
                url=url,
                headers=headers,
                params=params,
            )

            return container_provider_config.transform_container_retrieve_response(
                raw_response=response,
                logging_obj=logging_obj,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=container_provider_config,
            )
    
    def container_delete_handler(
        self,
        container_id: str,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Union[float, httpx.Timeout] = 600,
        _is_async: bool = False,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> Union["DeleteContainerResult", Coroutine[Any, Any, "DeleteContainerResult"]]:
        if _is_async:
            # Return the async coroutine if called with _is_async=True
            return self.async_container_delete_handler(
                container_id=container_id,
                container_provider_config=container_provider_config,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=timeout,
                client=client,
            )

        # For sync calls, use sync HTTP client
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        # Validate environment and get headers
        headers = container_provider_config.validate_environment(
            headers=extra_headers or {},
            api_key=litellm_params.get("api_key", None),
        )

        if extra_headers:
            headers.update(extra_headers)

        # Get the complete URL for the request
        api_base = container_provider_config.get_complete_url(
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, params = container_provider_config.transform_container_delete_request(
            container_id=container_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        
        # Add any extra query parameters
        if extra_query:
            params.update(extra_query)

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "params": params,
                "container_id": container_id,
            },
        )

        try:
            response = sync_httpx_client.delete(
                url=url,
                headers=headers,
                params=params,
            )

            return container_provider_config.transform_container_delete_response(
                raw_response=response,
                logging_obj=logging_obj,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=container_provider_config,
            )
    
    async def async_container_delete_handler(
        self,
        container_id: str,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Union[float, httpx.Timeout] = 600,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> "DeleteContainerResult":
        # For async calls, use async HTTP client
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.OPENAI,
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        # Validate environment and get headers
        headers = container_provider_config.validate_environment(
            headers=extra_headers or {},
            api_key=litellm_params.get("api_key", None),
        )

        if extra_headers:
            headers.update(extra_headers)

        # Get the complete URL for the request
        api_base = container_provider_config.get_complete_url(
            api_base=litellm_params.get("api_base", None),
            litellm_params=dict(litellm_params),
        )

        # Transform the request using the provider config
        url, params = container_provider_config.transform_container_delete_request(
            container_id=container_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        
        # Add any extra query parameters
        if extra_query:
            params.update(extra_query)

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "params": params,
                "container_id": container_id,
            },
        )

        try:
            response = await async_httpx_client.delete(
                url=url,
                headers=headers,
                params=params,
            )

            return container_provider_config.transform_container_delete_response(
                raw_response=response,
                logging_obj=logging_obj,
            )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=container_provider_config,
            )

    ###### VECTOR STORE HANDLER ######
    async def async_vector_store_search_handler(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        vector_store_provider_config: BaseVectorStoreConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> VectorStoreSearchResponse:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = vector_store_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        (
            url,
            request_body,
        ) = vector_store_provider_config.transform_search_vector_store_request(
            vector_store_id=vector_store_id,
            query=query,
            vector_store_search_optional_params=vector_store_search_optional_params,
            api_base=api_base,
            litellm_logging_obj=logging_obj,
            litellm_params=dict(litellm_params),
        )
        all_optional_params: Dict[str, Any] = dict(litellm_params)
        all_optional_params.update(vector_store_search_optional_params or {})
        headers, signed_json_body = vector_store_provider_config.sign_request(
            headers=headers,
            optional_params=all_optional_params,
            request_data=request_body,
            api_base=url,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        request_data = (
            json.dumps(request_body) if signed_json_body is None else signed_json_body
        )

        try:

            response = await async_httpx_client.post(
                url=url,
                headers=headers,
                data=request_data,
                timeout=timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=vector_store_provider_config)

        return vector_store_provider_config.transform_search_vector_store_response(
            response=response,
            litellm_logging_obj=logging_obj,
        )

    def vector_store_search_handler(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        vector_store_provider_config: BaseVectorStoreConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[
        VectorStoreSearchResponse, Coroutine[Any, Any, VectorStoreSearchResponse]
    ]:
        if _is_async:
            return self.async_vector_store_search_handler(
                vector_store_id=vector_store_id,
                query=query,
                vector_store_search_optional_params=vector_store_search_optional_params,
                vector_store_provider_config=vector_store_provider_config,
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

        headers = vector_store_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        (
            url,
            request_body,
        ) = vector_store_provider_config.transform_search_vector_store_request(
            vector_store_id=vector_store_id,
            query=query,
            vector_store_search_optional_params=vector_store_search_optional_params,
            api_base=api_base,
            litellm_logging_obj=logging_obj,
            litellm_params=dict(litellm_params),
        )

        all_optional_params: Dict[str, Any] = dict(litellm_params)
        all_optional_params.update(vector_store_search_optional_params or {})

        headers, signed_json_body = vector_store_provider_config.sign_request(
            headers=headers,
            optional_params=all_optional_params,
            request_data=request_body,
            api_base=url,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        request_data = (
            json.dumps(request_body) if signed_json_body is None else signed_json_body
        )

        try:
            response = sync_httpx_client.post(
                url=url,
                headers=headers,
                data=request_data,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=vector_store_provider_config)

        return vector_store_provider_config.transform_search_vector_store_response(
            response=response,
            litellm_logging_obj=logging_obj,
        )

    async def async_vector_store_create_handler(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        vector_store_provider_config: BaseVectorStoreConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> VectorStoreCreateResponse:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = vector_store_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        (
            url,
            request_body,
        ) = vector_store_provider_config.transform_create_vector_store_request(
            vector_store_create_optional_params=vector_store_create_optional_params,
            api_base=api_base,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.post(
                url=url, headers=headers, json=request_body, timeout=timeout
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=vector_store_provider_config)

        return vector_store_provider_config.transform_create_vector_store_response(
            response=response,
        )

    def vector_store_create_handler(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        vector_store_provider_config: BaseVectorStoreConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[
        VectorStoreCreateResponse, Coroutine[Any, Any, VectorStoreCreateResponse]
    ]:
        if _is_async:
            return self.async_vector_store_create_handler(
                vector_store_create_optional_params=vector_store_create_optional_params,
                vector_store_provider_config=vector_store_provider_config,
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

        headers = vector_store_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),
        )

        (
            url,
            request_body,
        ) = vector_store_provider_config.transform_create_vector_store_request(
            vector_store_create_optional_params=vector_store_create_optional_params,
            api_base=api_base,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.post(
                url=url, headers=headers, json=request_body
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=vector_store_provider_config)

        return vector_store_provider_config.transform_create_vector_store_response(
            response=response,
        )

    #####################################################################
    ################ Vector Store Files HANDLERS ########################
    #####################################################################
    async def async_vector_store_file_create_handler(
        self,
        *,
        vector_store_id: str,
        create_request: VectorStoreFileCreateRequest,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> VectorStoreFileObject:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        request_dict = dict(create_request)
        if extra_body:
            request_dict.update(extra_body)

        (
            url,
            request_body,
        ) = vector_store_files_provider_config.transform_create_vector_store_file_request(
            vector_store_id=vector_store_id,
            create_request=cast(VectorStoreFileCreateRequest, request_dict),
            api_base=api_base,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.post(
                url=url, headers=headers, json=request_body, timeout=timeout
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_create_vector_store_file_response(
            response=response
        )

    def vector_store_file_create_handler(
        self,
        *,
        vector_store_id: str,
        create_request: VectorStoreFileCreateRequest,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[VectorStoreFileObject, Coroutine[Any, Any, VectorStoreFileObject]]:
        if _is_async:
            return self.async_vector_store_file_create_handler(
                vector_store_id=vector_store_id,
                create_request=create_request,
                vector_store_files_provider_config=vector_store_files_provider_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        request_dict = dict(create_request)
        if extra_body:
            request_dict.update(extra_body)

        (
            url,
            request_body,
        ) = vector_store_files_provider_config.transform_create_vector_store_file_request(
            vector_store_id=vector_store_id,
            create_request=cast(VectorStoreFileCreateRequest, request_dict),
            api_base=api_base,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.post(
                url=url, headers=headers, json=request_body, timeout=timeout
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_create_vector_store_file_response(
            response=response
        )

    async def async_vector_store_file_list_handler(
        self,
        *,
        vector_store_id: str,
        query_params: VectorStoreFileListQueryParams,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> VectorStoreFileListResponse:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )
        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        params_dict = dict(query_params)
        if extra_query:
            params_dict.update(extra_query)

        (
            url,
            request_params,
        ) = vector_store_files_provider_config.transform_list_vector_store_files_request(
            vector_store_id=vector_store_id,
            query_params=cast(VectorStoreFileListQueryParams, params_dict),
            api_base=api_base,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.get(
                url=url, headers=headers, params=request_params
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_list_vector_store_files_response(
            response=response
        )

    def vector_store_file_list_handler(
        self,
        *,
        vector_store_id: str,
        query_params: VectorStoreFileListQueryParams,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[
        VectorStoreFileListResponse, Coroutine[Any, Any, VectorStoreFileListResponse]
    ]:
        if _is_async:
            return self.async_vector_store_file_list_handler(
                vector_store_id=vector_store_id,
                query_params=query_params,
                vector_store_files_provider_config=vector_store_files_provider_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )
        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        params_dict = dict(query_params)
        if extra_query:
            params_dict.update(extra_query)

        (
            url,
            request_params,
        ) = vector_store_files_provider_config.transform_list_vector_store_files_request(
            vector_store_id=vector_store_id,
            query_params=cast(VectorStoreFileListQueryParams, params_dict),
            api_base=api_base,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.get(
                url=url, headers=headers, params=request_params
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_list_vector_store_files_response(
            response=response
        )

    async def async_vector_store_file_retrieve_handler(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> VectorStoreFileObject:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )
        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        url, request_params = (
            vector_store_files_provider_config.transform_retrieve_vector_store_file_request(
                vector_store_id=vector_store_id,
                file_id=file_id,
                api_base=api_base,
            )
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.get(
                url=url, headers=headers, params=request_params
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_retrieve_vector_store_file_response(
            response=response
        )

    def vector_store_file_retrieve_handler(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[VectorStoreFileObject, Coroutine[Any, Any, VectorStoreFileObject]]:
        if _is_async:
            return self.async_vector_store_file_retrieve_handler(
                vector_store_id=vector_store_id,
                file_id=file_id,
                vector_store_files_provider_config=vector_store_files_provider_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )
        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        url, request_params = (
            vector_store_files_provider_config.transform_retrieve_vector_store_file_request(
                vector_store_id=vector_store_id,
                file_id=file_id,
                api_base=api_base,
            )
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.get(
                url=url, headers=headers, params=request_params
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_retrieve_vector_store_file_response(
            response=response
        )

    async def async_vector_store_file_content_handler(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> VectorStoreFileContentResponse:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )
        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        url, request_params = (
            vector_store_files_provider_config.transform_retrieve_vector_store_file_content_request(
                vector_store_id=vector_store_id,
                file_id=file_id,
                api_base=api_base,
            )
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.get(
                url=url, headers=headers, params=request_params
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_retrieve_vector_store_file_content_response(
            response=response
        )

    def vector_store_file_content_handler(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[
        VectorStoreFileContentResponse,
        Coroutine[Any, Any, VectorStoreFileContentResponse],
    ]:
        if _is_async:
            return self.async_vector_store_file_content_handler(
                vector_store_id=vector_store_id,
                file_id=file_id,
                vector_store_files_provider_config=vector_store_files_provider_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )
        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        url, request_params = (
            vector_store_files_provider_config.transform_retrieve_vector_store_file_content_request(
                vector_store_id=vector_store_id,
                file_id=file_id,
                api_base=api_base,
            )
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.get(
                url=url, headers=headers, params=request_params
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_retrieve_vector_store_file_content_response(
            response=response
        )

    async def async_vector_store_file_update_handler(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        update_request: VectorStoreFileUpdateRequest,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> VectorStoreFileObject:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )
        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        request_dict = dict(update_request)
        if extra_body:
            request_dict.update(extra_body)

        (
            url,
            request_body,
        ) = vector_store_files_provider_config.transform_update_vector_store_file_request(
            vector_store_id=vector_store_id,
            file_id=file_id,
            update_request=cast(VectorStoreFileUpdateRequest, request_dict),
            api_base=api_base,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.post(
                url=url, headers=headers, json=request_body, timeout=timeout
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_update_vector_store_file_response(
            response=response
        )

    def vector_store_file_update_handler(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        update_request: VectorStoreFileUpdateRequest,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[VectorStoreFileObject, Coroutine[Any, Any, VectorStoreFileObject]]:
        if _is_async:
            return self.async_vector_store_file_update_handler(
                vector_store_id=vector_store_id,
                file_id=file_id,
                update_request=update_request,
                vector_store_files_provider_config=vector_store_files_provider_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )
        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        request_dict = dict(update_request)
        if extra_body:
            request_dict.update(extra_body)

        (
            url,
            request_body,
        ) = vector_store_files_provider_config.transform_update_vector_store_file_request(
            vector_store_id=vector_store_id,
            file_id=file_id,
            update_request=cast(VectorStoreFileUpdateRequest, request_dict),
            api_base=api_base,
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.post(
                url=url, headers=headers, json=request_body, timeout=timeout
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_update_vector_store_file_response(
            response=response
        )

    async def async_vector_store_file_delete_handler(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> VectorStoreFileDeleteResponse:
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )
        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        url, request_params = (
            vector_store_files_provider_config.transform_delete_vector_store_file_request(
                vector_store_id=vector_store_id,
                file_id=file_id,
                api_base=api_base,
            )
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.delete(
                url=url, headers=headers, params=request_params, timeout=timeout
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_delete_vector_store_file_response(
            response=response
        )

    def vector_store_file_delete_handler(
        self,
        *,
        vector_store_id: str,
        file_id: str,
        vector_store_files_provider_config: BaseVectorStoreFilesConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[
        VectorStoreFileDeleteResponse,
        Coroutine[Any, Any, VectorStoreFileDeleteResponse],
    ]:
        if _is_async:
            return self.async_vector_store_file_delete_handler(
                vector_store_id=vector_store_id,
                file_id=file_id,
                vector_store_files_provider_config=vector_store_files_provider_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = vector_store_files_provider_config.validate_environment(
            headers=extra_headers or {}, litellm_params=litellm_params
        )
        if extra_headers:
            headers.update(extra_headers)

        api_base = vector_store_files_provider_config.get_complete_url(
            api_base=litellm_params.api_base,
            vector_store_id=vector_store_id,
            litellm_params=dict(litellm_params),
        )

        url, request_params = (
            vector_store_files_provider_config.transform_delete_vector_store_file_request(
                vector_store_id=vector_store_id,
                file_id=file_id,
                api_base=api_base,
            )
        )

        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "complete_input_dict": request_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.delete(
                url=url, headers=headers, params=request_params, timeout=timeout
            )
        except Exception as e:
            raise self._handle_error(
                e=e, provider_config=vector_store_files_provider_config
            )

        return vector_store_files_provider_config.transform_delete_vector_store_file_response(
            response=response
        )

    #####################################################################
    ################ Google GenAI GENERATE CONTENT HANDLER ###########################
    #####################################################################
    def generate_content_handler(
        self,
        model: str,
        contents: Any,
        generate_content_provider_config: BaseGoogleGenAIGenerateContentConfig,
        generate_content_config_dict: Dict,
        tools: Any,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        _is_async: bool = False,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        stream: bool = False,
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Handles Google GenAI generate content requests.
        When _is_async=True, returns a coroutine instead of making the call directly.
        """
        from litellm.google_genai.streaming_iterator import (
            GoogleGenAIGenerateContentStreamingIterator,
        )

        if _is_async:
            return self.async_generate_content_handler(
                model=model,
                contents=contents,
                generate_content_provider_config=generate_content_provider_config,
                generate_content_config_dict=generate_content_config_dict,
                tools=tools,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
                stream=stream,
                litellm_metadata=litellm_metadata,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        # Get headers and URL from the provider config
        (
            headers,
            api_base,
        ) = generate_content_provider_config.sync_get_auth_token_and_url(
            api_base=litellm_params.api_base,
            model=model,
            litellm_params=dict(litellm_params),
            stream=stream,
        )

        if extra_headers:
            headers.update(extra_headers)

        # Get the request body from the provider config
        data = generate_content_provider_config.transform_generate_content_request(
            model=model,
            contents=contents,
            tools=tools,
            generate_content_config_dict=generate_content_config_dict,
        )

        if extra_body:
            data.update(extra_body)

        ## LOGGING
        logging_obj.pre_call(
            input=contents,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            if stream:
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout,
                    stream=True,
                )
                # Return streaming iterator
                return GoogleGenAIGenerateContentStreamingIterator(
                    response=response,
                    model=model,
                    logging_obj=logging_obj,
                    generate_content_provider_config=generate_content_provider_config,
                    litellm_metadata=litellm_metadata or {},
                    custom_llm_provider=custom_llm_provider,
                    request_body=data,
                )
            else:
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout,
                )
        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=generate_content_provider_config,
            )

        return generate_content_provider_config.transform_generate_content_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_generate_content_handler(
        self,
        model: str,
        contents: Any,
        generate_content_provider_config: BaseGoogleGenAIGenerateContentConfig,
        generate_content_config_dict: Dict,
        tools: Any,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
        stream: bool = False,
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Async version of the generate content handler.
        Uses async HTTP client to make requests.
        """
        from litellm.google_genai.streaming_iterator import (
            AsyncGoogleGenAIGenerateContentStreamingIterator,
        )

        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        # Get headers and URL from the provider config
        (
            headers,
            api_base,
        ) = await generate_content_provider_config.get_auth_token_and_url(
            model=model,
            litellm_params=dict(litellm_params),
            stream=stream,
            api_base=litellm_params.api_base,
        )

        if extra_headers:
            headers.update(extra_headers)

        # Get the request body from the provider config
        data = generate_content_provider_config.transform_generate_content_request(
            model=model,
            contents=contents,
            tools=tools,
            generate_content_config_dict=generate_content_config_dict,
        )

        if extra_body:
            data.update(extra_body)

        ## LOGGING
        logging_obj.pre_call(
            input=contents,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            if stream:
                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout,
                    stream=True,
                )
                # Return async streaming iterator
                return AsyncGoogleGenAIGenerateContentStreamingIterator(
                    response=response,
                    model=model,
                    logging_obj=logging_obj,
                    generate_content_provider_config=generate_content_provider_config,
                    litellm_metadata=litellm_metadata or {},
                    custom_llm_provider=custom_llm_provider,
                    request_body=data,
                )
            else:
                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout,
                )
        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=generate_content_provider_config,
            )

        return generate_content_provider_config.transform_generate_content_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

    #####################################################################
    ################ TEXT TO SPEECH HANDLER ###########################
    #####################################################################
    def text_to_speech_handler(
        self,
        model: str,
        input: str,
        voice: Optional[str],
        text_to_speech_provider_config: BaseTextToSpeechConfig,
        text_to_speech_optional_params: Dict,
        custom_llm_provider: str,
        litellm_params: Dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        _is_async: bool = False,
    ) -> Union[
        "HttpxBinaryResponseContent",
        Coroutine[Any, Any, "HttpxBinaryResponseContent"],
    ]:
        """
        Handles text-to-speech requests.
        When _is_async=True, returns a coroutine instead of making the call directly.
        """
        if _is_async:
            return self.async_text_to_speech_handler(
                model=model,
                input=input,
                voice=voice,
                text_to_speech_provider_config=text_to_speech_provider_config,
                text_to_speech_optional_params=text_to_speech_optional_params,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = text_to_speech_provider_config.validate_environment(
            api_key=litellm_params.get("api_key"),
            headers=extra_headers or {},
            model=model,
            api_base=litellm_params.get("api_base"),
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = text_to_speech_provider_config.get_complete_url(
            model=model,
            api_base=litellm_params.get("api_base"),
            litellm_params=litellm_params,
        )

        request_data = text_to_speech_provider_config.transform_text_to_speech_request(
            model=model,
            input=input,
            voice=voice,
            optional_params=text_to_speech_optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Merge provider-specific headers
        if "headers" in request_data:
            headers.update(request_data["headers"])

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": request_data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            # Determine request body type and send appropriately
            if "dict_body" in request_data:
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=request_data["dict_body"],
                    timeout=timeout,
                )
            elif "ssml_body" in request_data:
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    data=request_data["ssml_body"],
                    timeout=timeout,
                )
            else:
                raise ValueError(
                    "No body found in request_data. Must provide one of: dict_body, ssml_body, text_body, binary_body"
                )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=text_to_speech_provider_config,
            )

        return text_to_speech_provider_config.transform_text_to_speech_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_text_to_speech_handler(
        self,
        model: str,
        input: str,
        voice: Optional[str],
        text_to_speech_provider_config: BaseTextToSpeechConfig,
        text_to_speech_optional_params: Dict,
        custom_llm_provider: str,
        litellm_params: Dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> "HttpxBinaryResponseContent":
        """
        Async version of the text-to-speech handler.
        Uses async HTTP client to make requests.
        """
        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = text_to_speech_provider_config.validate_environment(
            api_key=litellm_params.get("api_key"),
            headers=extra_headers or {},
            model=model,
            api_base=litellm_params.get("api_base"),
        )

        if extra_headers:
            headers.update(extra_headers)

        api_base = text_to_speech_provider_config.get_complete_url(
            model=model,
            api_base=litellm_params.get("api_base"),
            litellm_params=litellm_params,
        )

        request_data = text_to_speech_provider_config.transform_text_to_speech_request(
            model=model,
            input=input,
            voice=voice,
            optional_params=text_to_speech_optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Merge provider-specific headers
        if "headers" in request_data:
            headers.update(request_data["headers"])

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": request_data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            # Determine request body type and send appropriately
            if "dict_body" in request_data:
                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=request_data["dict_body"],
                    timeout=timeout,
                )
            elif "ssml_body" in request_data:
                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    data=request_data["ssml_body"],
                    timeout=timeout,
                )
            else:
                raise ValueError(
                    "No body found in request_data. Must provide one of: dict_body, ssml_body, text_body, binary_body"
                )

        except Exception as e:
            raise self._handle_error(
                e=e,
                provider_config=text_to_speech_provider_config,
            )

        return text_to_speech_provider_config.transform_text_to_speech_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )