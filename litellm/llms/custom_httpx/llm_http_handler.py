import copy
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import httpx  # type: ignore
from openai.types.chat.chat_completion_chunk import Choice as OpenAIStreamingChoice

import litellm
import litellm.litellm_core_utils
import litellm.types
import litellm.types.utils
from litellm import verbose_logger
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.utils import CustomStreamWrapper, ModelResponse, ProviderConfigManager

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseLLMHTTPHandler:
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
    ):
        if client is None:
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider)
            )
        else:
            async_httpx_client = client

        try:
            response = await async_httpx_client.post(
                url=api_base,
                headers=headers,
                data=json.dumps(data),
                timeout=timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=provider_config)
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
        headers={},
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ):
        provider_config = ProviderConfigManager.get_provider_chat_config(
            model=model, provider=litellm.LlmProviders(custom_llm_provider)
        )
        # get config from model, custom llm provider
        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
        )

        api_base = provider_config.get_complete_url(
            api_base=api_base,
            model=model,
        )

        data = provider_config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
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
                if fake_stream is not True:
                    data["stream"] = stream
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
                )

        if stream is True:
            if fake_stream is not True:
                data["stream"] = stream
            completion_stream, headers = self.make_sync_call(
                provider_config=provider_config,
                api_base=api_base,
                headers=headers,  # type: ignore
                data=json.dumps(data),
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
            )
            return CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
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
        )

    def make_sync_call(
        self,
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: str,
        model: str,
        messages: list,
        logging_obj,
        timeout: Optional[Union[float, httpx.Timeout]],
        fake_stream: bool = False,
        client: Optional[HTTPHandler] = None,
    ) -> Tuple[Any, httpx.Headers]:
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client()
        else:
            sync_httpx_client = client
        try:
            stream = True
            if fake_stream is True:
                stream = False
            response = sync_httpx_client.post(
                api_base, headers=headers, data=data, timeout=timeout, stream=stream
            )
        except httpx.HTTPStatusError as e:
            raise self._handle_error(
                e=e,
                provider_config=provider_config,
            )
        except Exception as e:
            for exception in litellm.LITELLM_EXCEPTION_TYPES:
                if isinstance(e, exception):
                    raise e
            raise self._handle_error(
                e=e,
                provider_config=provider_config,
            )

        if response.status_code != 200:
            raise BaseLLMException(
                status_code=response.status_code,
                message=str(response.read()),
            )

        if fake_stream is True:
            completion_stream = provider_config.get_model_response_iterator(
                streaming_response=response.json(), sync_stream=True
            )
        else:
            completion_stream = provider_config.get_model_response_iterator(
                streaming_response=response.iter_lines(), sync_stream=True
            )

        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        return completion_stream, response.headers

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
        fake_stream: bool = False,
        client: Optional[AsyncHTTPHandler] = None,
    ):
        completion_stream, _response_headers = await self.make_async_call(
            custom_llm_provider=custom_llm_provider,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            data=json.dumps(data),
            messages=messages,
            logging_obj=logging_obj,
            timeout=timeout,
            fake_stream=fake_stream,
            client=client,
        )
        streamwrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )
        return streamwrapper

    async def make_async_call(
        self,
        custom_llm_provider: str,
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: str,
        messages: list,
        logging_obj: LiteLLMLoggingObj,
        timeout: Optional[Union[float, httpx.Timeout]],
        fake_stream: bool = False,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Tuple[Any, httpx.Headers]:
        if client is None:
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider)
            )
        else:
            async_httpx_client = client
        stream = True
        if fake_stream is True:
            stream = False
        try:
            response = await async_httpx_client.post(
                api_base, headers=headers, data=data, stream=stream, timeout=timeout
            )
        except httpx.HTTPStatusError as e:
            raise self._handle_error(
                e=e,
                provider_config=provider_config,
            )
        except Exception as e:
            for exception in litellm.LITELLM_EXCEPTION_TYPES:
                if isinstance(e, exception):
                    raise e
            raise self._handle_error(
                e=e,
                provider_config=provider_config,
            )

        if response.status_code != 200:
            raise BaseLLMException(
                status_code=response.status_code,
                message=str(response.read()),
            )
        if fake_stream is True:
            completion_stream = provider_config.get_model_response_iterator(
                streaming_response=response.json(), sync_stream=False
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

    def _handle_error(self, e: Exception, provider_config: BaseConfig):
        status_code = getattr(e, "status_code", 500)
        error_headers = getattr(e, "headers", None)
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

    def embedding(self):
        pass
