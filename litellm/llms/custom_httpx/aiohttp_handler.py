from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union, cast

import aiohttp
import httpx  # type: ignore
from aiohttp import ClientSession, FormData

import litellm
import litellm.litellm_core_utils
import litellm.types
import litellm.types.utils
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.base_llm.image_variations.transformation import (
    BaseImageVariationConfig,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
)
from litellm.types.llms.openai import FileTypes
from litellm.types.utils import HttpHandlerRequestFields, ImageResponse, LlmProviders
from litellm.utils import CustomStreamWrapper, ModelResponse, ProviderConfigManager

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

DEFAULT_TIMEOUT = 600


class BaseLLMAIOHTTPHandler:

    def __init__(self):
        self.client_session: Optional[aiohttp.ClientSession] = None

    def _get_async_client_session(
        self, dynamic_client_session: Optional[ClientSession] = None
    ) -> ClientSession:
        if dynamic_client_session:
            return dynamic_client_session
        elif self.client_session:
            return self.client_session
        else:
            # init client session, and then return new session
            self.client_session = aiohttp.ClientSession()
            return self.client_session

    async def _make_common_async_call(
        self,
        async_client_session: Optional[ClientSession],
        provider_config: BaseConfig,
        api_base: str,
        headers: dict,
        data: Optional[dict],
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        form_data: Optional[FormData] = None,
        stream: bool = False,
    ) -> aiohttp.ClientResponse:
        """Common implementation across stream + non-stream calls. Meant to ensure consistent error-handling."""
        max_retry_on_unprocessable_entity_error = (
            provider_config.max_retry_on_unprocessable_entity_error
        )

        response: Optional[aiohttp.ClientResponse] = None
        async_client_session = self._get_async_client_session(
            dynamic_client_session=async_client_session
        )

        for i in range(max(max_retry_on_unprocessable_entity_error, 1)):
            try:
                response = await async_client_session.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    data=form_data,
                )
                if not response.ok:
                    response.raise_for_status()
            except aiohttp.ClientResponseError as e:
                setattr(e, "text", e.message)
                raise self._handle_error(e=e, provider_config=provider_config)
            except Exception as e:
                raise self._handle_error(e=e, provider_config=provider_config)
            break

        if response is None:
            raise provider_config.get_error_class(
                error_message="No response from the API",
                status_code=422,
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
        stream: bool = False,
        files: Optional[dict] = None,
        content: Any = None,
        params: Optional[dict] = None,
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
                    data=data,  # do not json dump the data here. let the individual endpoint handle this.
                    timeout=timeout,
                    stream=stream,
                    files=files,
                    content=content,
                    params=params,
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
        client: Optional[ClientSession] = None,
    ):
        _response = await self._make_common_async_call(
            async_client_session=client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            data=data,
            timeout=timeout,
            litellm_params=litellm_params,
            stream=False,
        )
        _transformed_response = await provider_config.transform_response(  # type: ignore
            model=model,
            raw_response=_response,  # type: ignore
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
        )
        return _transformed_response

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
        headers: Optional[dict] = {},
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler, ClientSession]] = None,
    ):
        provider_config = ProviderConfigManager.get_provider_chat_config(
            model=model, provider=litellm.LlmProviders(custom_llm_provider)
        )
        # get config from model, custom llm provider
        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers or {},
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_base=api_base,
        )

        api_base = provider_config.get_complete_url(
            api_base=api_base,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
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
                    if client is not None and isinstance(client, ClientSession)
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
                data=data,
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

        response = self._make_common_sync_call(
            sync_httpx_client=sync_httpx_client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            timeout=timeout,
            litellm_params=litellm_params,
            data=data,
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
        data: dict,
        model: str,
        messages: list,
        logging_obj,
        litellm_params: dict,
        timeout: Union[float, httpx.Timeout],
        fake_stream: bool = False,
        client: Optional[HTTPHandler] = None,
    ) -> Tuple[Any, dict]:
        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client()
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
            timeout=timeout,
            litellm_params=litellm_params,
            stream=stream,
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

        return completion_stream, dict(response.headers)

    async def async_image_variations(
        self,
        client: Optional[ClientSession],
        provider_config: BaseImageVariationConfig,
        api_base: str,
        headers: dict,
        data: HttpHandlerRequestFields,
        timeout: float,
        litellm_params: dict,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str,
        model: Optional[str],
        image: FileTypes,
        optional_params: dict,
    ) -> ImageResponse:
        # create aiohttp form data if files in data
        form_data: Optional[FormData] = None
        if "files" in data and "data" in data:
            form_data = FormData()
            for k, v in data["files"].items():
                form_data.add_field(k, v[1], filename=v[0], content_type=v[2])

            for key, value in data["data"].items():
                form_data.add_field(key, value)

        _response = await self._make_common_async_call(
            async_client_session=client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            data=None if form_data is not None else cast(dict, data),
            form_data=form_data,
            timeout=timeout,
            litellm_params=litellm_params,
            stream=False,
        )

        ## LOGGING
        logging_obj.post_call(
            api_key=api_key,
            original_response=_response.text,
            additional_args={
                "headers": headers,
                "api_base": api_base,
            },
        )

        ## RESPONSE OBJECT
        return await provider_config.async_transform_response_image_variation(
            model=model,
            model_response=model_response,
            raw_response=_response,
            logging_obj=logging_obj,
            request_data=cast(dict, data),
            image=image,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=None,
            api_key=api_key,
        )

    def image_variations(
        self,
        model_response: ImageResponse,
        api_key: str,
        model: Optional[str],
        image: FileTypes,
        timeout: float,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        print_verbose: Optional[Callable] = None,
        api_base: Optional[str] = None,
        aimage_variation: bool = False,
        logger_fn=None,
        client=None,
        organization: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> ImageResponse:
        if model is None:
            raise ValueError("model is required for non-openai image variations")

        provider_config = ProviderConfigManager.get_provider_image_variation_config(
            model=model,  # openai defaults to dall-e-2
            provider=LlmProviders(custom_llm_provider),
        )

        if provider_config is None:
            raise ValueError(
                f"image variation provider not found: {custom_llm_provider}."
            )

        api_base = provider_config.get_complete_url(
            api_base=api_base,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=False,
        )

        headers = provider_config.validate_environment(
            api_key=api_key,
            headers=headers or {},
            model=model,
            messages=[{"role": "user", "content": "test"}],
            optional_params=optional_params,
            api_base=api_base,
        )

        data = provider_config.transform_request_image_variation(
            model=model,
            image=image,
            optional_params=optional_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input="",
            api_key=api_key,
            additional_args={
                "headers": headers,
                "api_base": api_base,
                "complete_input_dict": data.copy(),
            },
        )

        if litellm_params.get("async_call", False):
            return self.async_image_variations(
                api_base=api_base,
                data=data,
                headers=headers,
                model_response=model_response,
                api_key=api_key,
                logging_obj=logging_obj,
                model=model,
                timeout=timeout,
                client=client,
                optional_params=optional_params,
                litellm_params=litellm_params,
                image=image,
                provider_config=provider_config,
            )  # type: ignore

        if client is None or not isinstance(client, HTTPHandler):
            sync_httpx_client = _get_httpx_client()
        else:
            sync_httpx_client = client

        response = self._make_common_sync_call(
            sync_httpx_client=sync_httpx_client,
            provider_config=provider_config,
            api_base=api_base,
            headers=headers,
            timeout=timeout,
            litellm_params=litellm_params,
            stream=False,
            data=data.get("data") or {},
            files=data.get("files"),
            content=data.get("content"),
            params=data.get("params"),
        )

        ## LOGGING
        logging_obj.post_call(
            api_key=api_key,
            original_response=response.text,
            additional_args={
                "headers": headers,
                "api_base": api_base,
            },
        )

        ## RESPONSE OBJECT
        return provider_config.transform_response_image_variation(
            model=model,
            model_response=model_response,
            raw_response=response,
            logging_obj=logging_obj,
            request_data=cast(dict, data),
            image=image,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=None,
            api_key=api_key,
        )

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
