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
import requests  # type: ignore
from openai.types.chat.chat_completion_chunk import Choice as OpenAIStreamingChoice

import litellm
import litellm.litellm_core_utils
import litellm.types
import litellm.types.utils
from litellm import verbose_logger
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.transformation import BaseConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.utils import ModelResponse, ProviderConfigManager

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
        api_key: str,
        messages: list,
        optional_params: dict,
        encoding: str,
    ):
        async_httpx_client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders(custom_llm_provider)
        )
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
            httpx_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
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
        api_key: str,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        acompletion: bool,
        logger_fn=None,
        headers={},
        client=None,
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
                encoding=encoding,
            )

        if client is None or not isinstance(client, HTTPHandler):
            client = HTTPHandler(timeout=timeout)  # type: ignore
        else:
            client = client

        try:
            response = client.post(
                api_base,
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
            httpx_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            encoding=encoding,
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
        raise provider_config.error_class(  # type: ignore
            message=error_text,
            status_code=status_code,
            headers=error_headers,
        )

    def embedding(self):
        pass
