from typing import Callable, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import CustomStreamingDecoder, ModelResponse

from ...openai_like.chat.handler import OpenAILikeChatHandler
from ..common_utils import _get_api_params
from .transformation import IBMWatsonXChatConfig

watsonx_chat_transformation = IBMWatsonXChatConfig()


class WatsonXChatHandler(OpenAILikeChatHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def completion(
        self,
        *,
        model: str,
        messages: list,
        api_base: str,
        custom_llm_provider: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: Optional[str],
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        headers: Optional[dict] = None,
        logger_fn=None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        custom_endpoint: Optional[bool] = None,
        streaming_decoder: Optional[CustomStreamingDecoder] = None,
        fake_stream: bool = False,
    ):
        api_params = _get_api_params(params=optional_params)

        ## UPDATE HEADERS
        headers = watsonx_chat_transformation.validate_environment(
            headers=headers or {},
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
        )

        ## UPDATE PAYLOAD (optional params)
        watsonx_auth_payload = watsonx_chat_transformation._prepare_payload(
            model=model,
            api_params=api_params,
        )
        optional_params.update(watsonx_auth_payload)

        ## GET API URL
        api_base = watsonx_chat_transformation.get_complete_url(
            api_base=api_base,
            model=model,
            optional_params=optional_params,
            stream=optional_params.get("stream", False),
        )

        return super().completion(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            optional_params=optional_params,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            headers=headers,
            timeout=timeout,
            client=client,
            custom_endpoint=True,
            streaming_decoder=streaming_decoder,
        )
