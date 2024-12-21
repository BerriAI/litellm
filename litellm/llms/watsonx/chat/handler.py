from typing import Callable, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.watsonx import WatsonXAIEndpoint, WatsonXAPIParams
from litellm.types.utils import CustomStreamingDecoder, ModelResponse

from ...openai_like.chat.handler import OpenAILikeChatHandler
from ..common_utils import WatsonXAIError, _generate_watsonx_token, _get_api_params


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
        logger_fn=None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        custom_endpoint: Optional[bool] = None,
        streaming_decoder: Optional[CustomStreamingDecoder] = None,
        fake_stream: bool = False,
    ):
        watsonx_token = _generate_watsonx_token(
            api_key=api_key, token=optional_params.pop("token", None)
        )
        if headers is None:
            headers = {}
        headers.update(
            {
                "Authorization": f"Bearer {watsonx_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

        stream: Optional[bool] = optional_params.get("stream", False)

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
