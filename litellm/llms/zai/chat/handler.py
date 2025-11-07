"""
Handles the chat completion request for ZAI
"""

from typing import Callable, List, Optional, Union, cast, Any
from httpx._config import Timeout

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CustomStreamingDecoder
from litellm.utils import ModelResponse

from ...openai_like.chat.handler import OpenAILikeChatHandler


class ZaiChatCompletion(OpenAILikeChatHandler):
    def __init__(self, **kwargs: Any):
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
        timeout: Optional[Union[float, Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        custom_endpoint: Optional[bool] = None,
        streaming_decoder: Optional[CustomStreamingDecoder] = None,
        fake_stream: bool = False,
    ):
        # Handle ZAI-specific reasoning (preserve without interfering with tools)
        if litellm_params and litellm_params.get("reasoning_tokens", False):
            extra_body = optional_params.get("extra_body", {})
            extra_body["thinking"] = {"type": "enabled"}  # ZAI's reasoning parameter
            optional_params["extra_body"] = extra_body
            litellm_params.pop("reasoning_tokens", None)

        # ZAI model prefix strip
        model = model.replace("zai/", "")

        # Use default ZAI API base if not provided (without /chat/completions)
        if not api_base:
            api_base = "https://open.bigmodel.cn/api/paas/v4"

        # Delegate to OpenAI-like handler for tool calling and HTTP
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
            custom_endpoint=custom_endpoint,
            streaming_decoder=streaming_decoder,
            fake_stream=fake_stream,
        )