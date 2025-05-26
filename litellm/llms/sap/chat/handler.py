"""
SAP OpenAI-like chat completion handler with OAuth authentication
File: openai_like/chat/sap_handler.py
"""

from typing import Callable, Optional, Tuple, Union

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import CustomStreamingDecoder, ModelResponse

from ..common_utils import SAPOpenAILikeBase
from ...custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from ...openai_like.chat.handler import OpenAILikeChatHandler


class SAPOpenAILikeChatCompletion(BaseLLMHTTPHandler, SAPOpenAILikeBase):
    """
    OpenAI-like chat handler with SAP OAuth authentication support.
    Inherits from both OpenAILikeChatHandler for functionality and SAPOpenAILikeBase for auth.
    """

    def __init__(self, **kwargs):
        # Initialize both parent classes
        BaseLLMHTTPHandler.__init__(self)
        SAPOpenAILikeBase.__init__(self, **kwargs)

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
            stream: Optional[bool] = False,
            fake_stream: bool = False,
            litellm_params: dict = {},
            logger_fn=None,
            headers: Optional[dict] = None,
            timeout: Optional[Union[float, httpx.Timeout]] = None,
            client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
            custom_endpoint: Optional[bool] = None,
            streaming_decoder: Optional[CustomStreamingDecoder] = None
    ):
        """
        Override completion to handle SAP-specific parameters.
        """
        # Store original optional_params for validation
        validation_params = optional_params.copy()

        # Extract SAP-specific parameters before validation
        headers["AI-Resource-Group"] = "default"


        # Call parent completion method
        return super().completion(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            model_response=model_response,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            optional_params=optional_params,
            acompletion=acompletion,
            stream=stream,
            litellm_params=litellm_params,
            headers=headers,
            timeout=timeout,
            client=client,
            fake_stream=fake_stream,
        )
