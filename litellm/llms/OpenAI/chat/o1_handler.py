"""
Handler file for calls to OpenAI's o1 family of models

Written separately to handle faking streaming for o1 models.
"""

import asyncio
from typing import Any, Callable, List, Optional, Union

from httpx._config import Timeout

from litellm.llms.bedrock.chat.invoke_handler import MockResponseIterator
from litellm.llms.OpenAI.openai import OpenAIChatCompletion
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper


class OpenAIO1ChatCompletion(OpenAIChatCompletion):

    def completion(
        self,
        model_response: ModelResponse,
        timeout: Union[float, Timeout],
        optional_params: dict,
        logging_obj: Any,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        print_verbose: Optional[Callable[..., Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        acompletion: bool = False,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[dict] = None,
        custom_prompt_dict: dict = {},
        client=None,
        organization: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        drop_params: Optional[bool] = None,
    ):
        # stream: Optional[bool] = optional_params.pop("stream", False)
        response = super().completion(
            model_response,
            timeout,
            optional_params,
            logging_obj,
            model,
            messages,
            print_verbose,
            api_key,
            api_base,
            acompletion,
            litellm_params,
            logger_fn,
            headers,
            custom_prompt_dict,
            client,
            organization,
            custom_llm_provider,
            drop_params,
        )

        return response
