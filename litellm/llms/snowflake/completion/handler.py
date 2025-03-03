from litellm.llms.base import BaseLLM
from typing import Any, List, Optional
from typing import List, Dict, Callable, Optional, Any, cast, Union

import litellm
from litellm.utils import ModelResponse
from litellm.types.llms.openai import AllMessageValues
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler
from ..common_utils import SnowflakeBase

class SnowflakeChatCompletion(OpenAILikeChatHandler,SnowflakeBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        JWT: str,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[dict] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> None:
        
        messages = litellm.SnowflakeConfig()._transform_messages(
            messages=cast(List[AllMessageValues], messages), model=model
        )

        headers = self.validate_environment(
            headers,
            JWT
        )

        return super().completion(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_llm_provider= "snowflake",
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            api_key=JWT,
            logging_obj=logging_obj,
            optional_params=optional_params,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            headers=headers,
            client=client,
            custom_endpoint=True,
        )
