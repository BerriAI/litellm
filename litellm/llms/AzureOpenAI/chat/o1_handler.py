"""
Handler file for calls to Azure OpenAI's o1 family of models

Written separately to handle faking streaming for o1 models.
"""

import asyncio
from typing import Any, Callable, List, Optional, Union

from httpx._config import Timeout

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.bedrock.chat.invoke_handler import MockResponseIterator
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

from ..azure import AzureChatCompletion


class AzureOpenAIO1ChatCompletion(AzureChatCompletion):

    async def mock_async_streaming(
        self,
        response: Any,
        model: Optional[str],
        logging_obj: Any,
    ):
        model_response = await response
        completion_stream = MockResponseIterator(model_response=model_response)
        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="azure",
            logging_obj=logging_obj,
        )
        return streaming_response

    def completion(
        self,
        model: str,
        messages: List,
        model_response: ModelResponse,
        api_key: str,
        api_base: str,
        api_version: str,
        api_type: str,
        azure_ad_token: str,
        dynamic_params: bool,
        print_verbose: Callable[..., Any],
        timeout: Union[float, Timeout],
        logging_obj: Logging,
        optional_params,
        litellm_params,
        logger_fn,
        acompletion: bool = False,
        headers: Optional[dict] = None,
        client=None,
    ):
        stream: Optional[bool] = optional_params.pop("stream", False)
        response = super().completion(
            model,
            messages,
            model_response,
            api_key,
            api_base,
            api_version,
            api_type,
            azure_ad_token,
            dynamic_params,
            print_verbose,
            timeout,
            logging_obj,
            optional_params,
            litellm_params,
            logger_fn,
            acompletion,
            headers,
            client,
        )

        if stream is True:
            if asyncio.iscoroutine(response):
                return self.mock_async_streaming(
                    response=response, model=model, logging_obj=logging_obj  # type: ignore
                )

            completion_stream = MockResponseIterator(model_response=response)
            streaming_response = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=model,
                custom_llm_provider="openai",
                logging_obj=logging_obj,
            )

            return streaming_response
        else:
            return response
