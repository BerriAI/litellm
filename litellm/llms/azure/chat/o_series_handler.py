"""
Handler file for calls to Azure OpenAI's o1/o3 family of models

Written separately to handle faking streaming for o1 and o3 models.
"""

from typing import Any, Callable, Optional, Union

import httpx

from litellm.types.utils import ModelResponse

from ...openai.openai import OpenAIChatCompletion
from ..common_utils import BaseAzureLLM


class AzureOpenAIO1ChatCompletion(BaseAzureLLM, OpenAIChatCompletion):
    def completion(
        self,
        model_response: ModelResponse,
        timeout: Union[float, httpx.Timeout],
        optional_params: dict,
        litellm_params: dict,
        logging_obj: Any,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        print_verbose: Optional[Callable] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        dynamic_params: Optional[bool] = None,
        azure_ad_token: Optional[str] = None,
        acompletion: bool = False,
        logger_fn=None,
        headers: Optional[dict] = None,
        custom_prompt_dict: dict = {},
        client=None,
        organization: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        drop_params: Optional[bool] = None,
    ):
        client = self.get_azure_openai_client(
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            client=client,
            _is_async=acompletion,
        )
        return super().completion(
            model_response=model_response,
            timeout=timeout,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            model=model,
            messages=messages,
            print_verbose=print_verbose,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            dynamic_params=dynamic_params,
            azure_ad_token=azure_ad_token,
            acompletion=acompletion,
            logger_fn=logger_fn,
            headers=headers,
            custom_prompt_dict=custom_prompt_dict,
            client=client,
            organization=organization,
            custom_llm_provider=custom_llm_provider,
            drop_params=drop_params,
        )
