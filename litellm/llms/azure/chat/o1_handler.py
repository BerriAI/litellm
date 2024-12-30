"""
Handler file for calls to Azure OpenAI's o1 family of models

Written separately to handle faking streaming for o1 models.
"""

from typing import Optional, Union

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from ...openai.openai import OpenAIChatCompletion
from ..common_utils import get_azure_openai_client


class AzureOpenAIO1ChatCompletion(OpenAIChatCompletion):
    def _get_openai_client(
        self,
        is_async: bool,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = httpx.Timeout(None),
        max_retries: Optional[int] = 2,
        organization: Optional[str] = None,
        client: Optional[
            Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
        ] = None,
    ) -> Optional[
        Union[
            OpenAI,
            AsyncOpenAI,
            AzureOpenAI,
            AsyncAzureOpenAI,
        ]
    ]:

        # Override to use Azure-specific client initialization
        if isinstance(client, OpenAI) or isinstance(client, AsyncOpenAI):
            client = None

        return get_azure_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            api_version=api_version,
            client=client,
            _is_async=is_async,
        )
