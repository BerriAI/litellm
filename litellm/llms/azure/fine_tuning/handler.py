from typing import Optional, Union

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from litellm.llms.azure.files.handler import get_azure_openai_client
from litellm.llms.openai.fine_tuning.handler import OpenAIFineTuningAPI


class AzureOpenAIFineTuningAPI(OpenAIFineTuningAPI):
    """
    AzureOpenAI methods to support fine tuning, inherits from OpenAIFineTuningAPI.
    """

    def get_openai_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[
            Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
        ] = None,
        _is_async: bool = False,
        api_version: Optional[str] = None,
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
            _is_async=_is_async,
        )
