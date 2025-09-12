from typing import Optional, Union

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.fine_tuning.handler import OpenAIFineTuningAPI


class AzureOpenAIFineTuningAPI(OpenAIFineTuningAPI, BaseAzureLLM):
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
        litellm_params: Optional[dict] = None,
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

        return self.get_azure_openai_client(
            litellm_params=litellm_params or {},
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            client=client,
            _is_async=_is_async,
        )
