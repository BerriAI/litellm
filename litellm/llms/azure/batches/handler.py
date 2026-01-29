"""
Azure Batches API Handler
"""

from typing import Any, Coroutine, Optional, Union, cast

import httpx
from openai import AsyncOpenAI, OpenAI

from litellm.llms.azure.azure import AsyncAzureOpenAI, AzureOpenAI
from litellm.types.llms.openai import (
    CancelBatchRequest,
    CreateBatchRequest,
    RetrieveBatchRequest,
)
from litellm.types.utils import LiteLLMBatch

from ..common_utils import BaseAzureLLM


class AzureBatchesAPI(BaseAzureLLM):
    """
    Azure methods to support for batches
    - create_batch()
    - retrieve_batch()
    - cancel_batch()
    - list_batch()
    """

    def __init__(self) -> None:
        super().__init__()

    async def acreate_batch(
        self,
        create_batch_data: CreateBatchRequest,
        azure_client: Union[AsyncAzureOpenAI, AsyncOpenAI],
    ) -> LiteLLMBatch:
        response = await azure_client.batches.create(**create_batch_data)
        return LiteLLMBatch(**response.model_dump())

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]] = None,
        litellm_params: Optional[dict] = None,
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        azure_client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = self.get_azure_openai_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            client=client,
            _is_async=_is_async,
            litellm_params=litellm_params or {},
        )
        if azure_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(azure_client, (AsyncAzureOpenAI, AsyncOpenAI)):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.acreate_batch(  # type: ignore
                create_batch_data=create_batch_data, azure_client=azure_client
            )
        response = cast(Union[AzureOpenAI, OpenAI], azure_client).batches.create(**create_batch_data)
        return LiteLLMBatch(**response.model_dump())

    async def aretrieve_batch(
        self,
        retrieve_batch_data: RetrieveBatchRequest,
        client: Union[AsyncAzureOpenAI, AsyncOpenAI],
    ) -> LiteLLMBatch:
        response = await client.batches.retrieve(**retrieve_batch_data)
        return LiteLLMBatch(**response.model_dump())

    def retrieve_batch(
        self,
        _is_async: bool,
        retrieve_batch_data: RetrieveBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]] = None,
        litellm_params: Optional[dict] = None,
    ):
        azure_client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = self.get_azure_openai_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            client=client,
            _is_async=_is_async,
            litellm_params=litellm_params or {},
        )
        if azure_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(azure_client, (AsyncAzureOpenAI, AsyncOpenAI)):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.aretrieve_batch(  # type: ignore
                retrieve_batch_data=retrieve_batch_data, client=azure_client
            )
        response = cast(Union[AzureOpenAI, OpenAI], azure_client).batches.retrieve(
            **retrieve_batch_data
        )
        return LiteLLMBatch(**response.model_dump())

    async def acancel_batch(
        self,
        cancel_batch_data: CancelBatchRequest,
        client: Union[AsyncAzureOpenAI, AsyncOpenAI],
    ) -> LiteLLMBatch:
        response = await client.batches.cancel(**cancel_batch_data)
        return LiteLLMBatch(**response.model_dump())

    def cancel_batch(
        self,
        _is_async: bool,
        cancel_batch_data: CancelBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]] = None,
        litellm_params: Optional[dict] = None,
    ):
        azure_client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = self.get_azure_openai_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            client=client,
            _is_async=_is_async,
            litellm_params=litellm_params or {},
        )
        if azure_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )
        
        if _is_async is True:
            if not isinstance(azure_client, (AsyncAzureOpenAI, AsyncOpenAI)):
                raise ValueError(
                    "Azure client is not an instance of AsyncAzureOpenAI or AsyncOpenAI. Make sure you passed an async client."
                )
            return self.acancel_batch(  # type: ignore
                cancel_batch_data=cancel_batch_data, client=azure_client
            )
        
        # At this point, azure_client is guaranteed to be a sync client
        if not isinstance(azure_client, (AzureOpenAI, OpenAI)):
            raise ValueError(
                "Azure client is not an instance of AzureOpenAI or OpenAI. Make sure you passed a sync client."
            )
        response = azure_client.batches.cancel(**cancel_batch_data)
        return LiteLLMBatch(**response.model_dump())

    async def alist_batches(
        self,
        client: Union[AsyncAzureOpenAI, AsyncOpenAI],
        after: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        response = await client.batches.list(after=after, limit=limit)  # type: ignore
        return response

    def list_batches(
        self,
        _is_async: bool,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        after: Optional[str] = None,
        limit: Optional[int] = None,
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]] = None,
        litellm_params: Optional[dict] = None,
    ):
        azure_client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = self.get_azure_openai_client(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            client=client,
            _is_async=_is_async,
            litellm_params=litellm_params or {},
        )
        if azure_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(azure_client, (AsyncAzureOpenAI, AsyncOpenAI)):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.alist_batches(  # type: ignore
                client=azure_client, after=after, limit=limit
            )
        response = azure_client.batches.list(after=after, limit=limit)  # type: ignore
        return response
