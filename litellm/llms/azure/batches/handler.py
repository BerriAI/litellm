"""
Azure Batches API Handler
"""

from collections.abc import Coroutine
from typing import cast

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
        azure_client: AsyncAzureOpenAI | AsyncOpenAI,
    ) -> LiteLLMBatch:
        response = await azure_client.batches.create(**create_batch_data)  # type: ignore[arg-type]
        return LiteLLMBatch.model_validate(response.model_dump())

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = None,
        litellm_params: dict | None = None,
    ) -> LiteLLMBatch | Coroutine[object, object, LiteLLMBatch]:
        azure_client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = self.get_azure_openai_client(
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
        response = cast(AzureOpenAI | OpenAI, azure_client).batches.create(**create_batch_data)  # type: ignore[arg-type]
        return LiteLLMBatch.model_validate(response.model_dump())

    async def aretrieve_batch(
        self,
        retrieve_batch_data: RetrieveBatchRequest,
        client: AsyncAzureOpenAI | AsyncOpenAI,
    ) -> LiteLLMBatch:
        response = await client.batches.retrieve(**retrieve_batch_data)  # type: ignore[arg-type]
        return LiteLLMBatch.model_validate(response.model_dump())

    def retrieve_batch(
        self,
        _is_async: bool,
        retrieve_batch_data: RetrieveBatchRequest,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = None,
        litellm_params: dict | None = None,
    ):
        azure_client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = self.get_azure_openai_client(
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
        response = cast(AzureOpenAI | OpenAI, azure_client).batches.retrieve(**retrieve_batch_data)
        return LiteLLMBatch.model_validate(response.model_dump())

    async def acancel_batch(
        self,
        cancel_batch_data: CancelBatchRequest,
        client: AsyncAzureOpenAI | AsyncOpenAI,
    ) -> LiteLLMBatch:
        response = await client.batches.cancel(**cancel_batch_data)
        return LiteLLMBatch.model_validate(response.model_dump())

    def cancel_batch(
        self,
        _is_async: bool,
        cancel_batch_data: CancelBatchRequest,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = None,
        litellm_params: dict | None = None,
    ):
        azure_client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = self.get_azure_openai_client(
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
        return LiteLLMBatch.model_validate(response.model_dump())

    async def alist_batches(
        self,
        client: AsyncAzureOpenAI | AsyncOpenAI,
        after: str | None = None,
        limit: int | None = None,
    ):
        response = await client.batches.list(after=after, limit=limit)  # type: ignore
        return response

    def list_batches(
        self,
        _is_async: bool,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        after: str | None = None,
        limit: int | None = None,
        client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = None,
        litellm_params: dict | None = None,
    ):
        azure_client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = self.get_azure_openai_client(
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
