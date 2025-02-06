"""
Azure Batches API Handler
"""

from typing import Any, Coroutine, Optional, Union

import httpx

import litellm
from litellm.llms.azure.azure import AsyncAzureOpenAI, AzureOpenAI
from litellm.types.llms.openai import (
    Batch,
    CancelBatchRequest,
    CreateBatchRequest,
    RetrieveBatchRequest,
)


class AzureBatchesAPI:
    """
    Azure methods to support for batches
    - create_batch()
    - retrieve_batch()
    - cancel_batch()
    - list_batch()
    """

    def __init__(self) -> None:
        super().__init__()

    def get_azure_openai_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        api_version: Optional[str] = None,
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
        _is_async: bool = False,
    ) -> Optional[Union[AzureOpenAI, AsyncAzureOpenAI]]:
        received_args = locals()
        openai_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None
        if client is None:
            data = {}
            for k, v in received_args.items():
                if k == "self" or k == "client" or k == "_is_async":
                    pass
                elif k == "api_base" and v is not None:
                    data["azure_endpoint"] = v
                elif v is not None:
                    data[k] = v
            if "api_version" not in data:
                data["api_version"] = litellm.AZURE_DEFAULT_API_VERSION
            if _is_async is True:
                openai_client = AsyncAzureOpenAI(**data)
            else:
                openai_client = AzureOpenAI(**data)  # type: ignore
        else:
            openai_client = client

        return openai_client

    async def acreate_batch(
        self,
        create_batch_data: CreateBatchRequest,
        azure_client: AsyncAzureOpenAI,
    ) -> Batch:
        response = await azure_client.batches.create(**create_batch_data)
        return response

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
    ) -> Union[Batch, Coroutine[Any, Any, Batch]]:
        azure_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = (
            self.get_azure_openai_client(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                api_version=api_version,
                max_retries=max_retries,
                client=client,
                _is_async=_is_async,
            )
        )
        if azure_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(azure_client, AsyncAzureOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.acreate_batch(  # type: ignore
                create_batch_data=create_batch_data, azure_client=azure_client
            )
        response = azure_client.batches.create(**create_batch_data)
        return response

    async def aretrieve_batch(
        self,
        retrieve_batch_data: RetrieveBatchRequest,
        client: AsyncAzureOpenAI,
    ) -> Batch:
        response = await client.batches.retrieve(**retrieve_batch_data)
        return response

    def retrieve_batch(
        self,
        _is_async: bool,
        retrieve_batch_data: RetrieveBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI] = None,
    ):
        azure_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = (
            self.get_azure_openai_client(
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                _is_async=_is_async,
            )
        )
        if azure_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(azure_client, AsyncAzureOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.aretrieve_batch(  # type: ignore
                retrieve_batch_data=retrieve_batch_data, client=azure_client
            )
        response = azure_client.batches.retrieve(**retrieve_batch_data)
        return response

    async def acancel_batch(
        self,
        cancel_batch_data: CancelBatchRequest,
        client: AsyncAzureOpenAI,
    ) -> Batch:
        response = await client.batches.cancel(**cancel_batch_data)
        return response

    def cancel_batch(
        self,
        _is_async: bool,
        cancel_batch_data: CancelBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[AzureOpenAI] = None,
    ):
        azure_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = (
            self.get_azure_openai_client(
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                timeout=timeout,
                max_retries=max_retries,
                client=client,
                _is_async=_is_async,
            )
        )
        if azure_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )
        response = azure_client.batches.cancel(**cancel_batch_data)
        return response

    async def alist_batches(
        self,
        client: AsyncAzureOpenAI,
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
        client: Optional[AzureOpenAI] = None,
    ):
        azure_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = (
            self.get_azure_openai_client(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                api_version=api_version,
                client=client,
                _is_async=_is_async,
            )
        )
        if azure_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(azure_client, AsyncAzureOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.alist_batches(  # type: ignore
                client=azure_client, after=after, limit=limit
            )
        response = azure_client.batches.list(after=after, limit=limit)  # type: ignore
        return response
