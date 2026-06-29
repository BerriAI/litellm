"""
Moonshot AI (Kimi) Batches API Handler

Moonshot's batch API is OpenAI-compatible; this handler wires the OpenAI SDK
to Moonshot's endpoint (https://api.moonshot.ai/v1) using the caller-supplied
or environment-sourced API key / base URL.
"""

from typing import Any, Coroutine, Optional, Union, cast

import httpx
from openai import AsyncOpenAI, OpenAI

from litellm.llms.base import BaseLLM
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    CancelBatchRequest,
    CreateBatchRequest,
    RetrieveBatchRequest,
)
from litellm.types.utils import LiteLLMBatch

MOONSHOT_DEFAULT_API_BASE = "https://api.moonshot.ai/v1"


class MoonshotBatchesAPI(BaseLLM):
    """
    Moonshot AI batch operations — create, retrieve, cancel, list.
    All four map 1-to-1 to the OpenAI SDK because the Kimi batch API is
    OpenAI-compatible.
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_client(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        _is_async: bool,
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ) -> Optional[Union[OpenAI, AsyncOpenAI]]:
        if client is not None:
            return client

        resolved_key = api_key or get_secret_str("MOONSHOT_API_KEY")
        resolved_base = (
            api_base
            or get_secret_str("MOONSHOT_API_BASE")
            or MOONSHOT_DEFAULT_API_BASE
        )

        kwargs: dict = {"base_url": resolved_base}
        if resolved_key is not None:
            kwargs["api_key"] = resolved_key
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        if not isinstance(timeout, httpx.Timeout):
            kwargs["timeout"] = float(timeout)
        else:
            kwargs["timeout"] = timeout

        if _is_async:
            return AsyncOpenAI(**kwargs)
        return OpenAI(**kwargs)

    # ------------------------------------------------------------------ create

    async def acreate_batch(
        self,
        create_batch_data: CreateBatchRequest,
        client: AsyncOpenAI,
    ) -> LiteLLMBatch:
        response = await client.batches.create(**create_batch_data)  # type: ignore[arg-type]
        return LiteLLMBatch(**response.model_dump())

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        moonshot_client = self._get_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            _is_async=_is_async,
            client=client,
        )
        if moonshot_client is None:
            raise ValueError(
                "Moonshot client could not be initialised. "
                "Pass api_key or set MOONSHOT_API_KEY in the environment."
            )

        if _is_async:
            if not isinstance(moonshot_client, AsyncOpenAI):
                raise ValueError(
                    "Expected an AsyncOpenAI client for async create_batch."
                )
            return self.acreate_batch(
                create_batch_data=create_batch_data, client=moonshot_client
            )

        response = cast(OpenAI, moonshot_client).batches.create(**create_batch_data)  # type: ignore[arg-type]
        return LiteLLMBatch(**response.model_dump())

    # ---------------------------------------------------------------- retrieve

    async def aretrieve_batch(
        self,
        retrieve_batch_data: RetrieveBatchRequest,
        client: AsyncOpenAI,
    ) -> LiteLLMBatch:
        response = await client.batches.retrieve(**retrieve_batch_data)  # type: ignore[arg-type]
        return LiteLLMBatch(**response.model_dump())

    def retrieve_batch(
        self,
        _is_async: bool,
        retrieve_batch_data: RetrieveBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        moonshot_client = self._get_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            _is_async=_is_async,
            client=client,
        )
        if moonshot_client is None:
            raise ValueError(
                "Moonshot client could not be initialised. "
                "Pass api_key or set MOONSHOT_API_KEY in the environment."
            )

        if _is_async:
            if not isinstance(moonshot_client, AsyncOpenAI):
                raise ValueError(
                    "Expected an AsyncOpenAI client for async retrieve_batch."
                )
            return self.aretrieve_batch(
                retrieve_batch_data=retrieve_batch_data, client=moonshot_client
            )

        response = cast(OpenAI, moonshot_client).batches.retrieve(**retrieve_batch_data)  # type: ignore[arg-type]
        return LiteLLMBatch(**response.model_dump())

    # ------------------------------------------------------------------ cancel

    async def acancel_batch(
        self,
        cancel_batch_data: CancelBatchRequest,
        client: AsyncOpenAI,
    ) -> LiteLLMBatch:
        response = await client.batches.cancel(**cancel_batch_data)
        return LiteLLMBatch(**response.model_dump())

    def cancel_batch(
        self,
        _is_async: bool,
        cancel_batch_data: CancelBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        moonshot_client = self._get_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            _is_async=_is_async,
            client=client,
        )
        if moonshot_client is None:
            raise ValueError(
                "Moonshot client could not be initialised. "
                "Pass api_key or set MOONSHOT_API_KEY in the environment."
            )

        if _is_async:
            if not isinstance(moonshot_client, AsyncOpenAI):
                raise ValueError(
                    "Expected an AsyncOpenAI client for async cancel_batch."
                )
            return self.acancel_batch(
                cancel_batch_data=cancel_batch_data, client=moonshot_client
            )

        response = cast(OpenAI, moonshot_client).batches.cancel(**cancel_batch_data)
        return LiteLLMBatch(**response.model_dump())

    # -------------------------------------------------------------------- list

    async def alist_batches(
        self,
        client: AsyncOpenAI,
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
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        after: Optional[str] = None,
        limit: Optional[int] = None,
        client: Optional[Union[OpenAI, AsyncOpenAI]] = None,
    ):
        moonshot_client = self._get_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            _is_async=_is_async,
            client=client,
        )
        if moonshot_client is None:
            raise ValueError(
                "Moonshot client could not be initialised. "
                "Pass api_key or set MOONSHOT_API_KEY in the environment."
            )

        if _is_async:
            if not isinstance(moonshot_client, AsyncOpenAI):
                raise ValueError(
                    "Expected an AsyncOpenAI client for async list_batches."
                )
            return self.alist_batches(client=moonshot_client, after=after, limit=limit)

        return cast(OpenAI, moonshot_client).batches.list(after=after, limit=limit)  # type: ignore
