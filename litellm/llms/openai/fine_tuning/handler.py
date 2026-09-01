from collections.abc import Coroutine
from typing import Any, Final, cast

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from litellm._logging import verbose_logger
from litellm.types.utils import LiteLLMFineTuningJob

_AZURE_STATUS_MAP: Final = {
    "pending": "queued",
    "notRunning": "queued",
    "running": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "canceled": "cancelled",
    "canceling": "cancelled",
}
# Note: Azure's "canceling" (in-progress) is mapped to "cancelled" (terminal)
# because LiteLLMFineTuningJob schema has no intermediate cancellation state.


def _normalize_fine_tuning_job_dict(data: dict[str, Any], is_azure: bool = False) -> dict[str, Any]:
    """
    Normalize Azure OpenAI FineTuningJob response to match OpenAI schema.

    Azure differences:
    - organization_id: null → ""
    - result_files: null → []
    - status: mapped via _AZURE_STATUS_MAP
    """
    if not is_azure:
        return data

    normalized: Final = data.copy()

    if normalized.get("organization_id") is None:
        normalized["organization_id"] = ""

    if normalized.get("result_files") is None:
        normalized["result_files"] = []

    status: Final = normalized.get("status")
    if status in _AZURE_STATUS_MAP:
        normalized["status"] = _AZURE_STATUS_MAP[status]

    return normalized


def _litellm_fine_tuning_job_from_response(response: Any, is_azure: bool = False) -> LiteLLMFineTuningJob:
    return LiteLLMFineTuningJob(**_normalize_fine_tuning_job_dict(response.model_dump(), is_azure=is_azure))


class OpenAIFineTuningAPI:
    """
    OpenAI methods to support for batches
    """

    def __init__(self) -> None:
        super().__init__()

    def get_openai_client(
        self,
        api_key: str | None,
        api_base: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None = None,
        _is_async: bool = False,
        api_version: str | None = None,
        litellm_params: dict | None = None,
    ) -> OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None:
        received_args: Final = locals()
        openai_client: OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None = None
        if client is None:
            data: Final = {}
            for k, v in received_args.items():
                if k == "self" or k == "client" or k == "_is_async":
                    pass
                elif k == "api_base" and v is not None:
                    data["base_url"] = v
                elif v is not None:
                    data[k] = v
            if _is_async is True:
                openai_client = AsyncOpenAI(**data)
            else:
                openai_client = OpenAI(**data)
        else:
            openai_client = client

        return openai_client

    async def acreate_fine_tuning_job(
        self,
        create_fine_tuning_job_data: dict,
        openai_client: AsyncOpenAI | AsyncAzureOpenAI,
    ) -> LiteLLMFineTuningJob:
        response: Final = await openai_client.fine_tuning.jobs.create(**create_fine_tuning_job_data)

        return _litellm_fine_tuning_job_from_response(response)

    def create_fine_tuning_job(
        self,
        _is_async: bool,
        create_fine_tuning_job_data: dict,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None = None,
    ) -> LiteLLMFineTuningJob | Coroutine[Any, Any, LiteLLMFineTuningJob]:
        openai_client: Final[OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
            api_version=api_version,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, (AsyncOpenAI, AsyncAzureOpenAI)):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.acreate_fine_tuning_job(
                create_fine_tuning_job_data=create_fine_tuning_job_data,
                openai_client=openai_client,
            )
        verbose_logger.debug("creating fine tuning job, args= %s", create_fine_tuning_job_data)
        response: Final = cast(OpenAI, openai_client).fine_tuning.jobs.create(**create_fine_tuning_job_data)
        return _litellm_fine_tuning_job_from_response(response)

    async def acancel_fine_tuning_job(
        self,
        fine_tuning_job_id: str,
        openai_client: AsyncOpenAI | AsyncAzureOpenAI,
    ) -> LiteLLMFineTuningJob:
        response: Final = await openai_client.fine_tuning.jobs.cancel(fine_tuning_job_id=fine_tuning_job_id)
        return _litellm_fine_tuning_job_from_response(response)

    def cancel_fine_tuning_job(
        self,
        _is_async: bool,
        fine_tuning_job_id: str,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None = None,
    ) -> LiteLLMFineTuningJob | Coroutine[Any, Any, LiteLLMFineTuningJob]:
        openai_client: Final[OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
            api_version=api_version,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, (AsyncOpenAI, AsyncAzureOpenAI)):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.acancel_fine_tuning_job(
                fine_tuning_job_id=fine_tuning_job_id,
                openai_client=openai_client,
            )
        verbose_logger.debug("canceling fine tuning job, args= %s", fine_tuning_job_id)
        response: Final = cast(OpenAI, openai_client).fine_tuning.jobs.cancel(fine_tuning_job_id=fine_tuning_job_id)
        return _litellm_fine_tuning_job_from_response(response)

    async def alist_fine_tuning_jobs(
        self,
        openai_client: AsyncOpenAI | AsyncAzureOpenAI,
        after: str | None = None,
        limit: int | None = None,
    ):
        response: Final = await openai_client.fine_tuning.jobs.list(after=after, limit=limit)
        return response

    def list_fine_tuning_jobs(
        self,
        _is_async: bool,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None = None,
        after: str | None = None,
        limit: int | None = None,
    ):
        openai_client: Final[OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
            api_version=api_version,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, (AsyncOpenAI, AsyncAzureOpenAI)):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.alist_fine_tuning_jobs(
                after=after,
                limit=limit,
                openai_client=openai_client,
            )
        verbose_logger.debug("list fine tuning job, after= %s, limit= %s", after, limit)
        response: Final = openai_client.fine_tuning.jobs.list(after=after, limit=limit)
        return response

    async def aretrieve_fine_tuning_job(
        self,
        fine_tuning_job_id: str,
        openai_client: AsyncOpenAI | AsyncAzureOpenAI,
    ) -> LiteLLMFineTuningJob:
        response: Final = await openai_client.fine_tuning.jobs.retrieve(fine_tuning_job_id=fine_tuning_job_id)
        return _litellm_fine_tuning_job_from_response(response)

    def retrieve_fine_tuning_job(
        self,
        _is_async: bool,
        fine_tuning_job_id: str,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None,
        client: OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None = None,
    ) -> LiteLLMFineTuningJob | Coroutine[Any, Any, LiteLLMFineTuningJob]:
        openai_client: Final[OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None] = self.get_openai_client(
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            organization=organization,
            client=client,
            _is_async=_is_async,
            api_version=api_version,
        )
        if openai_client is None:
            raise ValueError(
                "OpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncOpenAI):
                raise ValueError(
                    "OpenAI client is not an instance of AsyncOpenAI. Make sure you passed an AsyncOpenAI client."
                )
            return self.aretrieve_fine_tuning_job(
                fine_tuning_job_id=fine_tuning_job_id,
                openai_client=openai_client,
            )
        verbose_logger.debug("retrieving fine tuning job, id= %s", fine_tuning_job_id)
        response: Final = cast(OpenAI, openai_client).fine_tuning.jobs.retrieve(fine_tuning_job_id=fine_tuning_job_id)
        return _litellm_fine_tuning_job_from_response(response)
