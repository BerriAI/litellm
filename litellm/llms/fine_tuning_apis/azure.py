from typing import Any, Coroutine, Optional, Union

import httpx
from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.pagination import AsyncCursorPage
from openai.types.fine_tuning import FineTuningJob

from litellm._logging import verbose_logger
from litellm.llms.base import BaseLLM
from litellm.llms.files_apis.azure import get_azure_openai_client
from litellm.types.llms.openai import FineTuningJobCreate


class AzureOpenAIFineTuningAPI(BaseLLM):
    """
    AzureOpenAI methods to support for batches
    """

    def __init__(self) -> None:
        super().__init__()

    async def acreate_fine_tuning_job(
        self,
        create_fine_tuning_job_data: dict,
        openai_client: AsyncAzureOpenAI,
    ) -> FineTuningJob:
        response = await openai_client.fine_tuning.jobs.create(
            **create_fine_tuning_job_data  # type: ignore
        )
        return response

    def create_fine_tuning_job(
        self,
        _is_async: bool,
        create_fine_tuning_job_data: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str] = None,
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
        api_version: Optional[str] = None,
    ) -> Union[FineTuningJob, Coroutine[Any, Any, FineTuningJob]]:
        openai_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = (
            get_azure_openai_client(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                api_version=api_version,
                client=client,
                _is_async=_is_async,
            )
        )
        if openai_client is None:
            raise ValueError(
                "AzureOpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncAzureOpenAI):
                raise ValueError(
                    "AzureOpenAI client is not an instance of AsyncAzureOpenAI. Make sure you passed an AsyncAzureOpenAI client."
                )
            return self.acreate_fine_tuning_job(  # type: ignore
                create_fine_tuning_job_data=create_fine_tuning_job_data,
                openai_client=openai_client,
            )
        verbose_logger.debug(
            "creating fine tuning job, args= %s", create_fine_tuning_job_data
        )
        response = openai_client.fine_tuning.jobs.create(**create_fine_tuning_job_data)  # type: ignore
        return response

    async def acancel_fine_tuning_job(
        self,
        fine_tuning_job_id: str,
        openai_client: AsyncAzureOpenAI,
    ) -> FineTuningJob:
        response = await openai_client.fine_tuning.jobs.cancel(
            fine_tuning_job_id=fine_tuning_job_id
        )
        return response

    def cancel_fine_tuning_job(
        self,
        _is_async: bool,
        fine_tuning_job_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str] = None,
        api_version: Optional[str] = None,
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
    ):
        openai_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = (
            get_azure_openai_client(
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
                _is_async=_is_async,
            )
        )
        if openai_client is None:
            raise ValueError(
                "AzureOpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncAzureOpenAI):
                raise ValueError(
                    "AzureOpenAI client is not an instance of AsyncAzureOpenAI. Make sure you passed an AsyncAzureOpenAI client."
                )
            return self.acancel_fine_tuning_job(  # type: ignore
                fine_tuning_job_id=fine_tuning_job_id,
                openai_client=openai_client,
            )
        verbose_logger.debug("canceling fine tuning job, args= %s", fine_tuning_job_id)
        response = openai_client.fine_tuning.jobs.cancel(
            fine_tuning_job_id=fine_tuning_job_id
        )
        return response

    async def alist_fine_tuning_jobs(
        self,
        openai_client: AsyncAzureOpenAI,
        after: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        response = await openai_client.fine_tuning.jobs.list(after=after, limit=limit)  # type: ignore
        return response

    def list_fine_tuning_jobs(
        self,
        _is_async: bool,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str] = None,
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
        api_version: Optional[str] = None,
        after: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        openai_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = (
            get_azure_openai_client(
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                timeout=timeout,
                max_retries=max_retries,
                organization=organization,
                client=client,
                _is_async=_is_async,
            )
        )
        if openai_client is None:
            raise ValueError(
                "AzureOpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, AsyncAzureOpenAI):
                raise ValueError(
                    "AzureOpenAI client is not an instance of AsyncAzureOpenAI. Make sure you passed an AsyncAzureOpenAI client."
                )
            return self.alist_fine_tuning_jobs(  # type: ignore
                after=after,
                limit=limit,
                openai_client=openai_client,
            )
        verbose_logger.debug("list fine tuning job, after= %s, limit= %s", after, limit)
        response = openai_client.fine_tuning.jobs.list(after=after, limit=limit)  # type: ignore
        return response
