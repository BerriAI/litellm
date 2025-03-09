from typing import Any, Coroutine, Optional, Union

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.fine_tuning import FineTuningJob

from litellm._logging import verbose_logger


class OpenAIFineTuningAPI:
    """
    OpenAI methods to support for batches
    """

    def __init__(self) -> None:
        super().__init__()

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
        received_args = locals()
        openai_client: Optional[
            Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
        ] = None
        if client is None:
            data = {}
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
                openai_client = OpenAI(**data)  # type: ignore
        else:
            openai_client = client

        return openai_client

    async def acreate_fine_tuning_job(
        self,
        create_fine_tuning_job_data: dict,
        openai_client: Union[AsyncOpenAI, AsyncAzureOpenAI],
    ) -> FineTuningJob:
        response = await openai_client.fine_tuning.jobs.create(
            **create_fine_tuning_job_data
        )
        return response

    def create_fine_tuning_job(
        self,
        _is_async: bool,
        create_fine_tuning_job_data: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[
            Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
        ] = None,
    ) -> Union[FineTuningJob, Coroutine[Any, Any, FineTuningJob]]:
        openai_client: Optional[
            Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
        ] = self.get_openai_client(
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
            return self.acreate_fine_tuning_job(  # type: ignore
                create_fine_tuning_job_data=create_fine_tuning_job_data,
                openai_client=openai_client,
            )
        verbose_logger.debug(
            "creating fine tuning job, args= %s", create_fine_tuning_job_data
        )
        response = openai_client.fine_tuning.jobs.create(**create_fine_tuning_job_data)
        return response

    async def acancel_fine_tuning_job(
        self,
        fine_tuning_job_id: str,
        openai_client: Union[AsyncOpenAI, AsyncAzureOpenAI],
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
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[
            Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
        ] = None,
    ):
        openai_client: Optional[
            Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
        ] = self.get_openai_client(
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
        openai_client: Union[AsyncOpenAI, AsyncAzureOpenAI],
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
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[
            Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
        ] = None,
        after: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        openai_client: Optional[
            Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
        ] = self.get_openai_client(
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
            return self.alist_fine_tuning_jobs(  # type: ignore
                after=after,
                limit=limit,
                openai_client=openai_client,
            )
        verbose_logger.debug("list fine tuning job, after= %s, limit= %s", after, limit)
        response = openai_client.fine_tuning.jobs.list(after=after, limit=limit)  # type: ignore
        return response

    async def aretrieve_fine_tuning_job(
        self,
        fine_tuning_job_id: str,
        openai_client: Union[AsyncOpenAI, AsyncAzureOpenAI],
    ) -> FineTuningJob:
        response = await openai_client.fine_tuning.jobs.retrieve(
            fine_tuning_job_id=fine_tuning_job_id
        )
        return response

    def retrieve_fine_tuning_job(
        self,
        _is_async: bool,
        fine_tuning_job_id: str,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[
            Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
        ] = None,
    ):
        openai_client: Optional[
            Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
        ] = self.get_openai_client(
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
            return self.aretrieve_fine_tuning_job(  # type: ignore
                fine_tuning_job_id=fine_tuning_job_id,
                openai_client=openai_client,
            )
        verbose_logger.debug("retrieving fine tuning job, id= %s", fine_tuning_job_id)
        response = openai_client.fine_tuning.jobs.retrieve(
            fine_tuning_job_id=fine_tuning_job_id
        )
        return response
