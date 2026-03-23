from typing import Any, Coroutine, Dict, Optional, Union, cast

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from litellm._logging import verbose_logger
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.fine_tuning.handler import (
    OpenAIFineTuningAPI,
    _litellm_fine_tuning_job_from_response,
)
from litellm.types.utils import LiteLLMFineTuningJob


class AzureOpenAIFineTuningAPI(OpenAIFineTuningAPI, BaseAzureLLM):
    """
    AzureOpenAI methods to support fine tuning, inherits from OpenAIFineTuningAPI.
    """

    @staticmethod
    def _ensure_training_type(create_fine_tuning_job_data: Dict[str, Any]) -> None:
        """
        Azure requires trainingType in extra_body. Default to 1 (supervised) if omitted.
        """
        extra_body = create_fine_tuning_job_data.get("extra_body") or {}
        if not isinstance(extra_body, dict):
            extra_body = {}
        if extra_body.get("trainingType") is None:
            extra_body["trainingType"] = 1
            create_fine_tuning_job_data["extra_body"] = extra_body
            verbose_logger.debug(
                "Azure fine-tuning: defaulting trainingType=1 (supervised)"
            )

    async def acreate_fine_tuning_job(
        self,
        create_fine_tuning_job_data: dict,
        openai_client: Union[AsyncOpenAI, AsyncAzureOpenAI],
    ) -> LiteLLMFineTuningJob:
        response = await openai_client.fine_tuning.jobs.create(
            **create_fine_tuning_job_data
        )
        return _litellm_fine_tuning_job_from_response(response, is_azure=True)

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
    ) -> Union[LiteLLMFineTuningJob, Coroutine[Any, Any, LiteLLMFineTuningJob]]:
        self._ensure_training_type(create_fine_tuning_job_data)

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
                "Azure OpenAI client is not initialized. Make sure api_key is passed or AZURE_API_KEY is set in the environment."
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

        verbose_logger.debug(
            "creating fine tuning job, args= %s", create_fine_tuning_job_data
        )
        response = cast(OpenAI, openai_client).fine_tuning.jobs.create(
            **create_fine_tuning_job_data
        )
        return _litellm_fine_tuning_job_from_response(response, is_azure=True)

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
    ) -> Optional[Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI,]]:
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
