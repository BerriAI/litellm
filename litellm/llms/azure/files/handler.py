from collections.abc import Coroutine
from typing import Any, Final, cast

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.file_deleted import FileDeleted

from litellm._logging import verbose_logger
from litellm.types.llms.openai import *

from ..common_utils import BaseAzureLLM


class AzureOpenAIFilesAPI(BaseAzureLLM):
    """
    AzureOpenAI methods to support for batches
    - create_file()
    - retrieve_file()
    - list_files()
    - delete_file()
    - file_content()
    - update_file()
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _prepare_create_file_data(
        create_file_data: CreateFileRequest,
    ) -> dict[str, Any]:
        """
        Prepare create_file_data for OpenAI SDK.

        Removes expires_after if None to match SDK's Omit pattern.
        SDK expects file_create_params.ExpiresAfter | Omit, but FileExpiresAfter works at runtime.
        """
        data: Final = dict(create_file_data)
        if data.get("expires_after") is None:
            data.pop("expires_after", None)
        return data

    async def acreate_file(
        self,
        create_file_data: CreateFileRequest,
        openai_client: AsyncAzureOpenAI | AsyncOpenAI,
    ) -> OpenAIFileObject:
        verbose_logger.debug("create_file_data=%s", create_file_data)
        response = await openai_client.files.create(**self._prepare_create_file_data(create_file_data))
        verbose_logger.debug("create_file_response=%s", response)
        return OpenAIFileObject(**response.model_dump())

    def create_file(
        self,
        _is_async: bool,
        create_file_data: CreateFileRequest,
        api_base: str | None,
        api_key: str | None,
        api_version: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = None,
        litellm_params: dict | None = None,
    ) -> OpenAIFileObject | Coroutine[Any, Any, OpenAIFileObject]:
        openai_client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = self.get_azure_openai_client(
            litellm_params=litellm_params or {},
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "AzureOpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, (AsyncAzureOpenAI, AsyncOpenAI)):
                raise ValueError(
                    "AzureOpenAI client is not an instance of AsyncAzureOpenAI. Make sure you passed an AsyncAzureOpenAI client."
                )
            return self.acreate_file(create_file_data=create_file_data, openai_client=openai_client)
        response: Final = cast(AzureOpenAI | OpenAI, openai_client).files.create(
            **self._prepare_create_file_data(create_file_data)
        )
        return OpenAIFileObject(**response.model_dump())

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        openai_client: AsyncAzureOpenAI | AsyncOpenAI,
    ) -> HttpxBinaryResponseContent:
        response: Final = await openai_client.files.content(**file_content_request)
        return HttpxBinaryResponseContent(response=response.response)

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        api_version: str | None = None,
        client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = None,
        litellm_params: dict | None = None,
    ) -> HttpxBinaryResponseContent | Coroutine[Any, Any, HttpxBinaryResponseContent]:
        openai_client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = self.get_azure_openai_client(
            litellm_params=litellm_params or {},
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "AzureOpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, (AsyncAzureOpenAI, AsyncOpenAI)):
                raise ValueError(
                    "AzureOpenAI client is not an instance of AsyncAzureOpenAI. Make sure you passed an AsyncAzureOpenAI client."
                )
            return self.afile_content(
                file_content_request=file_content_request,
                openai_client=openai_client,
            )
        response: Final = cast(AzureOpenAI | OpenAI, openai_client).files.content(**file_content_request)

        return HttpxBinaryResponseContent(response=response.response)

    async def aretrieve_file(
        self,
        file_id: str,
        openai_client: AsyncAzureOpenAI | AsyncOpenAI,
    ) -> FileObject:
        response: Final = await openai_client.files.retrieve(file_id=file_id)
        return response

    def retrieve_file(
        self,
        _is_async: bool,
        file_id: str,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        api_version: str | None = None,
        client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = None,
        litellm_params: dict | None = None,
    ):
        openai_client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = self.get_azure_openai_client(
            litellm_params=litellm_params or {},
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "AzureOpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, (AsyncAzureOpenAI, AsyncOpenAI)):
                raise ValueError(
                    "AzureOpenAI client is not an instance of AsyncAzureOpenAI. Make sure you passed an AsyncAzureOpenAI client."
                )
            return self.aretrieve_file(
                file_id=file_id,
                openai_client=openai_client,
            )
        response: Final = openai_client.files.retrieve(file_id=file_id)

        return response

    async def adelete_file(
        self,
        file_id: str,
        openai_client: AsyncAzureOpenAI | AsyncOpenAI,
    ) -> FileDeleted:
        response: Final = await openai_client.files.delete(file_id=file_id)

        if not isinstance(response, FileDeleted):  # azure returns an empty string
            return FileDeleted(id=file_id, deleted=True, object="file")
        return response

    def delete_file(
        self,
        _is_async: bool,
        file_id: str,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        organization: str | None = None,
        api_version: str | None = None,
        client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = None,
        litellm_params: dict | None = None,
    ):
        openai_client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = self.get_azure_openai_client(
            litellm_params=litellm_params or {},
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "AzureOpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, (AsyncAzureOpenAI, AsyncOpenAI)):
                raise ValueError(
                    "AzureOpenAI client is not an instance of AsyncAzureOpenAI. Make sure you passed an AsyncAzureOpenAI client."
                )
            return self.adelete_file(
                file_id=file_id,
                openai_client=openai_client,
            )
        response: Final = openai_client.files.delete(file_id=file_id)

        if not isinstance(response, FileDeleted):  # azure returns an empty string
            return FileDeleted(id=file_id, deleted=True, object="file")

        return response

    async def alist_files(
        self,
        openai_client: AsyncAzureOpenAI | AsyncOpenAI,
        purpose: str | None = None,
    ):
        if isinstance(purpose, str):
            response = await openai_client.files.list(purpose=purpose)
        else:
            response = await openai_client.files.list()
        return response

    def list_files(
        self,
        _is_async: bool,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        purpose: str | None = None,
        api_version: str | None = None,
        client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = None,
        litellm_params: dict | None = None,
    ):
        openai_client: AzureOpenAI | AsyncAzureOpenAI | OpenAI | AsyncOpenAI | None = self.get_azure_openai_client(
            litellm_params=litellm_params or {},
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            client=client,
            _is_async=_is_async,
        )
        if openai_client is None:
            raise ValueError(
                "AzureOpenAI client is not initialized. Make sure api_key is passed or OPENAI_API_KEY is set in the environment."
            )

        if _is_async is True:
            if not isinstance(openai_client, (AsyncAzureOpenAI, AsyncOpenAI)):
                raise ValueError(
                    "AzureOpenAI client is not an instance of AsyncAzureOpenAI. Make sure you passed an AsyncAzureOpenAI client."
                )
            return self.alist_files(
                purpose=purpose,
                openai_client=openai_client,
            )

        if isinstance(purpose, str):
            response = openai_client.files.list(purpose=purpose)
        else:
            response = openai_client.files.list()

        return response
