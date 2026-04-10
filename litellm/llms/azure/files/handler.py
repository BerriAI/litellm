
from typing import Any, AsyncIterator, Coroutine, Iterator, Optional, Union, cast

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
        data = dict(create_file_data)
        if data.get("expires_after") is None:
            data.pop("expires_after", None)
        return data

    async def acreate_file(
        self,
        create_file_data: CreateFileRequest,
        openai_client: Union[AsyncAzureOpenAI, AsyncOpenAI],
    ) -> OpenAIFileObject:
        verbose_logger.debug("create_file_data=%s", create_file_data)
        response = await openai_client.files.create(**self._prepare_create_file_data(create_file_data))  # type: ignore[arg-type]
        verbose_logger.debug("create_file_response=%s", response)
        return OpenAIFileObject(**response.model_dump())

    def create_file(
        self,
        _is_async: bool,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        api_key: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = None,
        litellm_params: Optional[dict] = None,
    ) -> Union[OpenAIFileObject, Coroutine[Any, Any, OpenAIFileObject]]:
        openai_client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = self.get_azure_openai_client(
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
            return self.acreate_file(
                create_file_data=create_file_data, openai_client=openai_client
            )
        response = cast(Union[AzureOpenAI, OpenAI], openai_client).files.create(**self._prepare_create_file_data(create_file_data))  # type: ignore[arg-type]
        return OpenAIFileObject(**response.model_dump())

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        openai_client: Union[AsyncAzureOpenAI, AsyncOpenAI],
    ) -> HttpxBinaryResponseContent:
        response = await openai_client.files.content(**file_content_request)
        return HttpxBinaryResponseContent(response=response.response)

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: Optional[str],
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        api_version: Optional[str] = None,
        client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = None,
        litellm_params: Optional[dict] = None,
    ) -> Union[
        HttpxBinaryResponseContent, Coroutine[Any, Any, HttpxBinaryResponseContent]
    ]:
        openai_client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = self.get_azure_openai_client(
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
            return self.afile_content(  # type: ignore
                file_content_request=file_content_request,
                openai_client=openai_client,
            )
        response = cast(Union[AzureOpenAI, OpenAI], openai_client).files.content(
            **file_content_request
        )

        return HttpxBinaryResponseContent(response=response.response)

    async def afile_content_streaming(
        self,
        file_content_request: FileContentRequest,
        openai_client: Union[AsyncAzureOpenAI, AsyncOpenAI],
        chunk_size: int = 1024 * 1024,
    ) -> AsyncIterator[bytes]:
        async with openai_client.files.with_streaming_response.content(
            **file_content_request
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=chunk_size):
                yield chunk

    def file_content_streaming(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: Optional[str],
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        api_version: Optional[str] = None,
        chunk_size: int = 1024 * 1024,
        client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = None,
        litellm_params: Optional[dict] = None,
    ) -> Union[Iterator[bytes], AsyncIterator[bytes]]:
        openai_client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = self.get_azure_openai_client(
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
            return self.afile_content_streaming(
                file_content_request=file_content_request,
                openai_client=openai_client,
                chunk_size=chunk_size,
            )

        def _stream() -> Iterator[bytes]:
            with cast(Union[AzureOpenAI, OpenAI], openai_client).files.with_streaming_response.content(
                **file_content_request
            ) as response:
                yield from response.iter_bytes(chunk_size=chunk_size)

        return _stream()

    async def aretrieve_file(
        self,
        file_id: str,
        openai_client: Union[AsyncAzureOpenAI, AsyncOpenAI],
    ) -> FileObject:
        response = await openai_client.files.retrieve(file_id=file_id)
        return response

    def retrieve_file(
        self,
        _is_async: bool,
        file_id: str,
        api_base: Optional[str],
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        api_version: Optional[str] = None,
        client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = None,
        litellm_params: Optional[dict] = None,
    ):
        openai_client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = self.get_azure_openai_client(
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
            return self.aretrieve_file(  # type: ignore
                file_id=file_id,
                openai_client=openai_client,
            )
        response = openai_client.files.retrieve(file_id=file_id)

        return response

    async def adelete_file(
        self,
        file_id: str,
        openai_client: Union[AsyncAzureOpenAI, AsyncOpenAI],
    ) -> FileDeleted:
        response = await openai_client.files.delete(file_id=file_id)

        if not isinstance(response, FileDeleted):  # azure returns an empty string
            return FileDeleted(id=file_id, deleted=True, object="file")
        return response

    def delete_file(
        self,
        _is_async: bool,
        file_id: str,
        api_base: Optional[str],
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str] = None,
        api_version: Optional[str] = None,
        client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = None,
        litellm_params: Optional[dict] = None,
    ):
        openai_client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = self.get_azure_openai_client(
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
            return self.adelete_file(  # type: ignore
                file_id=file_id,
                openai_client=openai_client,
            )
        response = openai_client.files.delete(file_id=file_id)

        if not isinstance(response, FileDeleted):  # azure returns an empty string
            return FileDeleted(id=file_id, deleted=True, object="file")

        return response

    async def alist_files(
        self,
        openai_client: Union[AsyncAzureOpenAI, AsyncOpenAI],
        purpose: Optional[str] = None,
    ):
        if isinstance(purpose, str):
            response = await openai_client.files.list(purpose=purpose)
        else:
            response = await openai_client.files.list()
        return response

    def list_files(
        self,
        _is_async: bool,
        api_base: Optional[str],
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        purpose: Optional[str] = None,
        api_version: Optional[str] = None,
        client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = None,
        litellm_params: Optional[dict] = None,
    ):
        openai_client: Optional[
            Union[AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI]
        ] = self.get_azure_openai_client(
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
            return self.alist_files(  # type: ignore
                purpose=purpose,
                openai_client=openai_client,
            )

        if isinstance(purpose, str):
            response = openai_client.files.list(purpose=purpose)
        else:
            response = openai_client.files.list()

        return response
