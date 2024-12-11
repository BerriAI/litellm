from typing import Any, Coroutine, Dict, List, Optional, Union

import httpx
from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.file_deleted import FileDeleted

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base import BaseLLM
from litellm.types.llms.openai import *


def get_azure_openai_client(
    api_key: Optional[str],
    api_base: Optional[str],
    timeout: Union[float, httpx.Timeout],
    max_retries: Optional[int],
    api_version: Optional[str] = None,
    organization: Optional[str] = None,
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


class AzureOpenAIFilesAPI(BaseLLM):
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

    async def acreate_file(
        self,
        create_file_data: CreateFileRequest,
        openai_client: AsyncAzureOpenAI,
    ) -> FileObject:
        verbose_logger.debug("create_file_data=%s", create_file_data)
        response = await openai_client.files.create(**create_file_data)
        verbose_logger.debug("create_file_response=%s", response)
        return response

    def create_file(
        self,
        _is_async: bool,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        api_key: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
    ) -> Union[FileObject, Coroutine[Any, Any, FileObject]]:
        openai_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = (
            get_azure_openai_client(
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                timeout=timeout,
                max_retries=max_retries,
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
            return self.acreate_file(  # type: ignore
                create_file_data=create_file_data, openai_client=openai_client
            )
        response = openai_client.files.create(**create_file_data)
        return response

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        openai_client: AsyncAzureOpenAI,
    ) -> HttpxBinaryResponseContent:
        response = await openai_client.files.content(**file_content_request)
        return response

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: Optional[str],
        api_key: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        api_version: Optional[str] = None,
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
    ) -> Union[
        HttpxBinaryResponseContent, Coroutine[Any, Any, HttpxBinaryResponseContent]
    ]:
        openai_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = (
            get_azure_openai_client(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                api_version=api_version,
                max_retries=max_retries,
                organization=None,
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
            return self.afile_content(  # type: ignore
                file_content_request=file_content_request,
                openai_client=openai_client,
            )
        response = openai_client.files.content(**file_content_request)

        return response

    async def aretrieve_file(
        self,
        file_id: str,
        openai_client: AsyncAzureOpenAI,
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
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
    ):
        openai_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = (
            get_azure_openai_client(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=None,
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
            return self.aretrieve_file(  # type: ignore
                file_id=file_id,
                openai_client=openai_client,
            )
        response = openai_client.files.retrieve(file_id=file_id)

        return response

    async def adelete_file(
        self,
        file_id: str,
        openai_client: AsyncAzureOpenAI,
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
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
    ):
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
        openai_client: AsyncAzureOpenAI,
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
        client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = None,
    ):
        openai_client: Optional[Union[AzureOpenAI, AsyncAzureOpenAI]] = (
            get_azure_openai_client(
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
                organization=None,  # openai param
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
            return self.alist_files(  # type: ignore
                purpose=purpose,
                openai_client=openai_client,
            )

        if isinstance(purpose, str):
            response = openai_client.files.list(purpose=purpose)
        else:
            response = openai_client.files.list()

        return response
