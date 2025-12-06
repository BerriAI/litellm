from typing import Any, Coroutine, Optional, Union

import httpx
from openai.types.file_deleted import FileDeleted

from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
)
from litellm.types.llms.openai import (
    FileContentRequest,
    FileObject,
    HttpxBinaryResponseContent,
)


class AnthropicFilesAPI:
    """
    Anthropic methods to support for files
    - retrieve_file()
    - delete_file()
    - list_files()
    - file_content()
    """

    def __init__(self) -> None:
        pass

    def _get_anthropic_headers(self, api_key: str) -> dict:
        """Common headers for all Anthropic Files API requests"""
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "files-api-2025-04-14",
        }

    def _transform_anthropic_file_response(
        self, http_response: httpx.Response
    ) -> FileObject:
        """Transform Anthropic file response to OpenAI FileObject format"""
        import time

        response_json = http_response.json()

        # Parse created_at
        try:
            from dateutil import parser  # type: ignore[import-untyped]

            created_at = int(
                parser.parse(response_json.get("created_at", "")).timestamp()
            )
        except Exception:
            created_at = int(time.time())

        return FileObject(
            id=response_json.get("id", ""),
            object="file",
            bytes=response_json.get("size_bytes", 0),
            created_at=created_at,
            filename=response_json.get("filename", ""),
            purpose="assistants",
            status="processed",
        )

    async def aretrieve_file(
        self,
        file_id: str,
        api_base: str,
        api_key: str,
        timeout: Union[float, httpx.Timeout],
        client: Optional[AsyncHTTPHandler] = None,
    ) -> FileObject:
        """Async retrieve file metadata"""
        headers = self._get_anthropic_headers(api_key)
        url = f"{api_base}/v1/files/{file_id}"

        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_client = AsyncHTTPHandler(timeout=timeout)
        else:
            async_client = client

        http_response = await async_client.get(url=url, headers=headers)
        return self._transform_anthropic_file_response(http_response)

    def retrieve_file(
        self,
        _is_async: bool,
        file_id: str,
        api_base: str,
        api_key: str,
        timeout: Union[float, httpx.Timeout],
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> Union[FileObject, Coroutine[Any, Any, FileObject]]:
        """Retrieve file metadata"""
        if _is_async:
            return self.aretrieve_file(
                file_id=file_id,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                client=client,  # type: ignore
            )

        headers = self._get_anthropic_headers(api_key)
        url = f"{api_base}/v1/files/{file_id}"

        if client is None or not isinstance(client, HTTPHandler):
            sync_client = HTTPHandler(timeout=timeout)
        else:
            sync_client = client

        http_response = sync_client.get(url=url, headers=headers)
        return self._transform_anthropic_file_response(http_response)

    async def adelete_file(
        self,
        file_id: str,
        api_base: str,
        api_key: str,
        timeout: Union[float, httpx.Timeout],
        client: Optional[AsyncHTTPHandler] = None,
    ) -> FileDeleted:
        """Async delete file"""
        headers = self._get_anthropic_headers(api_key)
        url = f"{api_base}/v1/files/{file_id}"

        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_client = AsyncHTTPHandler(timeout=timeout)
        else:
            async_client = client

        await async_client.delete(url=url, headers=headers)
        return FileDeleted(id=file_id, deleted=True, object="file")

    def delete_file(
        self,
        _is_async: bool,
        file_id: str,
        api_base: str,
        api_key: str,
        timeout: Union[float, httpx.Timeout],
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> Union[FileDeleted, Coroutine[Any, Any, FileDeleted]]:
        """Delete file"""
        if _is_async:
            return self.adelete_file(
                file_id=file_id,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                client=client,  # type: ignore
            )

        headers = self._get_anthropic_headers(api_key)
        url = f"{api_base}/v1/files/{file_id}"

        if client is None or not isinstance(client, HTTPHandler):
            sync_client = HTTPHandler(timeout=timeout)
        else:
            sync_client = client

        sync_client.delete(url=url, headers=headers)
        return FileDeleted(id=file_id, deleted=True, object="file")

    async def alist_files(
        self,
        api_base: str,
        api_key: str,
        timeout: Union[float, httpx.Timeout],
        client: Optional[AsyncHTTPHandler] = None,
    ):
        """Async list files"""
        headers = self._get_anthropic_headers(api_key)
        url = f"{api_base}/v1/files"

        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_client = AsyncHTTPHandler(timeout=timeout)
        else:
            async_client = client

        http_response = await async_client.get(url=url, headers=headers)

        # Transform Anthropic response to OpenAI format
        import time

        response_json = http_response.json()
        files = []

        for file_data in response_json.get("data", []):
            # Parse created_at
            try:
                from dateutil import parser  # type: ignore[import-untyped]

                created_at = int(
                    parser.parse(file_data.get("created_at", "")).timestamp()
                )
            except Exception:
                created_at = int(time.time())

            files.append(
                FileObject(
                    id=file_data.get("id", ""),
                    object="file",
                    bytes=file_data.get("size_bytes", 0),
                    created_at=created_at,
                    filename=file_data.get("filename", ""),
                    purpose="assistants",
                    status="processed",
                )
            )

        return files

    def list_files(
        self,
        _is_async: bool,
        api_base: str,
        api_key: str,
        timeout: Union[float, httpx.Timeout],
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ):
        """List files"""
        if _is_async:
            return self.alist_files(
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                client=client,  # type: ignore
            )

        headers = self._get_anthropic_headers(api_key)
        url = f"{api_base}/v1/files"

        if client is None or not isinstance(client, HTTPHandler):
            sync_client = HTTPHandler(timeout=timeout)
        else:
            sync_client = client

        http_response = sync_client.get(url=url, headers=headers)

        # Transform Anthropic response to OpenAI format
        import time

        response_json = http_response.json()
        files = []

        for file_data in response_json.get("data", []):
            # Parse created_at
            try:
                from dateutil import parser  # type: ignore[import-untyped]

                created_at = int(
                    parser.parse(file_data.get("created_at", "")).timestamp()
                )
            except Exception:
                created_at = int(time.time())

            files.append(
                FileObject(
                    id=file_data.get("id", ""),
                    object="file",
                    bytes=file_data.get("size_bytes", 0),
                    created_at=created_at,
                    filename=file_data.get("filename", ""),
                    purpose="assistants",
                    status="processed",
                )
            )

        return files

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        api_base: str,
        api_key: str,
        timeout: Union[float, httpx.Timeout],
        client: Optional[AsyncHTTPHandler] = None,
    ) -> HttpxBinaryResponseContent:
        """Async download file content"""
        headers = self._get_anthropic_headers(api_key)
        url = f"{api_base}/v1/files/{file_content_request['file_id']}/content"

        if client is None or not isinstance(client, AsyncHTTPHandler):
            async_client = AsyncHTTPHandler(timeout=timeout)
        else:
            async_client = client

        http_response = await async_client.get(url=url, headers=headers)
        return HttpxBinaryResponseContent(response=http_response)

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: str,
        api_key: str,
        timeout: Union[float, httpx.Timeout],
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> Union[
        HttpxBinaryResponseContent, Coroutine[Any, Any, HttpxBinaryResponseContent]
    ]:
        """Download file content"""
        if _is_async:
            return self.afile_content(
                file_content_request=file_content_request,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                client=client,  # type: ignore
            )

        headers = self._get_anthropic_headers(api_key)
        url = f"{api_base}/v1/files/{file_content_request['file_id']}/content"

        if client is None or not isinstance(client, HTTPHandler):
            sync_client = HTTPHandler(timeout=timeout)
        else:
            sync_client = client

        http_response = sync_client.get(url=url, headers=headers)
        return HttpxBinaryResponseContent(response=http_response)
