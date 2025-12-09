import asyncio
from typing import Any, Coroutine, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.openai import (
    FileContentRequest,
    HttpxBinaryResponseContent,
)

from ..common_utils import AnthropicModelInfo


class AnthropicFilesHandler:
    """
    Handles Anthropic Files API operations.
    
    Currently supports:
    - file_content() for retrieving Anthropic Message Batch results
    """

    def __init__(self):
        self.anthropic_model_info = AnthropicModelInfo()

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = 600.0,
        max_retries: Optional[int] = None,
    ) -> HttpxBinaryResponseContent:
        """
        Async: Retrieve file content from Anthropic.
        
        For batch results, the file_id should be the batch_id.
        This will call Anthropic's /v1/messages/batches/{batch_id}/results endpoint.
        
        Args:
            file_content_request: Contains file_id (batch_id for batch results)
            api_base: Anthropic API base URL
            api_key: Anthropic API key
            timeout: Request timeout
            max_retries: Max retry attempts (unused for now)
            
        Returns:
            HttpxBinaryResponseContent: Binary content wrapped in compatible response format
        """
        file_id = file_content_request.get("file_id")
        if not file_id:
            raise ValueError("file_id is required in file_content_request")

        # Extract batch_id from file_id
        # Handle both formats: "anthropic_batch_results:{batch_id}" or just "{batch_id}"
        if file_id.startswith("anthropic_batch_results:"):
            batch_id = file_id.replace("anthropic_batch_results:", "", 1)
        else:
            batch_id = file_id

        # Get Anthropic API credentials
        api_base = self.anthropic_model_info.get_api_base(api_base)
        api_key = api_key or self.anthropic_model_info.get_api_key()

        if not api_key:
            raise ValueError("Missing Anthropic API Key")

        # Construct the Anthropic batch results URL
        results_url = f"{api_base.rstrip('/')}/v1/messages/batches/{batch_id}/results"

        # Prepare headers
        headers = {
            "accept": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": api_key,
        }

        # Make the request to Anthropic
        async_client = get_async_httpx_client(llm_provider="anthropic")
        try:
            anthropic_response = await async_client.get(
                url=results_url,
                headers=headers
            )
            anthropic_response.raise_for_status()

            # Return the response content
            return HttpxBinaryResponseContent(response=anthropic_response)
        finally:
            await async_client.aclose()

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = 600.0,
        max_retries: Optional[int] = None,
    ) -> Union[
        HttpxBinaryResponseContent, Coroutine[Any, Any, HttpxBinaryResponseContent]
    ]:
        """
        Retrieve file content from Anthropic.
        
        For batch results, the file_id should be the batch_id.
        This will call Anthropic's /v1/messages/batches/{batch_id}/results endpoint.
        
        Args:
            _is_async: Whether to run asynchronously
            file_content_request: Contains file_id (batch_id for batch results)
            api_base: Anthropic API base URL
            api_key: Anthropic API key
            timeout: Request timeout
            max_retries: Max retry attempts (unused for now)
            
        Returns:
            HttpxBinaryResponseContent or Coroutine: Binary content wrapped in compatible response format
        """
        if _is_async:
            return self.afile_content(
                file_content_request=file_content_request,
                api_base=api_base,
                api_key=api_key,
                max_retries=max_retries,
            )
        else:
            return asyncio.run(
                self.afile_content(
                    file_content_request=file_content_request,
                    api_base=api_base,
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )

