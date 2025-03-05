"""
- call /messages on Anthropic API
- Make streaming + non-streaming request - just pass it through direct to Anthropic. No need to do anything special here 
- Ensure requests are logged in the DB - stream + non-stream

"""

import json
from typing import Any, AsyncIterator, Dict, Optional, Union

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)


async def anthropic_messages_handler(
    api_base: str,
    api_key: str,
    stream: bool = False,
    client: Optional[AsyncHTTPHandler] = None,
    **kwargs,
) -> Union[Dict[str, Any], AsyncIterator]:
    """
    Handler for Anthropic Messages API
    """
    # Use provided client or create a new one
    if client is None:
        async_httpx_client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.ANTHROPIC
        )
    else:
        async_httpx_client = client

    # Prepare headers
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Prepare request body
    request_body = kwargs.copy()
    request_body["stream"] = stream

    # Make the request
    response = await async_httpx_client.post(
        url=f"{api_base}/v1/messages",
        headers=headers,
        data=json.dumps(request_body),
        timeout=600,
        stream=stream,
    )
    response.raise_for_status()

    if stream:
        return response.aiter_bytes()
    else:
        return response.json()
