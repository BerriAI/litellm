"""
- call /messages on Anthropic API
- Make streaming + non-streaming request - just pass it through direct to Anthropic. No need to do anything special here 
- Ensure requests are logged in the DB - stream + non-stream

"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)
from litellm.utils import client

DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com"


class AnthropicMessagesConfig:

    @staticmethod
    def get_supported_passthrough_params() -> List[str]:
        return [
            "messages",
            "model",
            "system",
            "max_tokens",
            # "metadata",
            "stop_sequences",
            "temperature",
            "top_p",
            "top_k",
            "tools",
            "tool_choice",
            "thinking",
        ]


@client
async def anthropic_messages(
    api_key: str,
    stream: bool = False,
    api_base: Optional[str] = None,
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

    litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj", None)

    # Prepare headers
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Prepare request body
    request_body = kwargs.copy()
    request_body = {
        k: v
        for k, v in request_body.items()
        if k in AnthropicMessagesConfig.get_supported_passthrough_params()
    }
    request_body["stream"] = stream
    litellm_logging_obj.model_call_details.update(request_body)

    # API base
    api_base = api_base or DEFAULT_ANTHROPIC_API_BASE

    verbose_logger.debug(
        "request_body= %s", json.dumps(request_body, indent=4, default=str)
    )

    # Make the request
    response = await async_httpx_client.post(
        url=f"{api_base}/v1/messages",
        headers=headers,
        data=json.dumps(request_body),
        timeout=600,
        stream=stream,
    )
    response.raise_for_status()

    # used for logging + cost tracking
    litellm_logging_obj.model_call_details["httpx_response"] = response

    if stream:
        return response.aiter_bytes()
    else:
        return response.json()
