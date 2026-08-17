"""
Handler for Bedrock AgentCore A2A-native agents.

Sends JSON-RPC envelopes directly to AgentCore endpoints, bypassing the
completion bridge that would otherwise strip the envelope.
"""

import json
from collections.abc import AsyncIterator
from typing import Any, Final, cast

from litellm._logging import verbose_logger
from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
    BedrockAgentCoreA2ATransformation,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider


class BedrockAgentCoreA2AHandler:
    """
    Handler for Bedrock AgentCore A2A requests.

    Constructs JSON-RPC envelopes, signs them via AmazonAgentCoreConfig,
    and POSTs directly to the AgentCore endpoint.
    """

    @staticmethod
    async def handle_non_streaming(
        request_id: str,
        params: dict[str, Any],
        litellm_params: dict[str, Any],
        agent_extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Handle non-streaming A2A request to AgentCore.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            litellm_params: Agent's litellm_params (model, api_key, etc.)
            agent_extra_headers: Per-request headers (from x-a2a-{agent}-* rewrite and
                admin extra_headers) to forward on the upstream HTTP call.

        Returns:
            A2A JSON-RPC response dict from the AgentCore agent
        """
        url, headers, body = BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            method="message/send",
            agent_extra_headers=agent_extra_headers,
        )

        verbose_logger.info("BedrockAgentCore A2A: Sending non-streaming request to %s", url)

        client: Final = get_async_httpx_client(
            llm_provider=cast(Any, httpxSpecialProvider.A2AProvider),
        )
        response: Final = await client.post(
            url,
            headers=headers,
            data=body,
        )
        response.raise_for_status()
        response_data: Final = response.json()

        if "error" in response_data:
            verbose_logger.warning("BedrockAgentCore A2A: Agent returned error: %s", response_data["error"])

        return response_data

    @staticmethod
    async def handle_streaming(
        request_id: str,
        params: dict[str, Any],
        litellm_params: dict[str, Any],
        agent_extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Handle streaming A2A request to AgentCore.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            litellm_params: Agent's litellm_params (model, api_key, etc.)
            agent_extra_headers: Per-request headers (from x-a2a-{agent}-* rewrite and
                admin extra_headers) to forward on the upstream HTTP call.

        Yields:
            A2A streaming response events from the AgentCore agent
        """
        url, headers, body = BedrockAgentCoreA2ATransformation.get_url_and_signed_request(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            method="message/send",
            stream=True,
            agent_extra_headers=agent_extra_headers,
        )

        verbose_logger.info("BedrockAgentCore A2A: Sending streaming request to %s", url)

        client: Final = get_async_httpx_client(
            llm_provider=cast(Any, httpxSpecialProvider.A2AProvider),
        )
        response: Final = await client.post(
            url,
            headers=headers,
            data=body,
            stream=True,
        )
        response.raise_for_status()

        # Check content type — AgentCore may return JSON instead of SSE
        content_type: Final = response.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            # Single JSON response fallback (not SSE)
            verbose_logger.debug(
                "BedrockAgentCore A2A streaming: received JSON instead of SSE, yielding as single event"
            )
            response_body: Final = await response.aread()
            response_data: Final = json.loads(response_body)
            yield response_data
        else:
            # SSE stream — parse data: lines
            async for event in BedrockAgentCoreA2ATransformation.parse_sse_events(response):
                yield event
