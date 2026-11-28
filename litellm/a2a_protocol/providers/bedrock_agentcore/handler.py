"""
Handler for Bedrock AgentCore A2A-native agents.

Sends JSON-RPC envelopes directly to AgentCore endpoints, bypassing the
completion bridge that would otherwise strip the envelope.
"""

import inspect
import json
from collections.abc import AsyncIterator, Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.a2a_protocol.providers.bedrock_agentcore.transformation import (
    BedrockAgentCoreA2ATransformation,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping, json_contains, subscript
from litellm.llms.bedrock.base_aws_llm import run_aws_signing
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

_DECODED_JSON: Final = TypeAdapter(object)


def _parse_jsonrpc_error_payload(payload: str | bytes | bytearray) -> Mapping[str, object] | None:
    try:
        decoded: Final = _DECODED_JSON.validate_python(json.loads(payload))
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        return None
    mapping: Final = as_str_mapping(decoded)
    if mapping is not None and "error" in mapping:
        return mapping
    return None


def _extract_jsonrpc_error_from_http_status_error(exc: httpx.HTTPStatusError) -> Mapping[str, object] | None:
    raw_text: Final = getattr(exc, "text", None)
    if isinstance(raw_text, str) and raw_text:
        parsed_text: Final = _parse_jsonrpc_error_payload(raw_text)
        if parsed_text is not None:
            return parsed_text
    try:
        raw_content: Final = exc.response.content
    except httpx.ResponseNotRead:
        return None
    if raw_content:
        return _parse_jsonrpc_error_payload(raw_content)
    return None


async def _close_stream_response(response: object) -> None:
    aclose: Final = getattr(response, "aclose", None)
    if callable(aclose):
        maybe_awaitable: Final = aclose()
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable


class BedrockAgentCoreA2AHandler:
    """
    Handler for Bedrock AgentCore A2A requests.

    Constructs JSON-RPC envelopes, signs them via AmazonAgentCoreConfig,
    and POSTs directly to the AgentCore endpoint.
    """

    @staticmethod
    async def handle_jsonrpc(
        method: str,
        request_id: str | int | None,
        params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        agent_extra_headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, object]:
        """Sign and send a non-streaming JSON-RPC request to a Bedrock AgentCore agent."""
        url, headers, body = await run_aws_signing(
            BedrockAgentCoreA2ATransformation.get_url_and_signed_request,
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            method=method,
            agent_extra_headers=agent_extra_headers,
        )

        verbose_logger.info("BedrockAgentCore A2A: Sending %s request to %s", method, url)

        client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.A2AProvider,
        )
        try:
            response: Final = await client.post(
                url,
                headers=headers,
                data=body,
            )
        except httpx.HTTPStatusError as exc:
            http_error_mapping: Final = _extract_jsonrpc_error_from_http_status_error(exc)
            if http_error_mapping is not None:
                verbose_logger.warning(
                    "BedrockAgentCore A2A: Agent returned error: %s",
                    http_error_mapping.get("error"),
                )
                return http_error_mapping
            raise
        try:
            raw_response: Final = response.json()
        except Exception:
            response.raise_for_status()
            raise

        decoded_response: Final = _DECODED_JSON.validate_python(raw_response)
        response_mapping: Final = as_str_mapping(decoded_response)
        is_error_envelope: Final = response_mapping is not None and "error" in response_mapping
        if getattr(response, "is_success", True) is not False or not is_error_envelope:
            response.raise_for_status()

        if json_contains(decoded_response, "error"):
            verbose_logger.warning(
                "BedrockAgentCore A2A: Agent returned error: %s", subscript(decoded_response, "error")
            )

        if response_mapping is None:
            raise ValueError(
                f"Expected JSON object response from AgentCore {method}, got {type(decoded_response).__name__}"
            )
        return response_mapping

    @staticmethod
    async def handle_non_streaming(
        request_id: str,
        params: Mapping[str, object],
        litellm_params: dict[str, object],
        agent_extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
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
        result: Final = await BedrockAgentCoreA2AHandler.handle_jsonrpc(
            method="message/send",
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            agent_extra_headers=agent_extra_headers,
        )
        return dict(result)

    @staticmethod
    async def handle_jsonrpc_streaming(
        method: str,
        request_id: str | int | None,
        params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        agent_extra_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[Mapping[str, object]]:
        """Sign and stream a JSON-RPC SSE/JSON request to a Bedrock AgentCore agent."""
        url, headers, body = await run_aws_signing(
            BedrockAgentCoreA2ATransformation.get_url_and_signed_request,
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            method=method,
            stream=True,
            agent_extra_headers=agent_extra_headers,
        )

        verbose_logger.info("BedrockAgentCore A2A: Sending streaming %s request to %s", method, url)

        client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.A2AProvider,
        )
        try:
            response: Final = await client.post(
                url,
                headers=headers,
                data=body,
                stream=True,
            )
        except httpx.HTTPStatusError as exc:
            http_error_mapping: Final = _extract_jsonrpc_error_from_http_status_error(exc)
            if http_error_mapping is not None:
                yield http_error_mapping
                return
            raise
        try:
            if getattr(response, "is_success", True) is False:
                error_body: Final = await response.aread()
                try:
                    decoded_error: Final = _DECODED_JSON.validate_python(json.loads(error_body))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    response.raise_for_status()
                    return
                error_mapping: Final = as_str_mapping(decoded_error)
                if error_mapping is not None and "error" in error_mapping:
                    yield error_mapping
                    return
                response.raise_for_status()
                return

            response.raise_for_status()

            # Check content type — AgentCore may return JSON instead of SSE
            content_type: Final = ", ".join(response.headers.get_list("content-type")).lower()

            if "application/json" in content_type:
                # Single JSON response fallback (not SSE)
                verbose_logger.debug(
                    "BedrockAgentCore A2A streaming: received JSON instead of SSE, yielding as single event"
                )
                response_body: Final = await response.aread()
                decoded_json: Final = _DECODED_JSON.validate_python(json.loads(response_body))
                json_mapping: Final = as_str_mapping(decoded_json)
                if json_mapping is not None:
                    yield json_mapping
            else:
                # SSE stream — parse data: lines
                async for event in BedrockAgentCoreA2ATransformation.parse_sse_events(response):
                    yield event
        finally:
            await _close_stream_response(response)

    @staticmethod
    async def handle_streaming(
        request_id: str,
        params: Mapping[str, object],
        litellm_params: dict[str, object],
        agent_extra_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[dict[str, object]]:
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
        async for event in BedrockAgentCoreA2AHandler.handle_jsonrpc_streaming(
            method="message/stream",
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            agent_extra_headers=agent_extra_headers,
        ):
            yield dict(event)
