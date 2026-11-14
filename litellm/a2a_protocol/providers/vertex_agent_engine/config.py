"""A2A completion bridge for Vertex AI Agent Engine sessions."""

from collections.abc import AsyncIterator
from uuid import uuid4

import httpx

from litellm.a2a_protocol.litellm_completion_bridge.handler import (
    A2A_USER_API_KEY_HASH_PARAM,
    A2ACompletionBridgeHandler,
)
from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig
from litellm.a2a_protocol.providers.vertex_agent_engine.backend import (
    VertexAgentEngineSessionBackend,
)
from litellm.a2a_protocol.providers.vertex_agent_engine.session import (
    resolve_session_id,
    session_user_id,
)
from litellm.a2a_protocol.utils import get_session_id_from_a2a_params
from litellm.llms.vertex_ai.agent_engine.transformation import (
    VertexAgentEngineConfig,
)


class VertexAgentEngineA2AConfig(BaseA2AProviderConfig):
    """Preserve an isolated Agent Engine session per A2A conversation."""

    @staticmethod
    async def _resolve_session_params(
        params: dict[str, object],
        litellm_params: dict,
    ) -> tuple[str, dict]:
        key_hash = litellm_params.get(A2A_USER_API_KEY_HASH_PARAM)

        if not isinstance(key_hash, str) or not key_hash:
            raise ValueError("Authenticated calling key hash is required for Vertex Agent Engine A2A sessions")

        incoming_context = get_session_id_from_a2a_params(params)
        context_id = incoming_context if isinstance(incoming_context, str) and incoming_context else uuid4().hex

        model = litellm_params.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("Vertex Agent Engine model is required")

        engine_config = VertexAgentEngineConfig()

        stream_url = engine_config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            optional_params={},
            litellm_params=litellm_params,
            stream=True,
        )

        if not stream_url.endswith(":streamQuery"):
            raise ValueError("Unexpected Vertex Agent Engine endpoint")

        query_url = stream_url.removesuffix(":streamQuery") + ":query"

        auth_headers = engine_config._get_auth_headers(
            optional_params={},
            litellm_params=litellm_params,
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            backend = VertexAgentEngineSessionBackend(
                client=client,
                query_url=query_url,
                auth_headers=auth_headers,
            )

            session_id = await resolve_session_id(
                backend,
                calling_key_hash=key_hash,
                context_id=context_id,
            )

        user_id = session_user_id(key_hash, context_id)

        resolved_params = {
            **litellm_params,
            "user": user_id,
            "user_id": user_id,
            "session_id": session_id,
        }

        return context_id, resolved_params

    @staticmethod
    def _set_stream_context(
        event: dict[str, object],
        context_id: str,
    ) -> dict[str, object]:
        """Preserve the A2A conversation ID in streaming events."""
        result = event.get("result")

        if not isinstance(result, dict):
            return event

        result["contextId"] = context_id

        history = result.get("history")
        if isinstance(history, list):
            for message in history:
                if isinstance(message, dict):
                    message["contextId"] = context_id

        status = result.get("status")
        if isinstance(status, dict):
            message = status.get("message")
            if isinstance(message, dict):
                message["contextId"] = context_id

        return event

    async def handle_non_streaming(
        self,
        request_id: str,
        params: dict[str, object],
        api_base: str | None = None,
        **kwargs,
    ) -> dict[str, object]:
        litellm_params = kwargs.get("litellm_params")

        if not isinstance(litellm_params, dict):
            raise ValueError("litellm_params is required")

        context_id, resolved_params = await self._resolve_session_params(
            params,
            litellm_params,
        )

        response = await A2ACompletionBridgeHandler.handle_non_streaming(
            request_id=request_id,
            params=params,
            litellm_params=resolved_params,
            api_base=api_base,
            agent_extra_headers=kwargs.get("agent_extra_headers"),
            _skip_a2a_provider_routing=True,
        )

        result = response.get("result")
        if isinstance(result, dict):
            result["contextId"] = context_id

        return response

    async def handle_streaming(
        self,
        request_id: str,
        params: dict[str, object],
        api_base: str | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, object]]:
        litellm_params = kwargs.get("litellm_params")

        if not isinstance(litellm_params, dict):
            raise ValueError("litellm_params is required")

        context_id, resolved_params = await self._resolve_session_params(
            params,
            litellm_params,
        )

        async for event in A2ACompletionBridgeHandler.handle_streaming(
            request_id=request_id,
            params=params,
            litellm_params=resolved_params,
            api_base=api_base,
            agent_extra_headers=kwargs.get("agent_extra_headers"),
            _skip_a2a_provider_routing=True,
        ):
            yield self._set_stream_context(event, context_id)
