from collections.abc import AsyncIterator, Mapping
from typing import Final

from litellm.a2a_protocol.litellm_completion_bridge.handler import (
    A2A_USER_API_KEY_HASH_PARAM,
    A2ACompletionBridgeHandler,
)
from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig
from litellm.llms.vertex_ai.agent_engine.a2a import merge_a2a_session_into_litellm_params


def _require_litellm_params(litellm_params: object) -> Mapping[str, object]:
    if not isinstance(litellm_params, Mapping) or not litellm_params:
        raise ValueError(
            "litellm_params is required for VertexAgentEngineA2AConfig (must contain custom_llm_provider and model)"
        )
    return {str(key): value for key, value in litellm_params.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # validated untyped kwargs boundary


def _principal(litellm_params: Mapping[str, object]) -> str | None:
    principal: Final = litellm_params.get(A2A_USER_API_KEY_HASH_PARAM)
    return principal if isinstance(principal, str) else None


def _agent_extra_headers(
    value: object,
) -> dict[str, str] | None:  # mutable-ok: the bridge handler signature takes a dict
    if not isinstance(value, Mapping):
        return None
    return {str(k): str(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # validated untyped kwargs boundary


class VertexAgentEngineA2AConfig(BaseA2AProviderConfig):
    """A2A bridge for Vertex AI Agent Engine: continues the caller's Agent Engine
    session for the same contextId (scoped to the authenticated key), then uses
    completion."""

    async def handle_non_streaming(
        self,
        request_id: str,
        params: dict[str, object],  # mutable-ok: signature fixed by BaseA2AProviderConfig
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: signature fixed by BaseA2AProviderConfig
    ) -> dict[str, object]:  # mutable-ok: signature fixed by BaseA2AProviderConfig
        litellm_params: Final = _require_litellm_params(kwargs.get("litellm_params"))
        merged: Final = await merge_a2a_session_into_litellm_params(litellm_params, params, _principal(litellm_params))
        return await A2ACompletionBridgeHandler.handle_non_streaming(
            request_id=request_id,
            params=params,
            litellm_params=dict(merged),
            api_base=api_base,
            agent_extra_headers=_agent_extra_headers(kwargs.get("agent_extra_headers")),
            _skip_a2a_provider_routing=True,
        )

    async def handle_streaming(
        self,
        request_id: str,
        params: dict[str, object],  # mutable-ok: signature fixed by BaseA2AProviderConfig
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: signature fixed by BaseA2AProviderConfig
    ) -> AsyncIterator[dict[str, object]]:  # mutable-ok: signature fixed by BaseA2AProviderConfig
        litellm_params: Final = _require_litellm_params(kwargs.get("litellm_params"))
        merged: Final = await merge_a2a_session_into_litellm_params(litellm_params, params, _principal(litellm_params))
        async for chunk in A2ACompletionBridgeHandler.handle_streaming(
            request_id=request_id,
            params=params,
            litellm_params=dict(merged),
            api_base=api_base,
            agent_extra_headers=_agent_extra_headers(kwargs.get("agent_extra_headers")),
            _skip_a2a_provider_routing=True,
        ):
            yield chunk
