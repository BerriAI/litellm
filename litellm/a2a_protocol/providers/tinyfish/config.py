"""A2A provider configuration for TinyFish Agent (goal-based web automation)."""

from collections.abc import AsyncIterator, Mapping
from typing import Final

from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig
from litellm.a2a_protocol.providers.tinyfish.handler import TinyfishAgentHandler
from litellm.a2a_protocol.providers.tinyfish.transformation import EMPTY_MAPPING, as_str_object_dict


class TinyfishA2AConfig(BaseA2AProviderConfig):
    """A2A bridge for the TinyFish Agent REST API (run-async + poll, native SSE)."""

    async def handle_non_streaming(
        self,
        request_id: str,
        params: Mapping[str, object],
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseA2AProviderConfig passes litellm_params via kwargs
    ) -> dict[str, object]:  # mutable-ok: BaseA2AProviderConfig contract returns a plain dict
        litellm_params: Final = as_str_object_dict(kwargs.get("litellm_params")) or EMPTY_MAPPING
        result: Final = await TinyfishAgentHandler.handle_non_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            api_base=api_base,
        )
        return {**result}  # mutable-ok: response is JSON-serialized downstream; must be a plain dict

    async def handle_streaming(
        self,
        request_id: str,
        params: Mapping[str, object],
        api_base: str | None = None,
        **kwargs: object,  # kwargs-ok: BaseA2AProviderConfig passes litellm_params via kwargs
    ) -> AsyncIterator[dict[str, object]]:  # mutable-ok: BaseA2AProviderConfig contract yields plain dicts
        litellm_params: Final = as_str_object_dict(kwargs.get("litellm_params")) or EMPTY_MAPPING
        async for chunk in TinyfishAgentHandler.handle_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            api_base=api_base,
        ):
            yield {**chunk}  # mutable-ok: chunks are JSON-serialized downstream; must be plain dicts
