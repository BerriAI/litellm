"""
Handler for Alibaba Cloud Model Studio (DashScope / Bailian) realtime sessions.

DashScope's Qwen-Omni-Realtime WebSocket API speaks OpenAI-compatible realtime events
(``session.update``, ``input_audio_buffer.append``, ``response.audio.delta``, ...), so the
WebSocket plumbing is inherited from ``OpenAIRealtime``. Only these differ:

- the endpoint path is ``/api-ws/v1/realtime``, not ``/v1/realtime``
- auth is ``Authorization: Bearer <DASHSCOPE_API_KEY>``, with no ``OpenAI-Beta`` header
- the event names are the OpenAI beta dialect, so the flat beta-style ``session.update``
  shape must reach the backend untouched
- sessions are served from a region host, or from a workspace-scoped host such as
  ``wss://{workspace_id}.cn-beijing.maas.aliyuncs.com``

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

from typing import Final

from httpx import URL

from litellm.types.realtime import RealtimeQueryParams

from ...openai.realtime.handler import OpenAIRealtime

REALTIME_WEBSOCKET_PATH: Final = "/api-ws/v1/realtime"

DASHSCOPE_REALTIME_API_BASE: Final = "wss://dashscope.aliyuncs.com"

# get_llm_provider resolves ``dashscope`` to this chat base, which is never where realtime lives.
_DASHSCOPE_CHAT_API_BASE: Final = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def resolve_dashscope_realtime_api_base(api_base: str | None, dynamic_api_base: str | None = None) -> str:
    """
    Resolve the WebSocket base URL for a DashScope realtime session.

    A configured ``api_base`` wins, which is how a workspace-scoped host such as
    ``wss://{workspace_id}.cn-beijing.maas.aliyuncs.com`` is supplied. Otherwise the chat
    base that ``get_llm_provider`` returns is swapped for the region realtime host.
    """
    for candidate in (api_base, dynamic_api_base):
        if candidate and candidate.rstrip("/") != _DASHSCOPE_CHAT_API_BASE:
            return candidate
    return DASHSCOPE_REALTIME_API_BASE


class DashScopeRealtime(OpenAIRealtime):
    """Handler for DashScope realtime WebSocket sessions."""

    def _get_default_api_base(self) -> str:
        return DASHSCOPE_REALTIME_API_BASE

    def get_auth_headers(self, api_key: str) -> dict:
        """DashScope authenticates with a bearer token only, no OpenAI-Beta header."""
        return {"Authorization": f"Bearer {api_key}"}

    def _get_additional_headers(
        self,
        api_key: str,
        *,
        openai_beta_realtime: bool = False,
    ) -> dict:
        return self.get_auth_headers(api_key)

    def _construct_url(self, api_base: str, query_params: RealtimeQueryParams) -> str:
        """Build the backend websocket URL on DashScope's ``/api-ws/v1/realtime`` path."""
        websocket_api_base: Final = api_base.replace("https://", "wss://").replace("http://", "ws://")
        url: Final = URL(websocket_api_base).copy_with(path=REALTIME_WEBSOCKET_PATH)
        if not query_params:
            return str(url)
        return str(url.copy_with(params=query_params))

    def get_websocket_url(self, api_base: str, query_params: RealtimeQueryParams) -> str:
        """Realtime WebSocket URL for callers outside this handler, such as the health check."""
        return self._construct_url(api_base, query_params)

    def _backend_uses_beta_protocol(self) -> bool | None:
        """DashScope always speaks the flat, beta-style realtime session shape.

        Sending a GA-shaped ``session.update`` (``output_modalities``, ``audio.input.*``)
        makes DashScope drop the modality and audio-format fields, so the remap must be
        skipped regardless of whether the client sent ``OpenAI-Beta``.
        """
        return True
