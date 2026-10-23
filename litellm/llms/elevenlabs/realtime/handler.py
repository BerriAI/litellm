"""
Handler for ElevenLabs' Eleven Realtime speech-to-speech models on `/v1/realtime`.

Eleven Realtime speaks the OpenAI Realtime GA wire protocol, so every client and
backend event is relayed unchanged by `OpenAIRealtime`. Only the handshake differs:
the websocket lives at `wss://api.elevenlabs.io/v1/convai/realtime?model=...` and is
authenticated with the `xi-api-key` header, the same header every other ElevenLabs
endpoint in LiteLLM uses, instead of a bearer token.

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

from typing import Final

from httpx import URL

from litellm.constants import ELEVENLABS_API_BASE
from litellm.types.realtime import RealtimeQueryParams

from ...openai.realtime.handler import OpenAIRealtime

ELEVENLABS_REALTIME_PATH: Final = "/v1/convai/realtime"


class ElevenLabsRealtime(OpenAIRealtime):
    def _get_default_api_base(self) -> str:
        return ELEVENLABS_API_BASE

    def _get_additional_headers(
        self,
        api_key: str,
        *,
        openai_beta_realtime: bool = False,
    ) -> dict[str, str]:  # mutable-ok: matches the OpenAIRealtime signature it overrides
        return {"xi-api-key": api_key}

    def _construct_url(self, api_base: str, query_params: RealtimeQueryParams) -> str:
        ws_base: Final = api_base.replace("https://", "wss://").replace("http://", "ws://")
        url: Final = URL(ws_base).copy_with(path=ELEVENLABS_REALTIME_PATH)
        params: Final = tuple(query_params.items())
        return str(url.copy_with(params=params)) if params else str(url)
