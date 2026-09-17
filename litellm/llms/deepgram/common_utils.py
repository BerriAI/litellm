from types import MappingProxyType
from typing import Final

import httpx

from litellm.constants import DEEPGRAM_DEFAULT_API_BASE, DEEPGRAM_LISTEN_DEFAULT_MODEL
from litellm.llms.base_llm.chat.transformation import BaseLLMException

_WEBSOCKET_SCHEMES: Final = MappingProxyType({"https": "wss", "http": "ws", "wss": "wss", "ws": "ws"})


class DeepgramException(BaseLLMException):
    pass


def deepgram_listen_websocket_target(api_base: str | None, query_string: str) -> str:
    """
    The upstream ``/listen`` socket for a streaming transcription, keeping the client's query string as sent
    and adding the default model only when the client named none
    """
    listen_url: Final = httpx.URL(f"{(api_base or DEEPGRAM_DEFAULT_API_BASE).rstrip('/')}/listen")
    websocket_url: Final = listen_url.copy_with(scheme=_WEBSOCKET_SCHEMES.get(listen_url.scheme, listen_url.scheme))
    params: Final = httpx.QueryParams(query_string)
    query: Final = (
        query_string if params.get("model") else str(params.remove("model").add("model", DEEPGRAM_LISTEN_DEFAULT_MODEL))
    )
    return f"{websocket_url}?{query}"
