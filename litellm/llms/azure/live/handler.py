from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Protocol
from urllib.parse import urlsplit, urlunsplit

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.llms.openai.live.handler import LiveClientWebSocket, OpenAILiveSessions

from ..realtime.handler import AzureOpenAIRealtime

_LIVE_WEBSOCKET_SCHEMES: Final = MappingProxyType({"https": "wss", "http": "ws"})


class LiveSessionsRelay(Protocol):
    async def async_live_session(
        self,
        *,
        model: str,
        websocket: LiveClientWebSocket,
        logging_obj: LiteLLMLogging,
        session_start: Mapping[str, object],
        api_base: str | None,
        api_key: str | None,
        upstream_url: str | None = None,
        additional_headers: Mapping[str, str] | None = None,
    ) -> None: ...


class AzureOpenAILiveSessions:
    def __init__(self, live_sessions: LiveSessionsRelay | None = None) -> None:
        self._live_sessions = live_sessions or OpenAILiveSessions()

    @staticmethod
    def construct_live_url(api_base: str) -> str:
        parsed_url: Final = urlsplit(api_base)
        websocket_scheme: Final = _LIVE_WEBSOCKET_SCHEMES.get(parsed_url.scheme, parsed_url.scheme)
        return urlunsplit((websocket_scheme, parsed_url.netloc, "/openai/v1/live/sessions", "", ""))

    async def async_live_session(
        self,
        *,
        model: str,
        websocket: LiveClientWebSocket,
        logging_obj: LiteLLMLogging,
        session_start: Mapping[str, object],
        api_base: str | None,
        api_key: str | None,
        azure_ad_token: str | None,
    ) -> None:
        if api_base is None:
            raise ValueError("api_base is required for Azure OpenAI Live session calls")
        auth_headers: Final = AzureOpenAIRealtime.get_auth_headers(
            api_key=api_key,
            azure_ad_token=azure_ad_token,
        )
        credential: Final = api_key or azure_ad_token
        await self._live_sessions.async_live_session(
            model=model,
            websocket=websocket,
            logging_obj=logging_obj,
            session_start=session_start,
            api_base=api_base,
            api_key=credential,
            upstream_url=self.construct_live_url(api_base),
            additional_headers=auth_headers,
        )
