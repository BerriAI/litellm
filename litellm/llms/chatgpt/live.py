import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias
from unicodedata import category
from urllib.parse import quote, unquote

import httpx
from pydantic import JsonValue

from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.llms.chatgpt.realtime import (
    ChatGPTRealtime,
    configured_realtime_headers,
    configured_realtime_query,
    realtime_headers,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared client has legacy untyped optional params
    get_shared_realtime_ssl_context,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

LiveQuery: TypeAlias = Mapping[str, str | int | float | bool | None | tuple[str | int | float | bool | None, ...]]
LiveBody: TypeAlias = Mapping[str, JsonValue]
LiveOperation: TypeAlias = Literal["fork", "accept", "reject", "refer", "hangup", "content", "attach"]
_PATH: Final = re.compile(r"live/sessions(?:/([^/]+)/(fork|accept|reject|refer|hangup|content|attach))?\Z")
_ROUTING_QUERY: Final = frozenset(("model", "session_id", "call_id", "api_key", "api_base", "authorization"))


@dataclass(frozen=True, slots=True)
class LiveDeployment:
    model: str
    model_id: str | None = None
    provider: Literal["chatgpt", "openai"] = "chatgpt"
    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)
    extra_headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}), repr=False)
    extra_query: LiveQuery = field(default_factory=lambda: MappingProxyType({}), repr=False)


def _validate_session_id(session_id: str) -> None:
    candidate: str = session_id  # rebind-ok: inspect every decoding layer without recursive stack exhaustion
    while True:
        if (
            not candidate
            or candidate in (".", "..")
            or any(char in ("/", "\\") or category(char) in ("Cc", "Cs") for char in candidate)
        ):
            raise ValueError("Invalid Live session ID")
        decoded: str = unquote(candidate, errors="strict")  # rebind-ok: validate successive decoding layers iteratively
        if decoded == candidate:
            return
        candidate = decoded  # rebind-ok: each percent-decoding pass reduces the input length


def _validate_path(path: str) -> None:
    match: Final = _PATH.fullmatch(path)
    if match is None:
        raise ValueError("Invalid Live endpoint")
    if match.group(1) is None:
        return
    session_id: Final = unquote(match.group(1), errors="strict")
    if quote(session_id, safe="") != match.group(1):
        raise ValueError("Noncanonical Live session path")
    _validate_session_id(session_id)


def live_session_path(session_id: str, operation: LiveOperation) -> str:
    _validate_session_id(session_id)
    path: Final = f"live/sessions/{quote(session_id, safe='')}/{operation}"
    _validate_path(path)
    return path


class LiveTransport:
    def __init__(
        self,
        deployment: LiveDeployment,
        inbound_headers: Mapping[str, str],
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.deployment = deployment
        self._http_client = http_client
        params: Final = GenericLiteLLMParams.model_validate(
            MappingProxyType(
                {
                    "api_base": deployment.api_base,
                    "extra_query": deployment.extra_query,
                }
            )
        )
        self._query = configured_realtime_query(params)
        self._headers = (
            realtime_headers(params, inbound_headers, deployment.extra_headers)
            if deployment.provider == "chatgpt"
            else MappingProxyType(
                {
                    **MappingProxyType(
                        {
                            key.lower(): value
                            for key, value in inbound_headers.items()
                            if key.lower() in ("openai-alpha", "openai-beta", "x-session-id", "x-oai-attestation")
                        }
                    ),
                    **configured_realtime_headers(deployment.extra_headers),
                    "authorization": f"Bearer {deployment.api_key or ''}",
                }
            )
        )

    def _url(self, path: str, query: LiveQuery | None, *, websocket: bool) -> str:
        _validate_path(path)
        base: Final = httpx.URL(
            ChatGPTRealtime.get_api_base(self.deployment.api_base)
            if self.deployment.provider == "chatgpt"
            else self.deployment.api_base or "https://api.openai.com/v1"
        )
        if base.scheme not in ("https", "http", "wss", "ws") or not base.host or base.userinfo or base.fragment:
            raise ValueError("Invalid Live API base")
        merged: Final = base.params.merge(query or MappingProxyType({})).merge(self._query)
        safe_query: Final = tuple(
            (key, value) for key, value in merged.multi_items() if key.lower() not in _ROUTING_QUERY
        )
        return str(
            base.copy_with(
                scheme=("wss" if base.scheme in ("https", "wss") else "ws")
                if websocket
                else ("https" if base.scheme in ("https", "wss") else "http"),
                path=f"{base.path.rstrip('/')}/{path}",
                params=safe_query,
            )
        )

    async def request(
        self,
        method: str,
        path: str,
        body: LiveBody | None = None,
        query: LiveQuery | None = None,
    ) -> httpx.Response:
        if (method, path.rsplit("/", 1)[-1]) not in (
            ("POST", "sessions"),
            ("POST", "fork"),
            ("POST", "accept"),
            ("POST", "reject"),
            ("POST", "refer"),
            ("POST", "hangup"),
            ("GET", "content"),
        ):
            raise ValueError("Invalid Live HTTP operation")
        url: Final = self._url(path, query, websocket=False)
        client: Final = (
            self._http_client
            or get_async_httpx_client(
                llm_provider=LlmProviders.CHATGPT if self.deployment.provider == "chatgpt" else LlmProviders.OPENAI
            ).client
        )
        return await client.request(
            method,
            url,
            headers=MappingProxyType({**self._headers, "content-type": "application/json"}),
            json=dict(body) if body is not None else None,  # mutable-ok: JSON encoder requires a concrete dict
            timeout=60,
            follow_redirects=False,
        )

    async def connect(self, path: str, query: LiveQuery | None = None) -> "ClientConnection":
        import websockets

        class DirectConnect(websockets.connect):
            def process_redirect(self, exc: Exception) -> Exception:
                return exc

        if path != "live/sessions" and path.rsplit("/", 1)[-1] not in ("attach", "fork"):
            raise ValueError("Invalid Live WebSocket operation")
        url: Final = self._url(path, query, websocket=True)
        ssl_context: Final = get_shared_realtime_ssl_context() if url.startswith("wss://") else None
        return await DirectConnect(
            url,
            additional_headers=self._headers,
            max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
            max_queue=16,
            ssl=True if ssl_context is False else ssl_context,
            open_timeout=20,
            close_timeout=10,
        )
