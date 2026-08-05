from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Protocol

import httpx

import litellm
from litellm.exceptions import APIConnectionError, AuthenticationError, Timeout
from litellm.types.utils import LlmProviders

from ..authenticator import Authenticator
from ..common_utils import GetAccessTokenError, get_chatgpt_default_headers

_FORWARDED_HEADERS: Final = frozenset({"originator", "x-codex-turn-metadata"})
_EMPTY_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})


class ChatGPTAuthenticator(Protocol):
    def get_access_token(self) -> str: ...

    def get_account_id(self) -> str | None: ...

    def get_api_base(self) -> str: ...


class ChatGPTSearchHandler:
    def __init__(
        self,
        authenticator: ChatGPTAuthenticator | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.authenticator: Final = authenticator or Authenticator()
        self.client: Final = client

    async def search(
        self,
        payload: bytes,
        model: str,
        session_id: str | None = None,
        api_base: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        try:
            access_token: Final = self.authenticator.get_access_token()
        except GetAccessTokenError as exc:
            raise AuthenticationError(
                model=model,
                llm_provider=LlmProviders.CHATGPT.value,
                message=str(exc),
            ) from exc

        account_id: Final = self.authenticator.get_account_id()
        default_headers: Final[Mapping[str, str]] = MappingProxyType(
            get_chatgpt_default_headers(access_token, account_id, session_id)
        )
        forwarded_headers: Final[Mapping[str, str]] = MappingProxyType(
            {
                name.lower(): value
                for name, value in (extra_headers or _EMPTY_HEADERS).items()
                if name.lower() in _FORWARDED_HEADERS
            }
        )
        headers: Final[Mapping[str, str]] = MappingProxyType(
            {
                **default_headers,
                "accept": "application/json",
                **forwarded_headers,
            }
        )
        endpoint: Final = f"{(api_base or self.authenticator.get_api_base()).rstrip('/')}/alpha/search"
        client: Final = self.client or litellm.module_level_aclient.client

        try:
            if timeout is None:
                return await client.post(endpoint, content=payload, headers=headers)
            return await client.post(endpoint, content=payload, headers=headers, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise Timeout(
                message=f"ChatGPT search request timed out: {exc}",
                model=model,
                llm_provider=LlmProviders.CHATGPT.value,
            ) from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(
                message=f"ChatGPT search request failed: {exc}",
                model=model,
                llm_provider=LlmProviders.CHATGPT.value,
                request=exc.request,
            ) from exc
