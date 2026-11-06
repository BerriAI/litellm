# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportUntypedBaseClass=false
# fastapi_sso (a proxy extra) and oauthlib are not installed in the type-check gate venv, so pyright sees them as Unknown
"""fastapi_sso providers whose IdP requests go through litellm's configured httpx client.

fastapi_sso opens a bare ``httpx.AsyncClient()`` for the token exchange and the userinfo call, so
``ssl_verify``, ``SSL_VERIFY`` and ``SSL_CERT_FILE`` were ignored on every non-PKCE SSO login.
"""

import json
import warnings
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, overload

import httpx
import jwt
from fastapi import Request, status
from fastapi_sso.sso.base import DiscoveryDocument, OpenID, ReusedOauthClientWarning, SSOBase, SSOLoginError
from fastapi_sso.sso.generic import create_provider
from fastapi_sso.sso.google import GoogleSSO
from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.types.llms.custom_http import httpxSpecialProvider

_IDP_REQUEST_TIMEOUT_SECONDS: Final = 30.0
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_STRING_HEADERS: Final = TypeAdapter(dict[str, str])


@dataclass(frozen=True, slots=True)
class _DiscoveryEndpoints:
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str


_DISCOVERY_ENDPOINTS: Final = TypeAdapter(_DiscoveryEndpoints)


_TOKEN_REQUEST: Final = TypeAdapter(tuple[str, dict[str, str], str])
_USERINFO_REQUEST: Final = TypeAdapter(tuple[str, dict[str, str], str | None])


def _sso_http_client() -> httpx.AsyncClient:
    return get_async_httpx_client(llm_provider=httpxSpecialProvider.SSO_HANDLER).client


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _required_endpoint(endpoint: str | None, name: str) -> str:
    if endpoint is None:
        raise SSOLoginError(500, f"SSO provider has no {name}")
    return endpoint


async def _send_or_raise(request: Awaitable[httpx.Response], endpoint: str, param: str) -> httpx.Response:
    try:
        return await request
    except httpx.HTTPError as exc:
        verbose_proxy_logger.error("SSO request to %s failed: %s", endpoint, exc)
        raise ProxyException(
            message=f"SSO request to {endpoint} failed: {exc}",
            type=ProxyErrorTypes.auth_error,
            param=param,
            code=status.HTTP_401_UNAUTHORIZED,
        ) from exc


class LiteLLMHttpClientSSO(SSOBase):
    """Mixin that replays fastapi_sso's login exchange on litellm's shared SSO httpx client."""

    @overload
    async def process_login(
        self,
        code: str,
        request: Request,
        *,
        params: Mapping[str, object] | None = None,
        additional_headers: Mapping[str, object] | None = None,
        redirect_uri: str | None = None,
        pkce_code_verifier: str | None = None,
        convert_response: Literal[True] = True,
    ) -> OpenID | None: ...

    @overload
    async def process_login(
        self,
        code: str,
        request: Request,
        *,
        params: Mapping[str, object] | None = None,
        additional_headers: Mapping[str, object] | None = None,
        redirect_uri: str | None = None,
        pkce_code_verifier: str | None = None,
        convert_response: Literal[False],
    ) -> dict[str, object] | None: ...  # mutable-ok: fastapi_sso returns the raw userinfo dict

    async def process_login(
        self,
        code: str,
        request: Request,
        *,
        params: Mapping[str, object] | None = None,
        additional_headers: Mapping[str, object] | None = None,
        redirect_uri: str | None = None,
        pkce_code_verifier: str | None = None,
        convert_response: bool = True,
    ) -> OpenID | dict[str, object] | None:  # mutable-ok: fastapi_sso returns the raw userinfo dict
        self._forget_reused_oauth_client()
        token_params: Final = {
            **(params or {}),
            **_JSON_OBJECT.validate_python(self._extra_query_params),
            **({"code_verifier": pkce_code_verifier} if pkce_code_verifier else {}),
        }
        extra_headers: Final = _STRING_HEADERS.validate_python(
            {**(additional_headers or {}), **(self.additional_headers or {})}
        )
        callback_url: Final = str(request.url)
        authorization_response: Final = (
            callback_url
            if self.allow_insecure_http or request.url.scheme == "https"
            else callback_url.replace("http://", "https://")
        )
        callback_path: Final = f"{request.url.scheme}://{request.url.netloc}{request.url.path}"
        token_url, token_headers, body = _TOKEN_REQUEST.validate_python(
            self.oauth_client.prepare_token_request(
                _required_endpoint(await self.token_endpoint, "token_endpoint"),
                authorization_response=authorization_response,
                redirect_url=redirect_uri or (str(self.redirect_uri) if self.redirect_uri else callback_path),
                code=code,
                **token_params,  # pyright: ignore[reportArgumentType]  # oauthlib stubs type the kwargs slots as str
            )
        )
        client: Final = _sso_http_client()
        token_response: Final = await _send_or_raise(
            client.post(
                token_url,
                headers={**token_headers, **extra_headers},
                content=body,
                auth=httpx.BasicAuth(self.client_id, self.client_secret),
                timeout=_IDP_REQUEST_TIMEOUT_SECONDS,
            ),
            endpoint=token_url,
            param="token_exchange",
        )
        token_content: Final = _JSON_OBJECT.validate_python(token_response.json())
        self._refresh_token = _optional_str(token_content.get("refresh_token"))
        self._id_token = _optional_str(token_content.get("id_token"))
        self.oauth_client.parse_request_body_response(json.dumps(token_content))
        userinfo_url, userinfo_headers, _ = _USERINFO_REQUEST.validate_python(
            self.oauth_client.add_token(_required_endpoint(await self.userinfo_endpoint, "userinfo_endpoint"))
        )
        userinfo_response: Final = await _send_or_raise(
            client.get(
                userinfo_url, headers={**userinfo_headers, **extra_headers}, timeout=_IDP_REQUEST_TIMEOUT_SECONDS
            ),
            endpoint=userinfo_url,
            param="userinfo",
        )
        userinfo: Final = _JSON_OBJECT.validate_python(userinfo_response.json())
        if not convert_response:
            return userinfo
        if not self.use_id_token_for_user_info:
            return await self.openid_from_response(userinfo, client)
        if not self._id_token:
            raise SSOLoginError(401, f"Provider {self.provider!r} did not return id token.")
        id_token_claims: Final = _JSON_OBJECT.validate_python(
            jwt.decode(self._id_token, options={"verify_signature": False})
        )
        return await self.openid_from_token(id_token_claims, client)

    def _forget_reused_oauth_client(self) -> None:
        if self._oauth_client is None:
            return
        self._oauth_client = None
        self._refresh_token = None
        self._id_token = None
        warnings.warn(
            "Reusing the SSO object is not safe and caused a security issue in previous versions. "
            "To make sure you don't see this warning, please use the SSO object as a context manager.",
            ReusedOauthClientWarning,
        )


class LiteLLMGoogleSSO(LiteLLMHttpClientSSO, GoogleSSO):
    async def get_discovery_document(self) -> DiscoveryDocument:
        response: Final = await _send_or_raise(
            _sso_http_client().get(self.discovery_url, timeout=_IDP_REQUEST_TIMEOUT_SECONDS),
            endpoint=self.discovery_url,
            param="discovery_document",
        )
        endpoints: Final = _DISCOVERY_ENDPOINTS.validate_python(response.json())
        return DiscoveryDocument(
            authorization_endpoint=endpoints.authorization_endpoint,
            token_endpoint=endpoints.token_endpoint,
            userinfo_endpoint=endpoints.userinfo_endpoint,
        )


def create_generic_sso_provider(
    *,
    name: str,
    discovery_document: DiscoveryDocument,
    response_convertor: Callable[[Mapping[str, object], httpx.AsyncClient | None], OpenID],
) -> type[SSOBase]:
    provider: Final = create_provider(
        name=name,
        discovery_document=discovery_document,
        response_convertor=response_convertor,
    )
    return type(f"LiteLLM{provider.__name__}", (LiteLLMHttpClientSSO, provider), {})
