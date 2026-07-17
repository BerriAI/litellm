from __future__ import annotations

import base64
from typing import List, Optional, Sequence

from fastapi import Request

from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

from litellm.proxy.auth_v2 import errors
from litellm.proxy.auth_v2.config import OAuth2IntrospectionConfig
from litellm.proxy.auth_v2.models import AuthMethod, Credential, SecuritySchemeType
from litellm.proxy.auth_v2.authenticators.base import (
    Authenticator,
    Carrier,
    CredentialLocation,
)
from litellm.proxy.auth_v2.authenticators.types import (
    IntrospectionClient,
    IntrospectionClientFactory,
)
from litellm.proxy.auth_v2.authenticators.utils import (
    JWTVerifier,
    authenticate_bearer_jwt,
    extract_bearer,
    looks_like_jwt,
    normalize_audience,
    split_scope,
)


def _default_introspection_client() -> IntrospectionClient:
    return get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)


class OAuth2Authenticator(Authenticator):
    def __init__(
        self,
        jwt_verifiers: List[JWTVerifier],
        introspection: Optional[OAuth2IntrospectionConfig],
        client_factory: Optional[IntrospectionClientFactory] = None,
    ) -> None:
        self._verifiers = jwt_verifiers
        self._introspection = introspection
        self._client_factory = client_factory or _default_introspection_client

    async def authenticate(self, request: Request) -> Optional[Credential]:
        token = extract_bearer(request)
        if token is None:
            return None
        if looks_like_jwt(token):
            return await authenticate_bearer_jwt(
                token,
                self._verifiers,
                SecuritySchemeType.OAUTH2,
                AuthMethod.BEARER_JWT,
                require_at_jwt=True,
            )
        if self._introspection is not None:
            return await self._introspect(token)
        raise errors.invalid_token()

    async def _introspect(self, token: str) -> Credential:
        config = self._introspection
        assert config is not None
        basic = base64.b64encode(
            f"{config.client_id}:{config.client_secret.get_secret_value()}".encode()
        ).decode()
        client = self._client_factory()
        response = await client.post(
            str(config.introspection_endpoint),
            data={"token": token},
            headers={"Authorization": f"Basic {basic}"},
            timeout=10.0,
        )
        if response.status_code != 200:
            raise errors.invalid_token("introspection failed")
        try:
            body = response.json()
        except ValueError as exc:
            raise errors.invalid_token("introspection failed") from exc
        if not isinstance(body, dict) or body.get("active") is not True:
            raise errors.invalid_token("token inactive")
        token_audience = normalize_audience(body.get("aud"))
        if config.audience and not set(token_audience) & set(config.audience):
            raise errors.invalid_token("audience mismatch")
        if config.issuer is not None and body.get("iss") != config.issuer:
            raise errors.invalid_token("issuer mismatch")
        claims = {key: value for key, value in body.items() if key != "roles"}
        return Credential(
            scheme=SecuritySchemeType.OAUTH2,
            method=AuthMethod.OAUTH2_INTROSPECTION,
            subject=str(body.get(config.subject_field, "")),
            issuer=body.get("iss"),
            audience=token_audience,
            scopes=split_scope(body.get("scope")),
            claims=claims,
            subject_token=token,
        )

    def carriers(self) -> Sequence[Carrier]:
        return (Carrier(CredentialLocation.AUTHORIZATION_SCHEME, "bearer"),)

    def challenge(self) -> str:
        return errors.bearer_challenge()
