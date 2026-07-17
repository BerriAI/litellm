from __future__ import annotations

from typing import Tuple, cast

from authlib.integrations.starlette_client import OAuth
from fastapi import Request
from fastapi.security import SecurityScopes
from saml2.client import Saml2Client

from litellm.proxy.auth_v2.models import Principal
from litellm.proxy.auth_v2.resolvers import ProvisioningStore
from litellm.proxy.auth_v2.security import AuthSecurity

from ..services.oidc import build_oauth_registry
from ..services.saml import SAMLProtocolStore, build_sp_client


def get_auth(request: Request) -> AuthSecurity:
    return request.app.state.auth_v2


def get_oauth_registry(request: Request) -> OAuth:
    cached = getattr(request.app.state, "oidc_oauth", None)
    if cached is None:
        cached = build_oauth_registry(get_auth(request).config.oidc_providers)
        request.app.state.oidc_oauth = cached
    return cached


def get_saml_runtime(request: Request) -> Tuple[Saml2Client, SAMLProtocolStore]:
    state = request.app.state
    client = getattr(state, "saml_client", None)
    if client is None:
        auth = get_auth(request)
        config = auth.config.saml
        assert config is not None
        client = build_sp_client(config)
        state.saml_client = client
        state.saml_protocol = SAMLProtocolStore(auth.config.session.ttl_seconds)
    return client, state.saml_protocol


async def scim_principal(request: Request) -> Principal:
    auth = get_auth(request)
    return await auth.principal(SecurityScopes(scopes=["scim:write"]), request)


def scim_store(request: Request) -> ProvisioningStore:
    return cast(ProvisioningStore, get_auth(request).resolver)
