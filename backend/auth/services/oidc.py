from __future__ import annotations

import re
from typing import Any, Dict, List

from authlib.integrations.starlette_client import OAuth
from scim2_models import User as ScimUser

from litellm.proxy.auth_v2.config import OIDCProviderConfig

CLAIM_KEYS = ("email", "preferred_username", "name", "groups", "roles")


def provider_key(provider: OIDCProviderConfig) -> str:
    return re.sub(r"[^a-z0-9]+", "-", provider.issuer.lower()).strip("-")


def providers_by_key(
    providers: List[OIDCProviderConfig],
) -> Dict[str, OIDCProviderConfig]:
    return {provider_key(provider): provider for provider in providers}


def user_from_userinfo(userinfo: Dict[str, Any]) -> ScimUser:
    return ScimUser(
        external_id=userinfo.get("sub"),
        user_name=userinfo.get("preferred_username") or userinfo.get("email"),
        display_name=userinfo.get("name"),
    )


def mapped_claims(userinfo: Dict[str, Any]) -> Dict[str, Any]:
    return {key: userinfo[key] for key in CLAIM_KEYS if userinfo.get(key) is not None}


def build_oauth_registry(providers: List[OIDCProviderConfig]) -> OAuth:
    oauth = OAuth()
    for provider in providers:
        oauth.register(
            name=provider_key(provider),
            server_metadata_url=f"{provider.issuer.rstrip('/')}/.well-known/openid-configuration",
            client_id=provider.client_id,
            client_secret=(provider.client_secret.get_secret_value() if provider.client_secret else None),
            client_kwargs={
                "scope": " ".join(provider.login_scopes),
                "code_challenge_method": "S256",
            },
        )
    return oauth
