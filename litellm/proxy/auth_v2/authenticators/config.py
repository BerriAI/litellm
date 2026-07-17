from __future__ import annotations

from typing import Dict, List, Optional

from litellm.proxy.auth_v2.config import AuthConfig
from litellm.proxy.auth_v2.models import SecuritySchemeType
from litellm.proxy.auth_v2.authenticators.base import Authenticator
from litellm.proxy.auth_v2.authenticators.http import HttpAuthenticator
from litellm.proxy.auth_v2.authenticators.key import APIKeyAuthenticator
from litellm.proxy.auth_v2.authenticators.mtls import MutualTLSAuthenticator
from litellm.proxy.auth_v2.authenticators.oauth import OAuth2Authenticator
from litellm.proxy.auth_v2.authenticators.oidc import OIDCAuthenticator
from litellm.proxy.auth_v2.authenticators.types import BasicAuthVerifier
from litellm.proxy.auth_v2.authenticators.utils import JWTVerifier


def build_authenticators(
    config: AuthConfig, *, basic_verifier: Optional[BasicAuthVerifier] = None
) -> List[Authenticator]:
    verifiers = [JWTVerifier(provider) for provider in config.oidc_providers]
    by_scheme: Dict[SecuritySchemeType, Authenticator] = {}
    if config.api_key is not None:
        by_scheme[SecuritySchemeType.API_KEY] = APIKeyAuthenticator(config.api_key)
    by_scheme[SecuritySchemeType.HTTP] = HttpAuthenticator(
        config.http_basic, verifiers, basic_verifier
    )
    by_scheme[SecuritySchemeType.OPENID_CONNECT] = OIDCAuthenticator(verifiers)
    by_scheme[SecuritySchemeType.OAUTH2] = OAuth2Authenticator(
        verifiers, config.oauth2_introspection
    )
    if config.mutual_tls.enabled:
        by_scheme[SecuritySchemeType.MUTUAL_TLS] = MutualTLSAuthenticator(
            config.mutual_tls, config.network
        )
    return [by_scheme[scheme] for scheme in config.scheme_order if scheme in by_scheme]
