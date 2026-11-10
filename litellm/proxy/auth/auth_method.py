from __future__ import annotations

from enum import Enum


class AuthMethod(str, Enum):
    API_KEY = "api_key"
    HTTP_BASIC = "http_basic"
    BEARER_JWT = "bearer_jwt"
    OAUTH2_INTROSPECTION = "oauth2_introspection"
    OIDC = "oidc"
    SAML = "saml"
    MUTUAL_TLS = "mutual_tls"
