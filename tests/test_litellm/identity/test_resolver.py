import base64
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath("../.."))

from litellm.constants import LITTELM_CLI_SERVICE_ACCOUNT_NAME
from litellm.identity.principal import (
    AnonymousPrincipal,
    ApiKeyPrincipal,
    JWTPrincipal,
    ServiceAccountPrincipal,
)
from litellm.identity.resolver import resolve_identity


def _fake_request(headers=None, client_host=None):
    return SimpleNamespace(
        headers=headers or {},
        client=SimpleNamespace(host=client_host) if client_host else None,
    )


def _jwt(claims):
    def b(d):
        return (
            base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
        )

    return f"{b({'alg':'HS256','typ':'JWT'})}.{b(claims)}.sig"


def test_anonymous_when_no_credentials():
    ctx = resolve_identity()
    assert isinstance(ctx.principal, AnonymousPrincipal)


def test_api_key_principal_for_sk_key():
    ctx = resolve_identity(api_key="sk-test")
    assert isinstance(ctx.principal, ApiKeyPrincipal)


def test_jwt_principal_for_jwt_shaped_key():
    ctx = resolve_identity(api_key=_jwt({"sub": "u1"}))
    assert isinstance(ctx.principal, JWTPrincipal)
    assert ctx.principal.sub == "u1"


def test_service_account_for_known_sentinel():
    ctx = resolve_identity(api_key=LITTELM_CLI_SERVICE_ACCOUNT_NAME)
    assert isinstance(ctx.principal, ServiceAccountPrincipal)
    assert ctx.principal.name == LITTELM_CLI_SERVICE_ACCOUNT_NAME


def test_end_user_and_audit_propagate():
    ctx = resolve_identity(
        body={"user": "eu-42"},
        headers={"litellm-changed-by": "admin"},
    )
    assert ctx.end_user_id == "eu-42"
    assert ctx.audit.changed_by == "admin"


def test_client_info_from_request():
    req = _fake_request({"user-agent": "curl/8"}, client_host="127.0.0.1")
    ctx = resolve_identity(request=req)
    assert ctx.client.ip == "127.0.0.1"
    assert ctx.client.user_agent == "curl/8"
