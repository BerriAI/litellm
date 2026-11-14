from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_method import AuthMethod
from litellm.proxy.auth.resolvers.models import PrincipalType
from litellm.proxy.auth.roles import Role
from litellm.proxy.auth.user_api_key_auth import _resolve_request_principal


def _request(
    *,
    headers: Optional[Dict[str, str]] = None,
    client: Optional[Tuple[str, int]] = ("203.0.113.7", 5555),
) -> Request:
    raw: List[Tuple[bytes, bytes]] = [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]
    scope: Dict[str, Any] = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": raw,
        "client": client,
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def test_seam_projects_full_identity_from_key_object():
    token = UserAPIKeyAuth(
        token="hashed-token",
        user_id="u-1",
        user_role="org_admin",
        team_id="t-1",
        team_alias="Eng",
        org_id="o-1",
        organization_alias="Acme",
        end_user_id="cust-9",
    )

    principal = _resolve_request_principal(_request(), token)

    assert principal.principal_type == PrincipalType.HUMAN
    assert principal.auth_method == AuthMethod.API_KEY
    assert principal.user is not None and principal.user.id == "u-1"
    assert principal.roles == [Role.ORG_ADMIN]
    assert [t.id for t in principal.teams] == ["t-1"]
    assert principal.teams[0].name == "Eng"
    assert principal.organization is not None and principal.organization.id == "o-1"
    assert principal.organization.name == "Acme"
    assert principal.end_user is not None and principal.end_user.id == "cust-9"
    # the key is always identifiable via credential_ref, even when other ids exist
    assert principal.credential_ref.token_id == "hashed-token"


def test_seam_principal_is_never_anonymous_for_keyless_service_account():
    # no user_id and no key_alias -> subject must still identify the key
    token = UserAPIKeyAuth(token="hashed-token")

    principal = _resolve_request_principal(_request(), token)

    assert principal.principal_type == PrincipalType.SERVICE_ACCOUNT
    assert principal.user is None
    assert principal.subject == "hashed-token"
    assert principal.credential_ref.token_id == "hashed-token"


def test_seam_stamps_direct_peer_when_no_trusted_proxy_configured():
    token = UserAPIKeyAuth(token="hashed-token", user_id="u-1")

    # No trusted_proxy_ranges configured -> XFF is not trusted, direct peer wins.
    principal = _resolve_request_principal(
        _request(headers={"x-forwarded-for": "10.9.9.9"}, client=("203.0.113.7", 5555)),
        token,
    )

    assert principal.network.client_ip == "203.0.113.7"
    assert principal.network.via_trusted_proxy is False


def test_seam_detects_jwt_auth_method():
    token = UserAPIKeyAuth(
        token="hashed-token", user_id="u-2", jwt_claims={"sub": "u-2"}
    )

    principal = _resolve_request_principal(_request(), token)

    assert principal.auth_method == AuthMethod.BEARER_JWT
