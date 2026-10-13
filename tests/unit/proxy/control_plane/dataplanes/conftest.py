from typing import Final

import pytest

from litellm.litellm_core_utils.oauth_token_exchange import OAuthTokenExchangeConfig
from litellm.proxy.control_plane.auth.rbac import (
    ADMIN_ROLE,
    Principal,
    RolePolicy,
    TokenKind,
    build_policy,
    build_principal,
)
from litellm.proxy.control_plane.dataplanes.credentials import Caller

POLICY: Final = build_policy({}, include_defaults=True)
assert isinstance(POLICY, RolePolicy)


def principal(
    kind: TokenKind, subject: str = "alice", email: str | None = "alice@x.test", *dataplanes: str
) -> Principal:
    return build_principal(POLICY, subject, email, frozenset({ADMIN_ROLE}), None, frozenset(dataplanes), kind)


def caller(kind: TokenKind, bearer: str = "cp-bearer", *dataplanes: str) -> Caller:
    return Caller(principal=principal(kind, "alice", "alice@x.test", *dataplanes), bearer=bearer)


@pytest.fixture
def exchange_config() -> OAuthTokenExchangeConfig:
    return OAuthTokenExchangeConfig(
        token_endpoint="https://idp.test/token",
        client_id="cp",
        client_secret="s",
        profile="rfc8693",
        scope=None,
        audience="dp",
    )
