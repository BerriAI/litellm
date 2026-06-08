import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

from litellm.identity.principal import (
    AnonymousPrincipal,
    ApiKeyPrincipal,
    JWTPrincipal,
    SSOPrincipal,
    ServiceAccountPrincipal,
)


def test_principal_kind_discriminators_are_fixed():
    assert ApiKeyPrincipal(token_hash="x").kind == "api_key"
    assert JWTPrincipal().kind == "jwt"
    assert SSOPrincipal(sso_user_id="s").kind == "sso"
    assert ServiceAccountPrincipal(name="n").kind == "service_account"
    assert AnonymousPrincipal().kind == "anonymous"


def test_principals_are_frozen():
    p = ApiKeyPrincipal(token_hash="x", user_id="u1")
    with pytest.raises(Exception):
        p.user_id = "u2"  # type: ignore[misc]


def test_principals_are_hashable():
    a = ApiKeyPrincipal(token_hash="x", user_id="u1")
    b = ApiKeyPrincipal(token_hash="x", user_id="u1")
    assert {a, b} == {a}


def test_kind_is_not_constructor_arg():
    with pytest.raises(TypeError):
        ApiKeyPrincipal(kind="api_key", token_hash="x")  # type: ignore[call-arg]
