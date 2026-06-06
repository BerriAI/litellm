import dataclasses
from types import SimpleNamespace

import pytest

from litellm.proxy.auth.v2.context import (
    AuthMethod,
    RequestAuthContext,
    attach_end_user,
    get_auth_context,
    set_auth_context,
    try_get_auth_context,
)
from litellm.proxy.auth.v2.principal import Principal

PRINCIPAL = Principal(subject="user:u1", domain="*", groupings=[])


def _request():
    # request.state is the only surface the accessors touch; a namespace stands in.
    return SimpleNamespace(state=SimpleNamespace())


def _context(**overrides):
    base = dict(
        identity=SimpleNamespace(user_id="u1"),
        principal=PRINCIPAL,
        auth_method=AuthMethod.VIRTUAL_KEY,
        route="/chat/completions",
    )
    base.update(overrides)
    return RequestAuthContext(**base)


def test_set_then_get_roundtrips():
    request = _request()
    ctx = _context()
    set_auth_context(request, ctx)
    assert get_auth_context(request) is ctx


def test_get_without_context_raises():
    with pytest.raises(LookupError):
        get_auth_context(_request())


def test_try_get_without_context_returns_none():
    assert try_get_auth_context(_request()) is None


def test_context_is_frozen():
    ctx = _context()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.route = "/mutated"  # type: ignore[misc]


def test_attach_end_user_replaces_without_mutating_original():
    request = _request()
    original = _context()
    set_auth_context(request, original)

    updated = attach_end_user(request, "customer-42")

    assert updated.end_user_id == "customer-42"
    assert get_auth_context(request).end_user_id == "customer-42"
    # The original frozen instance is untouched; attach produced a copy.
    assert original.end_user_id is None
    # Everything else carries over unchanged.
    assert updated.identity is original.identity
    assert updated.auth_method is AuthMethod.VIRTUAL_KEY


def test_attach_end_user_requires_existing_context():
    with pytest.raises(LookupError):
        attach_end_user(_request(), "customer-42")


def test_auth_method_values_are_stable():
    # Recorded on spend logs / telemetry, so the string values are a contract.
    assert AuthMethod.VIRTUAL_KEY.value == "virtual_key"
    assert AuthMethod.MASTER_KEY.value == "master_key"
    assert AuthMethod.JWT.value == "jwt"
    assert AuthMethod.OAUTH2.value == "oauth2"
    assert AuthMethod.ANONYMOUS.value == "anonymous"
