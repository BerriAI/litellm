from types import SimpleNamespace

import pytest

from litellm.proxy.auth.v2.context import (
    AuthMethod,
    RequestAuthContext,
    get_auth_context,
    set_auth_context,
)
from litellm.proxy.auth.v2.end_user import resolve_end_user
from litellm.proxy.auth.v2.principal import Principal


def _request_with_context():
    request = SimpleNamespace(state=SimpleNamespace())
    set_auth_context(
        request,
        RequestAuthContext(
            identity=SimpleNamespace(user_id="u1"),
            principal=Principal(subject="user:u1", domain="*", groupings=[]),
            auth_method=AuthMethod.VIRTUAL_KEY,
            route="/chat/completions",
        ),
    )
    return request


@pytest.mark.asyncio
async def test_resolved_end_user_is_attached_to_context():
    request = _request_with_context()
    result = await resolve_end_user(
        request, {"user": "cust-1"}, extractor=lambda data, headers: data.get("user")
    )
    assert result == "cust-1"
    assert get_auth_context(request).end_user_id == "cust-1"


@pytest.mark.asyncio
async def test_no_end_user_returns_none_and_leaves_context_unset():
    request = _request_with_context()
    result = await resolve_end_user(request, {}, extractor=lambda data, headers: None)
    assert result is None
    assert get_auth_context(request).end_user_id is None


@pytest.mark.asyncio
async def test_validator_can_transform_or_reject():
    request = _request_with_context()

    async def validator(raw):
        return f"validated:{raw}"

    result = await resolve_end_user(
        request,
        {"user": "cust-9"},
        extractor=lambda data, headers: data.get("user"),
        validator=validator,
    )
    assert result == "validated:cust-9"
    assert get_auth_context(request).end_user_id == "validated:cust-9"
