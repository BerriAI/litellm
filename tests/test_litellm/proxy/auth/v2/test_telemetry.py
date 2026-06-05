from types import SimpleNamespace

from litellm.proxy.auth.v2.context import AuthMethod, RequestAuthContext
from litellm.proxy.auth.v2.principal import Principal
from litellm.proxy.auth.v2.telemetry import identity_span_attributes

PRINCIPAL = Principal(subject="user:u1", domain="team:eng", groupings=[])


def _context(identity, end_user_id=None):
    return RequestAuthContext(
        identity=identity,
        principal=PRINCIPAL,
        auth_method=AuthMethod.JWT,
        route="/chat/completions",
        end_user_id=end_user_id,
    )


def test_present_identity_fields_become_span_attributes():
    identity = SimpleNamespace(
        user_id="u1", team_id="t1", key_alias="prod-key", org_id="o1"
    )
    attrs = identity_span_attributes(_context(identity, end_user_id="cust-1"))
    assert attrs["litellm.auth.method"] == "jwt"
    assert attrs["litellm.auth.subject"] == "user:u1"
    assert attrs["litellm.user_id"] == "u1"
    assert attrs["litellm.team_id"] == "t1"
    assert attrs["litellm.key_alias"] == "prod-key"
    assert attrs["litellm.org_id"] == "o1"
    assert attrs["litellm.end_user_id"] == "cust-1"


def test_absent_fields_are_omitted_not_emitted_empty():
    identity = SimpleNamespace(user_id="u1", team_id=None, key_alias=None, org_id=None)
    attrs = identity_span_attributes(_context(identity))
    # Always-present anchors.
    assert attrs["litellm.auth.method"] == "jwt"
    assert attrs["litellm.user_id"] == "u1"
    # Empty/missing fields must not appear at all.
    assert "litellm.team_id" not in attrs
    assert "litellm.key_alias" not in attrs
    assert "litellm.org_id" not in attrs
    assert "litellm.end_user_id" not in attrs
