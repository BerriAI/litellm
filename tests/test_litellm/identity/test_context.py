import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from litellm.identity.context import (
    AuditInfo,
    ClientInfo,
    IdentityContext,
    RequestIds,
)
from litellm.identity.principal import AnonymousPrincipal, ApiKeyPrincipal


def test_default_principal_is_anonymous():
    ctx = IdentityContext()
    assert isinstance(ctx.principal, AnonymousPrincipal)
    assert ctx.end_user_id is None
    assert ctx.tags == []
    assert ctx.access_group_ids == []
    assert ctx.request == RequestIds()
    assert ctx.client == ClientInfo()
    assert ctx.audit == AuditInfo()


def test_context_is_mutable():
    ctx = IdentityContext()
    ctx.end_user_id = "eu-1"
    ctx.tags.append("env:prod")
    assert ctx.end_user_id == "eu-1"
    assert "env:prod" in ctx.tags


def test_context_carries_principal():
    p = ApiKeyPrincipal(token_hash="abc", user_id="u1", team_id="t1")
    ctx = IdentityContext(principal=p)
    assert ctx.principal is p


def test_subobjects_are_independent_between_instances():
    a = IdentityContext()
    b = IdentityContext()
    a.tags.append("x")
    assert b.tags == []
    a.client.forwarded_chain.append("1.2.3.4")
    assert b.client.forwarded_chain == []
