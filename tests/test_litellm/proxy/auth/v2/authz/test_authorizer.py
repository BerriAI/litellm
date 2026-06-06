import logging

import pytest

from litellm.proxy.auth.v2.authz.authorizer import AuthorizationDenied, authorize
from litellm.proxy.auth.v2.principal import Principal

PRINCIPAL = Principal(subject="user:u1", domain="*", groupings=[])


class _Enforcer:
    def __init__(self, ret):
        self.ret = ret
        self.calls = []

    def enforce(self, sub, dom, obj, act):
        self.calls.append((sub, dom, obj, act))
        return self.ret


class _Exploding:
    def enforce(self, *_):
        raise AssertionError("enforce must not be called on a loud-open route")


def test_denied_governed_route_raises():
    enforcer = _Enforcer(ret=False)
    with pytest.raises(AuthorizationDenied) as exc:
        authorize(PRINCIPAL, "/model/delete", {"model_id": "m9"}, enforcer)
    assert "delete" in str(exc.value)
    assert "model:m9" in str(exc.value)


def test_allowed_governed_route_passes():
    enforcer = _Enforcer(ret=True)
    authorize(PRINCIPAL, "/model/new", {}, enforcer)  # must not raise
    assert enforcer.calls == [("user:u1", "*", "model:*", "write")]


def test_object_is_built_from_request_id_field():
    enforcer = _Enforcer(ret=True)
    authorize(PRINCIPAL, "/model/update", {"model_id": "abc123"}, enforcer)
    assert enforcer.calls[0][2] == "model:abc123"


def test_object_falls_back_to_wildcard_without_id():
    enforcer = _Enforcer(ret=True)
    authorize(PRINCIPAL, "/model/info", {}, enforcer)
    assert enforcer.calls[0][2] == "model:*"


def test_ungoverned_route_is_loud_open_and_never_enforces(caplog):
    with caplog.at_level(logging.WARNING, logger="litellm.proxy.auth.v2"):
        # _Exploding asserts enforce() is not reached; no raise expected.
        authorize(PRINCIPAL, "/chat/completions", {}, _Exploding())
    assert "not yet protected" in caplog.text
    assert "/chat/completions" in caplog.text


def test_rest_route_is_enforced_when_method_is_threaded():
    # Regression: a REST route (credentials) only resolves with the HTTP method.
    # If authorize() dropped the method it would loud-open and bypass enforcement.
    enforcer = _Enforcer(ret=True)
    authorize(PRINCIPAL, "/credentials", {}, enforcer, "POST")
    assert enforcer.calls == [("user:u1", "*", "credential:*", "write")]


def test_rest_route_without_method_is_not_silently_enforced(caplog):
    # Without the method the REST route can't resolve; it must loud-open, not 500.
    with caplog.at_level(logging.WARNING, logger="litellm.proxy.auth.v2"):
        authorize(PRINCIPAL, "/credentials", {}, _Exploding())
    assert "not yet protected" in caplog.text
