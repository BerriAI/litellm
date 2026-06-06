from typing import List

import pytest

from litellm.proxy.auth.v2.audit import (
    AuthzDecision,
    Decision,
    record,
    register_sink,
    reset_sinks,
)
from litellm.proxy.auth.v2.authz.authorizer import AuthorizationDenied, authorize
from litellm.proxy.auth.v2.principal import Principal

PRINCIPAL = Principal(subject="user:u1", domain="*", groupings=[])


@pytest.fixture(autouse=True)
def _clear_sinks():
    reset_sinks()
    yield
    reset_sinks()


def _capture() -> List[AuthzDecision]:
    captured: List[AuthzDecision] = []
    register_sink(captured.append)
    return captured


class _Enforcer:
    def __init__(self, ret: bool):
        self.ret = ret

    def enforce(self, *_args: object) -> bool:
        return self.ret


class _Exploding:
    def enforce(self, *_args: object) -> bool:
        raise AssertionError("enforce must not be called on a loud-open route")


def test_record_delivers_to_registered_sink():
    got = _capture()
    decision = AuthzDecision(
        decision=Decision.ALLOW,
        subject="user:u1",
        domain="*",
        obj="model:x",
        action="read",
        route="/model/info",
        reason="t",
    )
    record(decision)
    assert got == [decision]


def test_failing_sink_is_isolated():
    good: List[AuthzDecision] = []

    def boom(_decision: AuthzDecision) -> None:
        raise RuntimeError("sink down")

    register_sink(boom)
    register_sink(good.append)
    # The failing sink must not stop the good one or raise out of record().
    record(
        AuthzDecision(
            decision=Decision.DENY,
            subject="s",
            domain="*",
            obj="o",
            action="a",
            route="/r",
            reason="t",
        )
    )
    assert len(good) == 1


def test_authorize_records_allow_with_auth_method():
    got = _capture()
    authorize(PRINCIPAL, "/model/new", {}, _Enforcer(True), auth_method="virtual_key")
    assert len(got) == 1
    assert got[0].decision is Decision.ALLOW
    assert got[0].obj == "model:*"
    assert got[0].action == "write"
    assert got[0].auth_method == "virtual_key"


def test_authorize_records_deny_and_still_raises():
    got = _capture()
    with pytest.raises(AuthorizationDenied):
        authorize(PRINCIPAL, "/model/delete", {"model_id": "m9"}, _Enforcer(False))
    assert got[0].decision is Decision.DENY
    assert got[0].obj == "model:m9"


def test_authorize_records_loud_open_for_ungoverned_route():
    got = _capture()
    authorize(PRINCIPAL, "/chat/completions", {}, _Exploding())
    assert got[0].decision is Decision.LOUD_OPEN
    assert got[0].route == "/chat/completions"
