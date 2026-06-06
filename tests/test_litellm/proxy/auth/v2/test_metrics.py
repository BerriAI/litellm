import pytest

from litellm.proxy.auth.v2.audit import Decision
from litellm.proxy.auth.v2.authz.authorizer import AuthorizationDenied, authorize
from litellm.proxy.auth.v2.metrics import metrics
from litellm.proxy.auth.v2.principal import Principal

PRINCIPAL = Principal(subject="user:u1", domain="*", groupings=[])


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


class _Enforcer:
    def __init__(self, ret: bool):
        self.ret = ret

    def enforce(self, *_args: object) -> bool:
        return self.ret


def test_decisions_are_counted_by_decision_resource_action():
    metrics.observe_decision(Decision.ALLOW, "model", "read")
    metrics.observe_decision(Decision.ALLOW, "model", "read")
    metrics.observe_decision(Decision.DENY, "model", "write")
    snap = metrics.snapshot()
    assert snap.decisions["allow:model:read"] == 2
    assert snap.decisions["deny:model:write"] == 1


def test_latency_accumulates_count_and_sum():
    metrics.observe_latency(0.1)
    metrics.observe_latency(0.3)
    snap = metrics.snapshot()
    assert snap.authz_latency_count == 2
    assert snap.authz_latency_sum_seconds == pytest.approx(0.4)


def test_cache_hit_rate_tracking():
    metrics.record_cache(hit=True)
    metrics.record_cache(hit=True)
    metrics.record_cache(hit=False)
    snap = metrics.snapshot()
    assert snap.policy_cache_hits == 2
    assert snap.policy_cache_misses == 1


def test_reset_clears_everything():
    metrics.observe_decision(Decision.ALLOW, "model", "read")
    metrics.observe_latency(0.1)
    metrics.record_cache(hit=True)
    metrics.reset()
    snap = metrics.snapshot()
    assert snap.decisions == {}
    assert snap.authz_latency_count == 0
    assert snap.policy_cache_hits == 0


def test_authorize_feeds_decision_and_latency_metrics():
    # A governed allow records one decision and one latency observation.
    authorize(PRINCIPAL, "/model/new", {}, _Enforcer(True))
    snap = metrics.snapshot()
    assert snap.decisions["allow:model:write"] == 1
    assert snap.authz_latency_count == 1


def test_authorize_deny_is_counted():
    with pytest.raises(AuthorizationDenied):
        authorize(PRINCIPAL, "/model/delete", {"model_id": "m9"}, _Enforcer(False))
    assert metrics.snapshot().decisions["deny:model:delete"] == 1


def test_loud_open_is_counted_without_latency():
    # Ungoverned routes don't reach the enforcer, so no latency is observed.
    class _Exploding:
        def enforce(self, *_args: object) -> bool:
            raise AssertionError("must not enforce")

    authorize(PRINCIPAL, "/chat/completions", {}, _Exploding())
    snap = metrics.snapshot()
    assert snap.decisions["loud_open:route:*"] == 1
    assert snap.authz_latency_count == 0
