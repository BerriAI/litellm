"""Regression tests for the deferred parallel slot check on the queued-admission path.

The spec (openspec/changes/litellm-queue-backpressure/design.md, D1) calls for a
router-side check that runs after ``wait_for_turn`` admits a queued request and
before the call is dispatched. When the proxy gate was deferred (opted into the
queue via ``slot_wait_timeout``), the 503/429 that would have fired at proxy
entry must instead fire at queued admission, only if the slot is actually taken
by that moment.

These tests drive ``DeferredParallelSlotCheck.__call__`` against a real
``MaxParallelRequestsLimit`` so a mutation that drops the check or the 503 turns
red.
"""

from typing import Final

import pytest

from litellm.exceptions import RateLimitError, ServiceUnavailableError
from litellm.router_utils.client_initalization_utils import MaxParallelRequestsLimit
from litellm.scheduler import DeferredParallelSlotCheck


class _Deployment:
    __slots__ = ("model_id", "group", "limit")

    def __init__(self, limit: MaxParallelRequestsLimit) -> None:
        self.limit = limit
        self.model_id: Final = limit.model_id
        self.group: Final = limit.model_group


def _check(dep: _Deployment) -> DeferredParallelSlotCheck:
    return DeferredParallelSlotCheck(model_group=dep.group, limits={dep.model_id: dep.limit})


def test_passes_when_slot_free_at_admission() -> None:
    limit = MaxParallelRequestsLimit(max_parallel_requests=2, model_id="dep", model_group="g")
    limit.acquire()
    check = _check(_Deployment(limit))
    check()  # in_flight=1 < max=2, no raise
    limit.release()
    assert limit.in_flight == 0


def test_raises_503_when_slot_taken_at_admission() -> None:
    limit = MaxParallelRequestsLimit(max_parallel_requests=1, model_id="dep", model_group="g")
    limit.acquire()
    check = _check(_Deployment(limit))
    with pytest.raises(ServiceUnavailableError):
        check()


def test_legacy_keying_skips_the_deferred_check() -> None:
    """A queue that does not opt into the queue (deployment_id=None) must not
    have its gate deferred; the proxy gate kept its original fail-fast 503."""
    limit = MaxParallelRequestsLimit(max_parallel_requests=1, model_id="dep", model_group="g")
    limit.acquire()
    check = DeferredParallelSlotCheck(
        model_group="g", limits={"dep": limit}, deferred=False
    )
    check()  # no raise, the check was skipped
    # The original 429 path is still live for non-queued traffic.
    with pytest.raises(RateLimitError):
        limit.acquire()
