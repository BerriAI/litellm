from collections.abc import Mapping
from datetime import datetime, timezone

import pytest

from litellm.models.task_insights import Decision, PendingRequest, TaskDefinition
from litellm.task_insights.bucketing import bucket_pending

_AT = datetime(2026, 11, 11, tzinfo=timezone.utc)
_TASK = TaskDefinition(
    task_id="t1",
    name="Debug",
    description="d",
    inclusion_criteria="i",
    exclusion_criteria="e",
    status="frozen",
    source="discovered",
    map_x=1,
    map_y=2,
    created_at=_AT,
)


def _pending(request_id: str, text_hash: str) -> PendingRequest:
    return PendingRequest(
        request_id=request_id, text_hash=text_hash, request_text=f"text {text_hash}", context_text=None
    )


def _decision(task_id: str) -> Decision:
    return Decision(outcome="assigned", task_id=task_id, decision_model="jev")


class _Store:
    def __init__(self, pending, reused: Mapping[str, Decision] | None = None, tasks=(_TASK,)) -> None:
        self.pending = pending
        self.reused = reused or {}
        self.tasks = tasks
        self.saved: tuple[tuple[PendingRequest, Decision], ...] = ()

    async def frozen_definitions(self):
        return self.tasks

    async def pending_requests(self, limit):
        return self.pending[:limit]

    async def decisions_by_hash(self, text_hashes):
        return {key: value for key, value in self.reused.items() if key in text_hashes}

    async def taxonomy_version(self):
        return _AT

    async def save_decisions(self, decisions, bucketed_at, taxonomy_version):
        assert taxonomy_version == _AT
        self.saved = decisions
        return len(decisions)


class _Decider:
    decision_model = "jev"

    def __init__(self, fail_on: frozenset[str] = frozenset()) -> None:
        self.asked: list[str] = []
        self.fail_on = fail_on

    async def decide(self, tasks, request):
        self.asked.append(request.text_hash)
        if request.text_hash in self.fail_on:
            raise RuntimeError("provider down")
        return _decision("t1")


@pytest.mark.asyncio
async def test_requests_sharing_a_text_hash_cost_one_decision():
    store = _Store((_pending("a", "h1"), _pending("b", "h1"), _pending("c", "h2")))
    decider = _Decider()

    assert await bucket_pending(store, decider, now=lambda: _AT, batch_size=10) == 3

    assert sorted(decider.asked) == ["h1", "h2"]
    assert [request.request_id for request, _ in store.saved] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_an_earlier_decision_for_the_same_text_is_reused_without_a_call():
    store = _Store((_pending("a", "h1"),), reused={"h1": _decision("t1")})
    decider = _Decider()

    await bucket_pending(store, decider, now=lambda: _AT, batch_size=10)

    assert decider.asked == []
    assert store.saved[0][1].task_id == "t1"


@pytest.mark.asyncio
async def test_a_failed_decision_is_recorded_and_the_rest_of_the_batch_still_lands():
    store = _Store((_pending("a", "h1"), _pending("b", "h2")))

    await bucket_pending(store, _Decider(fail_on=frozenset({"h1"})), now=lambda: _AT, batch_size=10)

    outcomes = {request.request_id: (decision.outcome, decision.error) for request, decision in store.saved}
    assert outcomes == {"a": ("failed", "RuntimeError"), "b": ("assigned", None)}


@pytest.mark.asyncio
async def test_nothing_is_decided_without_a_frozen_taxonomy():
    store = _Store((_pending("a", "h1"),), tasks=())
    decider = _Decider()

    assert await bucket_pending(store, decider, now=lambda: _AT, batch_size=10) == 0
    assert decider.asked == []
