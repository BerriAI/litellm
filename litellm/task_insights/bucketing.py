import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.models.task_insights import Decision, PendingRequest, TaskAnalysisStore, TaskDefinition


class TaskDecider(Protocol):
    @property
    def decision_model(self) -> str: ...

    async def decide(self, tasks: tuple[TaskDefinition, ...], request: PendingRequest) -> Decision: ...


async def _decide(decider: TaskDecider, tasks: tuple[TaskDefinition, ...], request: PendingRequest) -> Decision:
    try:
        return await decider.decide(tasks, request)
    except Exception as error:  # noqa: BLE001  # one failed decision is recorded and retried, never fatal to the batch
        verbose_proxy_logger.warning("Task Insights decision failed for %s: %s", request.request_id, error)
        return Decision(
            outcome="failed", task_id=None, decision_model=decider.decision_model, error=type(error).__name__
        )


async def bucket_pending(
    store: TaskAnalysisStore, decider: TaskDecider, *, now: Callable[[], datetime], batch_size: int
) -> int:
    """Assign pending requests to the frozen taxonomy. Requests sharing a text hash, such as the turns of one agent
    session that repeat the same instruction, share one decision, so a replayed history costs one model call. The
    taxonomy version is read before the definitions, so a freeze that lands mid-batch makes the save a no-op."""
    version: Final = await store.taxonomy_version()
    tasks: Final = await store.frozen_definitions()
    if not tasks:
        return 0
    pending: Final = await store.pending_requests(batch_size)
    if not pending:
        return 0
    reused: Final = await store.decisions_by_hash(tuple(dict.fromkeys(request.text_hash for request in pending)))
    fresh_requests: Final = tuple(
        {request.text_hash: request for request in pending if request.text_hash not in reused}.values()
    )
    fresh: Final = await asyncio.gather(*(_decide(decider, tasks, request) for request in fresh_requests))
    decided: Final = {**reused, **{request.text_hash: decision for request, decision in zip(fresh_requests, fresh)}}
    return await store.save_decisions(
        tuple((request, decided[request.text_hash]) for request in pending), now(), version
    )
