import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

from pydantic import JsonValue
from starlette.types import ASGIApp

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.models.workflow import WorkflowExecution, WorkflowRunState, WorkflowStore
from litellm.models.workflow_bundle import WorkflowFailure
from litellm.proxy._types import ProxyLifespanState
from litellm.proxy.workflows.child import GatewayChildCaller
from litellm.proxy.workflows.monty.runtime import MontyRuntime
from litellm.proxy.workflows.runner import ChildCaller, execute_run
from litellm.proxy.workflows.validation import DEFAULT_MONTY_LIMITS
from litellm.repositories.workflow_repository import WorkflowRepository


async def maintain_lease(
    store: WorkflowStore, execution: WorkflowExecution, work: asyncio.Task[JsonValue | WorkflowFailure]
) -> None:
    try:
        while not work.done():
            await asyncio.sleep(2)
            current = await store.execution(execution.run_id)
            if (
                current is None
                or current.cancel_requested
                or not await store.renew(execution.run_id, execution.lease_owner or "")
            ):
                work.cancel()
                return
    except Exception:  # noqa: BLE001  # Any persistence failure must stop the worker before its lease expires
        work.cancel()
        verbose_proxy_logger.exception("Workflow lease renewal failed; stopping dispatch")


async def format_response(store: WorkflowStore, execution: WorkflowExecution, result: JsonValue) -> str:
    steps: Final = await store.steps(execution.run_id)
    prompt_tokens: Final = sum(step.prompt_tokens for step in steps)
    completion_tokens: Final = sum(step.completion_tokens for step in steps)
    return json.dumps(
        {
            "id": f"chatcmpl-{execution.run_id}",
            "object": "chat.completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": f"workflow/{execution.name}",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result if isinstance(result, str) else json.dumps(result, allow_nan=False),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
        allow_nan=False,
    )


async def dispatch_one(
    store: WorkflowStore, caller: ChildCaller, runtime: MontyRuntime, execution: WorkflowExecution
) -> None:
    work: Final = asyncio.create_task(execute_run(store, caller, runtime, execution))
    heartbeat: Final = asyncio.create_task(maintain_lease(store, execution, work))
    try:
        result: Final = await work
        if isinstance(result, WorkflowFailure):
            await store.finish(
                execution.run_id,
                WorkflowRunState.CANCELLED if result.code == "cancelled" else WorkflowRunState.FAILED,
                error=result.model_dump_json(),
                owner=execution.lease_owner or "",
            )
            return
        await store.finish(
            execution.run_id,
            WorkflowRunState.COMPLETED,
            response_json=await format_response(store, execution, result),
            owner=execution.lease_owner or "",
        )
    except asyncio.CancelledError:
        current: Final = await store.execution(execution.run_id)
        if current is not None and current.cancel_requested and current.lease_owner == execution.lease_owner:
            await store.finish(execution.run_id, WorkflowRunState.CANCELLED, owner=execution.lease_owner or "")
        elif (task := asyncio.current_task()) is not None and task.cancelling():
            raise
    finally:
        work.cancel()
        heartbeat.cancel()
        await asyncio.gather(work, heartbeat, return_exceptions=True)


async def worker(store: WorkflowStore, caller: ChildCaller, runtime: MontyRuntime, owner: str) -> None:
    while True:
        try:
            execution = await store.lease_next(f"{owner}/{uuid.uuid4()}")
            if execution is None:
                await asyncio.sleep(0.25)
                continue
            await dispatch_one(store, caller, runtime, execution)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  # Durable unfinished runs must remain recoverable after unexpected failures
            verbose_proxy_logger.exception("Workflow dispatcher failed; unfinished work remains lease-recoverable")
            await asyncio.sleep(1)


@dataclass(frozen=True, slots=True)
class WorkflowDispatcher:
    tasks: tuple[asyncio.Task[None], ...]

    async def aclose(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


def start_dispatcher(
    store: WorkflowStore, app: ASGIApp, state: ProxyLifespanState, runtime: MontyRuntime
) -> WorkflowDispatcher:
    caller: Final = GatewayChildCaller(app, state)
    owner: Final = str(uuid.uuid4())
    return WorkflowDispatcher(
        tasks=tuple(asyncio.create_task(worker(store, caller, runtime, f"{owner}/{index}")) for index in range(4))
    )


@asynccontextmanager
async def workflow_dispatch_lifespan(
    state: ProxyLifespanState, app: ASGIApp, enabled: bool
) -> AsyncGenerator[None, None]:
    database: Final = state["database"]
    if not enabled or database is None:
        yield
        return
    runtime: Final = MontyRuntime(DEFAULT_MONTY_LIMITS)
    failure: Final = await runtime.start()
    if failure is not None:
        raise RuntimeError(f"Cannot enable workflows, sandbox unavailable ({failure.kind.value})")
    dispatcher: Final = start_dispatcher(WorkflowRepository(database.prisma), app, state, runtime)
    try:
        yield
    finally:
        await dispatcher.aclose()
        await runtime.close()
