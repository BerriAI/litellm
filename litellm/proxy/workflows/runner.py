import asyncio
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Final, Protocol
from uuid import uuid4

from pydantic import JsonValue

from litellm._logging import verbose_proxy_logger
from litellm.models.workflow_definition import (
    Binding,
    ChatCompletionNode,
    ValueBinding,
    WorkflowBundle,
    WorkflowDefinitionStore,
    WorkflowNode,
)
from litellm.models.workflow_execution import (
    ChildCallResult,
    EngineRun,
    ExecutionError,
    RunState,
    StepTrace,
    WorkflowExecutionStore,
    workflow_call_id,
)
from litellm.proxy.workflows.monty.compiler import compile_program
from litellm.proxy.workflows.monty.models import MontyFailure, MontyLimits
from litellm.proxy.workflows.monty.runtime import MontyRuntime
from litellm.proxy.workflows.monty.validation import ValidatorProcess


class ChatExecutor(Protocol):
    async def __call__(
        self,
        *,
        key_hash: str,
        team_id: str,
        workflow_id: str,
        node: ChatCompletionNode,
        messages: JsonValue,
        run_id: str,
        session_id: str,
        attempt: int,
        deadline: datetime,
        end_user_id: str | None,
    ) -> ChildCallResult | ExecutionError: ...


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SchemaValidator(Protocol):
    async def __call__(self, messages: JsonValue, schema: Mapping[str, JsonValue]) -> ExecutionError | None: ...


async def _validate_run_input(messages: JsonValue, schema: Mapping[str, JsonValue]) -> ExecutionError | None:
    validator: Final = ValidatorProcess()
    try:
        failure: Final = await validator.validate(messages, dict(schema), deadline=time.monotonic() + 5)
        return (
            ExecutionError("invalid_input", "Workflow messages did not satisfy the pinned input schema")
            if failure is not None
            else None
        )
    finally:
        await validator.close()


def resolve_binding(binding: Binding, run: EngineRun, steps: tuple[StepTrace, ...]) -> JsonValue | ExecutionError:
    if isinstance(binding, ValueBinding):
        return binding.value
    if binding.source == "$input":
        return [dict(message) for message in run.input.messages]
    source: Final = next((step for step in steps if step.node_id == binding.source), None)
    if source is None or source.state is not RunState.COMPLETED:
        return ExecutionError("missing_input", "The referenced node has no completed output")
    return source.output


def _bound_inputs(
    bindings: Mapping[str, Binding], run: EngineRun, steps: tuple[StepTrace, ...]
) -> JsonValue | ExecutionError:
    resolved: Final = {name: resolve_binding(binding, run, steps) for name, binding in bindings.items()}
    failure: Final = next((value for value in resolved.values() if isinstance(value, ExecutionError)), None)
    if failure is not None:
        return failure
    return {name: value for name, value in resolved.items() if not isinstance(value, ExecutionError)}


async def _settle_task(task: asyncio.Task[object]) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


class WorkflowRunner:
    def __init__(
        self,
        runs: WorkflowExecutionStore,
        definitions: WorkflowDefinitionStore,
        execute_chat: ChatExecutor,
        *,
        clock: Callable[[], datetime] = utc_now,
        validate_input: SchemaValidator = _validate_run_input,
    ) -> None:
        self.runs: Final = runs
        self.definitions: Final = definitions
        self._execute_chat: Final = execute_chat
        self._clock: Final = clock
        self._validate_input: Final = validate_input
        self._monty: Final = MontyRuntime(MontyLimits())
        self._monty_start: Final = asyncio.Lock()
        self._poller: asyncio.Task[None] | None = None
        self._active: tuple[asyncio.Task[None], ...] = ()

    async def start(self) -> None:
        self._poller = asyncio.create_task(self._poll())

    async def close(self) -> None:
        if self._poller is not None:
            self._poller.cancel()
            await asyncio.gather(self._poller, return_exceptions=True)
        for task in self._active:
            task.cancel()
        await asyncio.gather(*self._active, return_exceptions=True)
        await self._monty.close()

    async def _poll(self) -> None:
        while True:
            for task in self._active:
                if task.done() and not task.cancelled() and task.exception() is not None:
                    verbose_proxy_logger.error("Workflow execution task failed before settlement")
            self._active = tuple(task for task in self._active if not task.done())
            try:
                pending: tuple[EngineRun, ...] = (
                    await self.runs.recover(4 - len(self._active)) if len(self._active) < 4 else ()
                )
                self._active = (*self._active, *(asyncio.create_task(self._run(run.run_id)) for run in pending))
            except Exception:  # noqa: BLE001  # polling must survive transient database failures without logging inputs
                verbose_proxy_logger.error("Workflow dispatch polling failed")
            await asyncio.sleep(1)

    async def _heartbeat(self, run: EngineRun, token: str) -> ExecutionError:
        while True:
            await asyncio.sleep(1)
            current: EngineRun | None = await self.runs.get(run.run_id)
            if current is None or current.lease_token != token:
                return ExecutionError("lease_lost", "Execution ownership changed")
            if current.cancel_requested:
                return ExecutionError("cancelled", "Cancellation requested")
            if self._clock() >= run.input.deadline:
                return ExecutionError("deadline_exceeded", "Workflow deadline expired")
            renewed: bool = await self.runs.renew(run.run_id, token, self._clock() + timedelta(seconds=15))
            if not renewed:
                return ExecutionError("lease_lost", "Execution ownership expired")

    async def _run(self, run_id: str) -> None:
        token: Final = str(uuid4())
        run: Final = await self.runs.claim(run_id, token, self._clock() + timedelta(seconds=15))
        if run is None:
            return
        if run.cancel_requested:
            await self.runs.finish(run_id, token, RunState.CANCELLED, None, "Cancellation requested")
            return
        execution: Final = asyncio.create_task(self._execute(run, token))
        heartbeat: Final = asyncio.create_task(self._heartbeat(run, token))
        try:
            done, _ = await asyncio.wait((execution, heartbeat), return_when=asyncio.FIRST_COMPLETED)
            if execution in done:
                result: Final = execution.result()
                await self.runs.finish(
                    run_id,
                    token,
                    RunState.FAILED if isinstance(result, ExecutionError) else RunState.COMPLETED,
                    None if isinstance(result, ExecutionError) else result,
                    result.message if isinstance(result, ExecutionError) else None,
                )
            else:
                reason: Final = heartbeat.result()
                await _settle_task(execution)
                await self.runs.finish(
                    run_id,
                    token,
                    RunState.CANCELLED if reason.code == "cancelled" else RunState.FAILED,
                    None,
                    reason.message,
                )
        except asyncio.CancelledError:
            await _settle_task(execution)
            await self.runs.finish(run_id, token, RunState.FAILED, None, "Execution worker stopped")
            raise
        except Exception:  # noqa: BLE001  # settle durable execution without exposing provider errors or credentials
            await _settle_task(execution)
            await self.runs.finish(run_id, token, RunState.FAILED, None, "Execution could not complete")
        finally:
            await _settle_task(heartbeat)

    async def _execute(self, run: EngineRun, token: str) -> JsonValue | ExecutionError:
        snapshot: Final = await self.definitions.get_snapshot(run.snapshot_id, run.team_id)
        if snapshot is None or snapshot.workflow_id != run.workflow_id:
            return ExecutionError("snapshot_unavailable", "The pinned workflow definition is unavailable")
        invalid: Final = await self._validate_input(
            [dict(message) for message in run.input.messages], snapshot.bundle.input_schema
        )
        if invalid is not None:
            return invalid
        steps: Final = await self.runs.steps(run.run_id)
        if any(step.state is not RunState.COMPLETED for step in steps):
            return ExecutionError(
                "interrupted_execution", "A previous worker stopped before recording a completed node outcome"
            )
        return await self._sequence(run, token, snapshot.bundle, steps, 0)

    async def _sequence(
        self, run: EngineRun, token: str, bundle: WorkflowBundle, steps: tuple[StepTrace, ...], index: int
    ) -> JsonValue | ExecutionError:
        if self._clock() >= run.input.deadline:
            return ExecutionError("deadline_exceeded", "Workflow deadline expired")
        if index == len(bundle.nodes):
            return resolve_binding(bundle.output, run, steps)
        node: Final = bundle.nodes[index]
        recorded: Final = next((step for step in steps if step.node_id == node.id), None)
        if recorded is not None:
            return await self._sequence(run, token, bundle, steps, index + 1)
        value: Final = (
            resolve_binding(node.messages, run, steps)
            if isinstance(node, ChatCompletionNode)
            else _bound_inputs(node.inputs, run, steps)
        )
        if isinstance(value, ExecutionError):
            return value
        started: Final = self._clock()
        step: Final = StepTrace(
            step_id=str(uuid4()),
            run_id=run.run_id,
            node_id=node.id,
            state=RunState.RUNNING,
            input=value,
            attempts=1,
            deadline=min(run.input.deadline, started + timedelta(seconds=node.timeout_seconds)),
            started_at=started,
            call_id=workflow_call_id(run.run_id, node.id, 1) if isinstance(node, ChatCompletionNode) else None,
        )
        completed: Final = await self._attempt(run, token, node, step)
        if isinstance(completed, ExecutionError):
            return completed
        return await self._sequence(run, token, bundle, (*steps, completed), index + 1)

    async def _attempt(
        self, run: EngineRun, token: str, node: WorkflowNode, step: StepTrace
    ) -> StepTrace | ExecutionError:
        if not await self.runs.checkpoint(run.run_id, token, step):
            return ExecutionError("lease_lost", "Execution ownership expired before dispatch")
        result: Final = await self._recorded_node_result(run, token, node, step)
        receipt: Final = result.receipt if isinstance(result, ExecutionError) else result
        finished: Final = step.model_copy(
            update={
                **(
                    {
                        "state": RunState.FAILED,
                        "finished_at": self._clock(),
                        "error": result.message,
                    }
                    if isinstance(result, ExecutionError)
                    else {
                        "state": RunState.COMPLETED,
                        "finished_at": self._clock(),
                        "output": result.output,
                    }
                ),
                "prompt_tokens": receipt.prompt_tokens if receipt is not None else 0,
                "completion_tokens": receipt.completion_tokens if receipt is not None else 0,
                "response_cost": receipt.response_cost if receipt is not None else None,
                "request_id": receipt.request_id if receipt is not None else None,
            }
        )
        if not await self.runs.checkpoint(run.run_id, token, finished):
            return ExecutionError("lease_lost", "Execution ownership expired before the checkpoint")
        if not isinstance(result, ExecutionError):
            return finished
        retryable: Final = result.code in {"provider_error", "rate_limit", "runtime_error", "worker_crashed"}
        remaining_after_backoff: Final = (step.deadline - self._clock()).total_seconds() - node.retry.backoff_seconds
        if not retryable or step.attempts >= node.retry.max_attempts or remaining_after_backoff <= 0:
            return result
        await asyncio.sleep(node.retry.backoff_seconds)
        return await self._attempt(
            run,
            token,
            node,
            step.model_copy(
                update={
                    "attempts": step.attempts + 1,
                    "call_id": (
                        workflow_call_id(run.run_id, node.id, step.attempts + 1)
                        if isinstance(node, ChatCompletionNode)
                        else None
                    ),
                }
            ),
        )

    async def _recorded_node_result(
        self, run: EngineRun, token: str, node: WorkflowNode, step: StepTrace
    ) -> ChildCallResult | ExecutionError:
        try:
            return await self._timed_node_result(run, node, step)
        except asyncio.CancelledError:
            current: Final = await self.runs.get(run.run_id)
            await self.runs.checkpoint(
                run.run_id,
                token,
                step.model_copy(
                    update={
                        "state": (
                            RunState.CANCELLED if current is not None and current.cancel_requested else RunState.FAILED
                        ),
                        "finished_at": self._clock(),
                        "error": "Node execution interrupted",
                    }
                ),
            )
            raise
        except Exception:  # noqa: BLE001  # convert unexpected node failures into a sanitized terminal checkpoint
            return ExecutionError("execution_failed", "The node could not complete")

    async def _timed_node_result(
        self, run: EngineRun, node: WorkflowNode, step: StepTrace
    ) -> ChildCallResult | ExecutionError:
        if isinstance(node, ChatCompletionNode):
            return await self._node_result(run, node, step)
        try:
            remaining: Final = max(0, (step.deadline - self._clock()).total_seconds())
            return await asyncio.wait_for(self._node_result(run, node, step), remaining)
        except (TimeoutError, asyncio.TimeoutError):
            return ExecutionError("deadline_exceeded", "Node deadline expired")

    async def _node_result(
        self, run: EngineRun, node: WorkflowNode, step: StepTrace
    ) -> ChildCallResult | ExecutionError:
        if isinstance(node, ChatCompletionNode):
            return await self._execute_chat(
                key_hash=run.created_by,
                team_id=run.team_id,
                workflow_id=run.workflow_id,
                node=node,
                messages=step.input,
                run_id=run.run_id,
                session_id=run.session_id,
                attempt=step.attempts,
                deadline=step.deadline,
                end_user_id=run.input.end_user_id,
            )
        compiled: Final = compile_program(node.program, self._monty.limits)
        if isinstance(compiled, MontyFailure):
            return ExecutionError("invalid_program", compiled.message)
        async with self._monty_start:
            availability: Final = await self._monty.start()
        if availability is not None:
            return ExecutionError("runtime_unavailable", "The Monty runtime is unavailable")
        outcome: Final = await self._monty.run_program(
            compiled,
            step.input,
            deadline_monotonic=time.monotonic() + max(0, (step.deadline - self._clock()).total_seconds()),
        )
        if isinstance(outcome, MontyFailure):
            return ExecutionError(outcome.kind.value.lower(), outcome.message)
        return ChildCallResult(outcome.output, 0, 0, None, None)
