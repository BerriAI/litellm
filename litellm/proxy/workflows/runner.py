import asyncio
import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Final, Protocol
from uuid import uuid4

from fastapi import Request
from pydantic import JsonValue, TypeAdapter

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
    RunInput,
    RunState,
    StepTrace,
    WorkflowExecutionStore,
    workflow_call_id,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.workflows.monty.compiler import compile_program
from litellm.proxy.workflows.monty.models import MontyFailure, MontyLimits
from litellm.proxy.workflows.monty.runtime import MontyRuntime
from litellm.proxy.workflows.monty.validation import ValidatorProcess
from litellm.types.utils import ModelResponse


class ChatExecutor(Protocol):
    async def __call__(
        self,
        request: Request,
        run: EngineRun,
        node: ChatCompletionNode,
        step: StepTrace,
    ) -> ChildCallResult | ExecutionError: ...


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_binding(binding: Binding, run: EngineRun, steps: tuple[StepTrace, ...]) -> JsonValue | ExecutionError:
    if isinstance(binding, ValueBinding):
        return binding.value
    if binding.source == "$input":
        return [dict(message) for message in run.input.messages]
    source: Final = next((step for step in steps if step.node_id == binding.source), None)
    return source.output if source is not None else ExecutionError("missing_input", "Node output is unavailable")


def _inputs(node: WorkflowNode, run: EngineRun, steps: tuple[StepTrace, ...]) -> JsonValue | ExecutionError:
    if isinstance(node, ChatCompletionNode):
        return resolve_binding(node.messages, run, steps)
    resolved: Final = {name: resolve_binding(binding, run, steps) for name, binding in node.inputs.items()}
    failure: Final = next((value for value in resolved.values() if isinstance(value, ExecutionError)), None)
    return failure if failure is not None else {k: v for k, v in resolved.items() if not isinstance(v, ExecutionError)}


def _settled(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        verbose_proxy_logger.error("Workflow execution could not persist its final status")


class WorkflowRunner:
    def __init__(
        self,
        runs: WorkflowExecutionStore,
        definitions: WorkflowDefinitionStore,
        execute_chat: ChatExecutor,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.runs: Final = runs
        self.definitions: Final = definitions
        self._execute_chat: Final = execute_chat
        self._clock: Final = clock
        self._slots: Final = asyncio.Semaphore(4)
        self._monty: Final = MontyRuntime(MontyLimits())
        self._monty_start: Final = asyncio.Lock()
        self._active: tuple[asyncio.Task[None], ...] = ()

    async def submit(self, run: EngineRun, request: Request, caller: UserAPIKeyAuth) -> ExecutionError | None:
        self._active = tuple(task for task in self._active if not task.done())
        if len(self._active) >= 64:
            await self.runs.finish(run.run_id, RunState.failed, None, "Workflow capacity reached")
            return ExecutionError("capacity_exceeded", "Workflow capacity reached")
        task: Final = asyncio.create_task(self._run(run, request, caller))
        task.add_done_callback(_settled)
        self._active = (*self._active, task)
        return None

    async def close(self) -> None:
        if self._active:
            _, pending = await asyncio.wait(self._active, timeout=30)
            for task in pending:
                task.cancel()
        await asyncio.gather(*self._active, return_exceptions=True)
        await self._monty.close()

    async def _watch(self, run_id: str) -> ExecutionError:
        while True:
            await asyncio.sleep(0.5)
            current: EngineRun | None = await self.runs.get(run_id)
            if current is None or current.status in {RunState.failed, RunState.cancelled}:
                return ExecutionError("interrupted", "Execution stopped or its deadline expired")
            if current.cancel_requested:
                return ExecutionError("cancelled", "Cancellation requested")

    async def _run(self, run: EngineRun, request: Request, caller: UserAPIKeyAuth) -> None:
        execution: Final = asyncio.create_task(self._execute(run, request, caller))
        watch: Final = asyncio.create_task(self._watch(run.run_id))
        try:
            done, _ = await asyncio.wait((execution, watch), return_when=asyncio.FIRST_COMPLETED)
            result: Final = execution.result() if execution in done else watch.result()
            state: Final = (
                RunState.cancelled
                if isinstance(result, ExecutionError) and result.code == "cancelled"
                else RunState.failed
                if isinstance(result, ExecutionError)
                else RunState.completed
            )
            await self.runs.finish(
                run.run_id,
                state,
                None if isinstance(result, ExecutionError) else result,
                result.message if isinstance(result, ExecutionError) else None,
            )
        except asyncio.CancelledError:
            await self.runs.finish(run.run_id, RunState.failed, None, "Execution worker stopped")
            raise
        except Exception:  # noqa: BLE001  # settle background failures without exposing inputs or provider credentials
            await self.runs.finish(run.run_id, RunState.failed, None, "Execution could not complete")
            verbose_proxy_logger.error("Workflow execution failed")
        finally:
            execution.cancel()
            watch.cancel()
            await asyncio.gather(execution, watch, return_exceptions=True)

    async def _execute(self, run: EngineRun, request: Request, caller: UserAPIKeyAuth) -> JsonValue | ExecutionError:
        from litellm.proxy.common_request_processing import process_workflow_request

        async with self._slots:
            snapshot: Final = await self.definitions.get_snapshot(run.snapshot_id, run.team_id)
            if snapshot is None or snapshot.workflow_id != run.workflow_id:
                return ExecutionError("snapshot_unavailable", "Pinned definition unavailable")

            async def execute(messages: JsonValue) -> ModelResponse | ExecutionError:
                guarded: Final = run.model_copy(
                    update={
                        "input": RunInput.model_validate({**run.input.model_dump(), "messages": messages}),
                    }
                )
                validator: Final = ValidatorProcess()
                try:
                    invalid: Final = await validator.validate(
                        messages,
                        dict(snapshot.bundle.input_schema),
                        deadline=time.monotonic()
                        + min(5, max(0, (run.input.deadline - self._clock()).total_seconds())),
                    )
                finally:
                    await validator.close()
                if invalid is not None:
                    return ExecutionError("invalid_input", "Messages did not satisfy the workflow input schema")
                output: Final = await self._sequence(guarded, request, snapshot.bundle, ())
                if isinstance(output, ExecutionError):
                    return output
                steps: Final = await self.runs.steps(run.run_id)
                return ModelResponse(
                    id=f"chatcmpl-workflow-{run.run_id}",
                    model=f"workflow/{snapshot.name}",
                    created=int(run.created_at.timestamp()),
                    choices=[
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": output
                                if isinstance(output, str)
                                else json.dumps(output, ensure_ascii=False, allow_nan=False),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    usage={
                        "prompt_tokens": sum(step.prompt_tokens for step in steps),
                        "completion_tokens": sum(step.completion_tokens for step in steps),
                        "total_tokens": sum(step.prompt_tokens + step.completion_tokens for step in steps),
                    },
                )

            response: Final = await process_workflow_request(
                request=request,
                caller=caller,
                workflow_model=f"workflow/{snapshot.name}",
                messages=[dict(message) for message in run.input.messages],
                execute=execute,
            )
            return (
                response
                if isinstance(response, ExecutionError)
                else TypeAdapter[JsonValue](JsonValue).validate_json(response.model_dump_json())
            )

    async def _sequence(
        self,
        run: EngineRun,
        request: Request,
        bundle: WorkflowBundle,
        steps: tuple[StepTrace, ...],
    ) -> JsonValue | ExecutionError:
        if len(steps) == len(bundle.nodes):
            return resolve_binding(bundle.output, run, steps)
        node: Final = bundle.nodes[len(steps)]
        value: Final = _inputs(node, run, steps)
        if isinstance(value, ExecutionError):
            return value
        started: Final = self._clock()
        step: Final = StepTrace(
            step_id=str(uuid4()),
            run_id=run.run_id,
            node_id=node.id,
            state=RunState.running,
            input=value,
            attempts=1,
            started_at=started,
            deadline=min(run.input.deadline, started + timedelta(seconds=node.timeout_seconds)),
        )
        result: Final = await self._attempt(run, request, node, step)
        return (
            result
            if isinstance(result, ExecutionError)
            else await self._sequence(run, request, bundle, (*steps, result))
        )

    async def _attempt(
        self,
        run: EngineRun,
        request: Request,
        node: WorkflowNode,
        step: StepTrace,
    ) -> StepTrace | ExecutionError:
        attempt: Final = step.model_copy(
            update={
                "call_id": workflow_call_id(run.run_id, node.id, step.attempts)
                if isinstance(node, ChatCompletionNode)
                else None,
            }
        )
        if not await self.runs.checkpoint(run.run_id, attempt):
            return ExecutionError("cancelled", "Execution stopped before dispatch")
        try:
            result: Final = await self._safe_node(run, request, node, attempt)
        except asyncio.CancelledError:
            await self.runs.checkpoint(
                run.run_id,
                attempt.model_copy(
                    update={
                        "state": RunState.cancelled,
                        "finished_at": self._clock(),
                        "error": "Node execution interrupted",
                    }
                ),
            )
            raise
        receipt: Final = result.receipt if isinstance(result, ExecutionError) else result
        finished: Final = attempt.model_copy(
            update={
                "state": RunState.failed if isinstance(result, ExecutionError) else RunState.completed,
                "finished_at": self._clock(),
                "error": result.message if isinstance(result, ExecutionError) else None,
                "output": None if isinstance(result, ExecutionError) else result.output,
                "prompt_tokens": receipt.prompt_tokens if receipt else 0,
                "completion_tokens": receipt.completion_tokens if receipt else 0,
                "response_cost": receipt.response_cost if receipt else None,
                "request_id": receipt.request_id if receipt else None,
            }
        )
        await self.runs.checkpoint(run.run_id, finished)
        if not isinstance(result, ExecutionError):
            return finished
        retryable: Final = (
            result.code == "admission_rate_limit"
            if isinstance(node, ChatCompletionNode)
            else result.code in {"runtime_error", "worker_crashed"}
        )
        if (
            not retryable
            or result.receipt is not None
            or step.attempts >= node.retry.max_attempts
            or (step.deadline - self._clock()).total_seconds() <= node.retry.backoff_seconds
        ):
            return result
        await asyncio.sleep(node.retry.backoff_seconds)
        return await self._attempt(run, request, node, step.model_copy(update={"attempts": step.attempts + 1}))

    async def _safe_node(
        self,
        run: EngineRun,
        request: Request,
        node: WorkflowNode,
        step: StepTrace,
    ) -> ChildCallResult | ExecutionError:
        try:
            return await self._node(run, request, node, step)
        except Exception:  # noqa: BLE001  # persist a sanitized terminal step for unexpected runtime failures
            return ExecutionError("execution_failed", "The node could not complete")

    async def _node(
        self,
        run: EngineRun,
        request: Request,
        node: WorkflowNode,
        step: StepTrace,
    ) -> ChildCallResult | ExecutionError:
        if isinstance(node, ChatCompletionNode):
            return await self._execute_chat(request, run, node, step)
        compiled: Final = compile_program(node.program, self._monty.limits)
        if isinstance(compiled, MontyFailure):
            return ExecutionError("invalid_program", compiled.message)
        async with self._monty_start:
            availability: Final = await self._monty.start()
        if availability is not None:
            return ExecutionError("runtime_unavailable", "The Monty runtime is unavailable")
        result: Final = await self._monty.run_program(
            compiled,
            step.input,
            deadline_monotonic=time.monotonic() + max(0, (step.deadline - self._clock()).total_seconds()),
        )
        return (
            ExecutionError(result.kind.value.lower(), result.message)
            if isinstance(result, MontyFailure)
            else ChildCallResult(result.output, 0, 0, None, None)
        )
