from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import JsonValue, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from .compiler import escape_diagnostics
from .models import CompiledProgram, JsonSchema, MontyErrorKind, MontyFailure, MontyLimits, MontySuccess
from .validation import ValidatorProcess

if TYPE_CHECKING:
    from pydantic_monty import AsyncMonty, AsyncMontySession, CollectString, MontyRuntimeError

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True)
class BinaryArtifact:
    path: Path
    sha256: str


def installed_binary() -> BinaryArtifact | MontyFailure:
    try:
        distribution: Final = importlib.metadata.distribution("pydantic-monty-runtime")
        candidates: Final = tuple(file for file in distribution.files or () if file.name in {"monty", "monty.exe"})
        if len(candidates) != 1:
            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "runtime binary missing or ambiguous")
        file: Final = candidates[0]
        if file.hash is None or file.hash.mode != "sha256":
            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "runtime binary has no SHA-256 RECORD hash")
        digest: Final = base64.urlsafe_b64decode(file.hash.value + "=" * (-len(file.hash.value) % 4)).hex()
        return BinaryArtifact(Path(str(distribution.locate_file(file))).resolve(), digest)
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError) as exc:
        return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, escape_diagnostics(str(exc)))


def _monty_json_issue(value: JsonValue, depth: int, max_depth: int) -> MontyErrorKind | None:
    if depth > max_depth:
        return MontyErrorKind.INPUT_TOO_LARGE
    if value is None or type(value) in (str, int, bool):
        return None
    if type(value) is float:
        return None if math.isfinite(value) else MontyErrorKind.INPUT_INVALID
    if type(value) is list:
        return next(
            (issue for item in value if (issue := _monty_json_issue(item, depth + 1, max_depth)) is not None), None
        )
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            return MontyErrorKind.INPUT_INVALID
        return next(
            (
                child_issue
                for item in value.values()
                if (child_issue := _monty_json_issue(item, depth + 1, max_depth)) is not None
            ),
            None,
        )
    return MontyErrorKind.INPUT_INVALID


def validate_json(value: object, *, max_bytes: int, max_depth: int, output: bool = False) -> JsonValue | MontyFailure:
    invalid: Final = MontyErrorKind.OUTPUT_INVALID if output else MontyErrorKind.INPUT_INVALID
    too_large: Final = MontyErrorKind.OUTPUT_TOO_LARGE if output else MontyErrorKind.INPUT_TOO_LARGE
    try:
        parsed: Final = _JSON.validate_python(value, strict=True)
        issue: Final = _monty_json_issue(parsed, 0, max_depth)
        if issue is not None:
            return MontyFailure(
                too_large if issue == MontyErrorKind.INPUT_TOO_LARGE else invalid, "JSON depth or type invalid"
            )
        encoded: Final = json.dumps(parsed, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > max_bytes:
            return MontyFailure(too_large, "JSON byte limit exceeded")
    except PydanticValidationError as exc:
        if '"type":"recursion_loop"' in exc.json():
            return MontyFailure(too_large, "JSON depth limit exceeded")
        return MontyFailure(invalid, escape_diagnostics(str(exc))[:4096])
    except (ValueError, OverflowError, RecursionError) as exc:
        return MontyFailure(invalid, escape_diagnostics(str(exc))[:4096])
    else:
        return parsed


def _kill(session: AsyncMontySession) -> None:
    pid: Final = session.worker_pid
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _worker_limit(session: AsyncMontySession, limits: MontyLimits) -> MontyFailure | None:
    if sys.platform != "linux":
        return None
    import resource

    pid: Final = session.worker_pid
    if pid is None:
        return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "worker PID unavailable")
    try:
        resource.prlimit(
            pid, resource.RLIMIT_AS, (limits.worker_address_space_bytes, limits.worker_address_space_bytes)
        )
    except (OSError, ValueError) as exc:
        return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, escape_diagnostics(str(exc)))
    return None


def _runtime_failure(exc: MontyRuntimeError, collector: CollectString, limits: MontyLimits) -> MontyFailure:
    error: Final = exc.exception()
    message: Final = escape_diagnostics(exc.display())[:4096]
    if isinstance(error, TimeoutError):
        return MontyFailure(MontyErrorKind.TIMEOUT, message)
    if isinstance(error, MemoryError):
        kind: Final = (
            MontyErrorKind.STDOUT_LIMIT
            if len(collector.output.encode("utf-8")) >= limits.max_stdout_bytes and not exc.traceback()
            else MontyErrorKind.MEMORY_LIMIT
        )
        return MontyFailure(kind, message)
    if isinstance(error, RecursionError):
        return MontyFailure(MontyErrorKind.RECURSION_LIMIT, message)
    return MontyFailure(MontyErrorKind.RUNTIME_ERROR, message)


class MontyRuntime:
    def __init__(self, limits: MontyLimits, *, binary: BinaryArtifact | None = None) -> None:
        self.limits = limits
        self._binary = binary
        self._pool: AsyncMonty | None = None
        self._slots = asyncio.Semaphore(limits.max_concurrent_workers)
        self._admitted = 0
        self._validators = tuple(ValidatorProcess() for _ in range(limits.max_concurrent_workers))

    async def start(self) -> MontyFailure | None:
        if self._pool is not None:
            return None
        try:
            from pydantic_monty import AsyncMonty

            artifact: Final = self._binary if self._binary is not None else installed_binary()
            if isinstance(artifact, MontyFailure):
                return artifact
            if hashlib.sha256(artifact.path.read_bytes()).hexdigest() != artifact.sha256:
                return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "runtime binary SHA-256 mismatch")
            pool: Final = AsyncMonty(
                binary_path=artifact.path,
                min_processes=0,
                max_processes=self.limits.max_concurrent_workers,
                max_checkouts_per_worker=1,
                checkout_timeout=self.limits.feed_seconds,
                request_timeout=self.limits.feed_seconds + 1,
                feed_duration_limit_grace=0.1,
            )
            await pool.__aenter__()
            validators: Final = await asyncio.gather(*(worker.start() for worker in self._validators))
            if (failure := next((result for result in validators if result is not None), None)) is not None:
                await asyncio.gather(*(worker.close() for worker in self._validators))
                await pool.__aexit__(None, None, None)
                return failure
            self._pool = pool
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, escape_diagnostics(str(exc)))
        else:
            return None

    async def close(self) -> None:
        pool: Final = self._pool
        self._pool = None
        await asyncio.gather(*(worker.close() for worker in self._validators))
        if pool is not None:
            await pool.__aexit__(None, None, None)

    async def run_program(
        self, compiled: CompiledProgram, input: object, *, deadline_monotonic: float
    ) -> MontySuccess | MontyFailure:
        started: Final = time.monotonic()
        if started >= deadline_monotonic:
            return MontyFailure(MontyErrorKind.TIMEOUT, "node deadline expired")
        if self._pool is None:
            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "Monty runtime is not started")
        if self._admitted >= self.limits.max_concurrent_workers + self.limits.max_queued_waiters:
            return MontyFailure(MontyErrorKind.QUEUE_FULL, "Monty execution queue is full")
        self._admitted += 1
        try:
            async with asyncio.timeout_at(asyncio.get_running_loop().time() + deadline_monotonic - time.monotonic()):
                async with self._slots:
                    worker: Final = next(worker for worker in self._validators if not worker.busy)
                    worker.busy = True
                    try:
                        validated: Final = await self._validate(
                            input, compiled.input_schema, worker, deadline_monotonic
                        )
                        if isinstance(validated, MontyFailure):
                            return validated
                        return await self._execute(compiled, validated, deadline_monotonic, started, worker)
                    except asyncio.CancelledError:
                        await worker.close()
                        raise
                    finally:
                        worker.busy = False
        except TimeoutError:
            return MontyFailure(MontyErrorKind.TIMEOUT, "node deadline expired")
        finally:
            self._admitted -= 1

    async def _execute(
        self, compiled: CompiledProgram, input: JsonValue, deadline: float, started: float, worker: ValidatorProcess
    ) -> MontySuccess | MontyFailure:
        from pydantic_monty import CollectString, MontyCrashedError, MontyRuntimeError, MontySyntaxError, ResourceLimits

        pool: Final = self._pool
        if pool is None:
            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "Monty runtime is closed")
        remaining: Final = deadline - time.monotonic()
        if remaining <= 0:
            return MontyFailure(MontyErrorKind.TIMEOUT, "node deadline expired")
        collector: Final = CollectString(max_bytes=self.limits.max_stdout_bytes)
        session: Final = pool.checkout(
            script_name="workflow.py",
            limits=ResourceLimits(
                max_feed_duration_secs=min(self.limits.feed_seconds, remaining),
                max_memory=self.limits.max_memory,
                max_recursion_depth=self.limits.max_recursion_depth,
            ),
            os_policy={"sleep": "zero"},
        )
        try:
            async with session:
                try:
                    if (limit_failure := _worker_limit(session, self.limits)) is not None:
                        return limit_failure
                    result: Final = _JSON.validate_python(
                        await session.feed_run(
                            compiled.sandbox_source,
                            inputs={"_liteflow_input": input},
                            print_callback=collector,
                        ),
                        strict=True,
                    )
                    return await self._output(result, compiled, collector.output, deadline, started, worker)
                except asyncio.CancelledError:
                    _kill(session)
                    raise
        except MontyRuntimeError as exc:
            return _runtime_failure(exc, collector, self.limits)
        except MontyCrashedError as exc:
            return MontyFailure(
                MontyErrorKind.TIMEOUT if exc.timed_out else MontyErrorKind.WORKER_CRASHED,
                escape_diagnostics(str(exc))[:4096],
            )
        except MontySyntaxError as exc:
            return MontyFailure(MontyErrorKind.INVALID_PROGRAM, escape_diagnostics(str(exc))[:4096])
        except (ValueError, OSError) as exc:
            return MontyFailure(MontyErrorKind.RUNTIME_ERROR, escape_diagnostics(str(exc))[:4096])

    async def _output(
        self,
        result: object,
        compiled: CompiledProgram,
        stdout: str,
        deadline: float,
        started: float,
        worker: ValidatorProcess,
    ) -> MontySuccess | MontyFailure:
        try:
            envelope: Final = _JSON.validate_python(result, strict=True)
        except ValueError:
            return MontyFailure(MontyErrorKind.OUTPUT_INVALID, "return value is not strict JSON")
        if not isinstance(envelope, list) or len(envelope) != 2 or envelope[0] is not True:
            return MontyFailure(MontyErrorKind.OUTPUT_INVALID, "return value is not strict JSON")
        output: Final = await self._validate(envelope[1], compiled.output_schema, worker, deadline, output=True)
        if isinstance(output, MontyFailure):
            return output
        if time.monotonic() >= deadline:
            return MontyFailure(MontyErrorKind.TIMEOUT, "node deadline expired during validation")
        return MontySuccess(output, escape_diagnostics(stdout), int((time.monotonic() - started) * 1000))

    async def _validate(
        self, value: object, schema: JsonSchema, worker: ValidatorProcess, deadline: float, *, output: bool = False
    ) -> JsonValue | MontyFailure:
        parsed: Final = validate_json(
            value,
            max_bytes=self.limits.max_output_bytes if output else self.limits.max_input_bytes,
            max_depth=self.limits.max_output_depth if output else self.limits.max_input_depth,
            output=output,
        )
        if isinstance(parsed, MontyFailure):
            return parsed
        failure: Final = await worker.validate(parsed, schema, deadline=deadline, output=output)
        return failure if failure is not None else parsed

    async def evaluate_predicate(
        self, compiled: CompiledProgram, input: object, *, deadline_monotonic: float
    ) -> bool | MontyFailure:
        result: Final = await self.run_program(compiled, input, deadline_monotonic=deadline_monotonic)
        if isinstance(result, MontyFailure):
            return result
        if type(result.output) is not bool:
            return MontyFailure(MontyErrorKind.PREDICATE_NOT_BOOL, "predicate must return exactly bool")
        return result.output
